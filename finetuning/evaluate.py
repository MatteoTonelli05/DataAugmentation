"""
Valutazione del modello fine-tuned sul test set.
Gira automaticamente dopo il training (vedi lo script SLURM). Produce:
  - evaluation_report.md   -> riepilogo leggibile, pronto da guardare/incollare nella tesi
  - evaluation_raw.jsonl   -> ogni esempio con predizione, per analisi piu' fine

NOTA IMPORTANTE: use_cache=True e' impostato ESPLICITAMENTE nella chiamata a
generate(). Per Gemma 4 E2B/E4B (che condividono la KV-cache tra alcuni layer),
generare con use_cache=False -- che e' cio' che il training con gradient
checkpointing lascia come stato di default -- produce testo ILLEGGIBILE, non
solo "sbagliato". Senza questo fix, questo script avrebbe segnalato un
modello rotto anche se il training fosse andato benissimo.
"""
import json
import re
import sys
from pathlib import Path

from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

MODEL_DIR = "gemma4-sparql-lora-final"
TEST_FILE = "test_augmented.jsonl"
KG_API_URL = None  # es. "http://localhost:7202" -- lascia None se non raggiungibile da HPC


def normalize(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip())


def looks_syntactically_valid(query: str) -> bool:
    q = query.strip()
    if not re.match(r"^(SELECT|ASK|CONSTRUCT)\b", q, re.IGNORECASE):
        return False
    return q.count("{") == q.count("}") and q.count("{") > 0


def try_execute(query: str):
    if not KG_API_URL:
        return None
    import requests
    try:
        resp = requests.post(f"{KG_API_URL}/query", json={"query": query}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def main():
    if not Path(MODEL_DIR).exists():
        sys.exit(f"ERRORE: '{MODEL_DIR}' non esiste. Il training e' stato completato?")
    if not Path(TEST_FILE).exists():
        sys.exit(f"ERRORE: manca '{TEST_FILE}'. Copialo nella cartella prima di lanciare.")

    test_rows = [json.loads(l) for l in open(TEST_FILE, encoding="utf-8") if l.strip()]
    if not test_rows:
        sys.exit(f"ERRORE: '{TEST_FILE}' esiste ma non contiene righe valide.")

    model, tokenizer = FastModel.from_pretrained(MODEL_DIR, max_seq_length=2048)
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
    FastModel.for_inference(model)

    results = []
    n_exact = 0
    n_valid_syntax = 0
    n_exec_match = 0
    n_exec_attempted = 0

    for row in test_rows:
        messages = [{"role": "user", "content": row["prompt"]}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,   # <-- il fix critico, vedi nota in cima al file
        )
        predicted = tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        ).strip()

        exact = normalize(predicted) == normalize(row["query"])
        valid_syntax = looks_syntactically_valid(predicted)

        exec_match = None
        if KG_API_URL and valid_syntax:
            n_exec_attempted += 1
            pred_results = try_execute(predicted)
            true_results = try_execute(row["query"])
            if pred_results is not None and true_results is not None:
                exec_match = pred_results == true_results
                if exec_match:
                    n_exec_match += 1

        if exact:
            n_exact += 1
        if valid_syntax:
            n_valid_syntax += 1

        results.append({
            "source_id": row.get("source_id"),
            "prompt": row["prompt"],
            "expected_query": row["query"],
            "predicted_query": predicted,
            "exact_match": exact,
            "syntactically_valid": valid_syntax,
            "execution_match": exec_match,
        })

    n = len(test_rows)
    with open("evaluation_raw.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    correct_examples = [r for r in results if r["exact_match"]][:3]
    wrong_examples = [r for r in results if not r["exact_match"]][:3]

    with open("evaluation_report.md", "w", encoding="utf-8") as f:
        f.write("# Report di valutazione\n\n")
        f.write(f"Esempi di test valutati: {n}\n\n")
        f.write("## Metriche\n\n")
        f.write(f"- Exact match: {n_exact}/{n} ({100*n_exact/n:.1f}%)\n")
        f.write(f"- Sintassi SPARQL valida: {n_valid_syntax}/{n} ({100*n_valid_syntax/n:.1f}%)\n")
        if KG_API_URL:
            f.write(f"- Match di esecuzione (tentati): {n_exec_attempted}\n")
            f.write(f"- Match di esecuzione (riusciti): {n_exec_match}\n")
        else:
            f.write("- Match di esecuzione: non calcolato (KG_API_URL non impostato)\n")
        f.write("\n## Esempi corretti\n\n")
        for r in correct_examples:
            f.write(f"**Prompt:** {r['prompt']}\n\n")
            f.write(f"**Query generata:** `{r['predicted_query']}`\n\n---\n\n")
        f.write("\n## Esempi sbagliati\n\n")
        for r in wrong_examples:
            f.write(f"**Prompt:** {r['prompt']}\n\n")
            f.write(f"**Attesa:** `{r['expected_query']}`\n\n")
            f.write(f"**Generata:** `{r['predicted_query']}`\n\n---\n\n")

    print(f"Fatto. Exact match: {n_exact}/{n} ({100*n_exact/n:.1f}%)")
    print("Vedi evaluation_report.md ed evaluation_raw.jsonl")

    # controllo di sanita': se la maggioranza delle predizioni e' vuota o
    # palesemente illeggibile, e' quasi certamente il bug use_cache -- meglio
    # segnalarlo chiaramente che lasciare un report silenzioso e fuorviante.
    empty_or_short = sum(1 for r in results if len(r["predicted_query"]) < 10)
    if empty_or_short > n * 0.5:
        print(f"ATTENZIONE: {empty_or_short}/{n} predizioni sono vuote o "
              f"molto corte. Se il testo generato sembra illeggibile, "
              f"verifica che use_cache=True sia stato applicato davvero "
              f"(potrebbe essere un problema di versione di Unsloth/transformers).")


if __name__ == "__main__":
    main()