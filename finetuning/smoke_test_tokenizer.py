"""
Verifica il chat template e la lunghezza delle sequenze SENZA bisogno di GPU
o del pacchetto Unsloth completo -- scarica solo il tokenizer (pochi MB),
non i pesi del modello (che sono GB). Puoi lanciarlo su qualunque macchina
con connessione internet, anche il portatile.

Cosa controlla, nell'ordine:
1. Che il chat template produca davvero i marcatori "<|turn>user" e
   "<|turn>model" che train_lora.py usa per mascherare la loss sul prompt.
   Se questi non corrispondono, il mascheramento fallisce silenziosamente
   (nessun errore, ma il training non fa quello che dovrebbe).
2. La distribuzione delle lunghezze (in token) delle tue query reali, per
   verificare che MAX_SEQ_LENGTH=1024 sia sufficiente e non tronchi nulla
   di nascosto.

Richiede: pip install transformers --break-system-packages
(NON serve torch con supporto CUDA, NON serve unsloth per questo controllo)

Uso: python smoke_test_tokenizer.py
"""
import json
import sys

try:
    from transformers import AutoTokenizer
except ImportError:
    sys.exit("Manca 'transformers'. Installa con: pip install transformers --break-system-packages")

MODEL_NAME = "unsloth/gemma-4-E2B-it"
MAX_SEQ_LENGTH = 2048
INSTRUCTION_MARKER = "<|turn>user\n"
RESPONSE_MARKER = "<|turn>model\n"


def main():
    print(f"Scarico il tokenizer di {MODEL_NAME} (solo tokenizer, non i pesi -- pochi MB)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # --- 1. Verifica dei marcatori usati da train_on_responses_only ---
    sample_conversation = [
        {"role": "user", "content": "Which patients scored 4 on ECOG?"},
        {"role": "assistant", "content": "SELECT ?patient WHERE { ?patient a <...> }"},
    ]
    formatted = tokenizer.apply_chat_template(
        sample_conversation, tokenize=False, add_generation_prompt=False
    )

    print("\n--- Output del chat template su un esempio ---")
    print(formatted)
    print("--- fine esempio ---\n")

    problems = False
    if INSTRUCTION_MARKER not in formatted:
        problems = True
        print(f"[ERRORE] Il marcatore '{INSTRUCTION_MARKER!r}' NON compare nell'output sopra.")
        print("         train_lora.py usa questa stringa per capire dove inizia il")
        print("         prompt dell'utente -- se non corrisponde, il mascheramento")
        print("         della loss non funzionera' come previsto. Guarda l'output")
        print("         sopra e correggi INSTRUCTION_MARKER in train_lora.py con la")
        print("         stringa esatta che vedi li'.")
    else:
        print(f"[OK] Il marcatore '{INSTRUCTION_MARKER!r}' e' presente.")

    if RESPONSE_MARKER not in formatted:
        problems = True
        print(f"[ERRORE] Il marcatore '{RESPONSE_MARKER!r}' NON compare nell'output sopra.")
        print("         Stesso discorso del marcatore precedente, per la risposta.")
    else:
        print(f"[OK] Il marcatore '{RESPONSE_MARKER!r}' e' presente.")

    # --- 2. Distribuzione delle lunghezze in token sui dati reali ---
    print("\n--- Lunghezza delle sequenze sui tuoi dati reali ---")
    for fname in ["train_augmented.jsonl", "val_augmented.jsonl", "test_augmented.jsonl"]:
        try:
            rows = [json.loads(l) for l in open(fname, encoding="utf-8") if l.strip()]
        except FileNotFoundError:
            print(f"  {fname}: non trovato, salto")
            continue

        lengths = []
        for row in rows:
            convo = [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["query"]},
            ]
            text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            lengths.append(len(tokenizer.encode(text)))

        lengths.sort()
        n = len(lengths)
        p50 = lengths[n // 2]
        p95 = lengths[int(n * 0.95)]
        p99 = lengths[int(n * 0.99)]
        n_over = sum(1 for l in lengths if l > MAX_SEQ_LENGTH)

        print(f"  {fname}: {n} righe -- mediana={p50} token, "
              f"95° percentile={p95}, 99° percentile={p99}, massimo={max(lengths)}")
        if n_over:
            problems = True
            print(f"    [ERRORE] {n_over} righe superano MAX_SEQ_LENGTH={MAX_SEQ_LENGTH} "
                  f"e verrebbero troncate silenziosamente. Alza MAX_SEQ_LENGTH in "
                  f"train_lora.py ed evaluate.py ad almeno {max(lengths)}.")

    print()
    if problems:
        print("Trovati problemi -- sistemali prima di mandare il job su HPC.")
        sys.exit(1)
    else:
        print("Tutto ok: marcatori corretti, nessuna sequenza troncata.")


if __name__ == "__main__":
    main()