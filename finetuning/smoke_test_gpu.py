"""
Mini training reale (pochi passi, non l'addestramento completo) sulla tua
GPU locale. Obiettivo: far girare TUTTO il codice vero -- caricamento
modello, chat template, LoRA, training, generazione -- almeno una volta,
per beccare bug di codice (nomi sbagliati, API cambiate, errori di
battitura) PRIMA di consumare tempo di coda HPC per scoprirli la'.

Con 4GB di VRAM (GTX 1650) e' molto probabile che questo vada comunque in
out-of-memory anche solo con pochi passi -- Gemma 4 E2B richiede
ufficialmente 8-10GB anche per LoRA. Se succede, NON significa che il job
HPC fallira' (li' hai molta piu' VRAM): significa solo che qui non puoi
verificare la memoria, ma puoi comunque verificare che il codice non abbia
errori fino al punto in cui la memoria finisce.

Uso: python smoke_test_gpu.py
"""
import json
import sys
import traceback

N_SAMPLE_ROWS = 20   # bastano pochissime righe, non serve tutto il dataset
MAX_STEPS = 3        # solo per vedere che il ciclo di training parte e avanza
MAX_SEQ_LENGTH = 256  # ridotto apposta per questo test, non e' il valore finale


def main():
    print("--- Passo 1: import di Unsloth ---")
    try:
        from unsloth import FastModel
        from unsloth.chat_templates import get_chat_template, train_on_responses_only
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset
    except ImportError as e:
        sys.exit(f"[ERRORE] Import fallito: {e}\n"
                  f"Hai installato con 'pip install unsloth trl datasets --break-system-packages'?")
    print("[OK] Import riusciti.\n")

    print("--- Passo 2: preparo un mini-campione dai dati reali ---")
    try:
        rows = [json.loads(l) for l in open("train_augmented.jsonl", encoding="utf-8") if l.strip()]
    except FileNotFoundError:
        sys.exit("[ERRORE] 'train_augmented.jsonl' non trovato in questa cartella.")

    sample = [r for r in rows if not r.get("fallback")][:N_SAMPLE_ROWS]
    if not sample:
        sys.exit("[ERRORE] Nessuna riga utilizzabile trovata nel dataset.")
    print(f"[OK] Uso {len(sample)} righe per il test.\n")

    print("--- Passo 3: carico il modello in 4-bit (puo' volerci un po' al primo scaricamento) ---")
    try:
        model, tokenizer = FastModel.from_pretrained(
            model_name="unsloth/gemma-4-E2B-it",
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
            full_finetuning=False,
        )
    except Exception as e:
        print(f"[ERRORE] Caricamento modello fallito: {e}")
        traceback.print_exc()
        sys.exit(1)
    print("[OK] Modello caricato.\n")

    print("--- Passo 4: applico il chat template ---")
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
    print("[OK]\n")

    print("--- Passo 5: aggiungo gli adapter LoRA ---")
    try:
        model = FastModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=16,
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            random_state=3407,
        )
    except Exception as e:
        print(f"[ERRORE] Setup LoRA fallito: {e}")
        traceback.print_exc()
        sys.exit(1)
    print("[OK]\n")

    print("--- Passo 6: preparo il mini-dataset ---")
    conversations = [
        {"conversations": [
            {"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": r["query"]},
        ]}
        for r in sample
    ]
    dataset = Dataset.from_list(conversations)

    def formatting_prompts_func(examples):
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            .removeprefix("<bos>")
            for convo in examples["conversations"]
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    print("[OK]\n")

    print(f"--- Passo 7: provo {MAX_STEPS} passi di training vero (batch=1) ---")
    try:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=SFTConfig(
                dataset_text_field="text",
                output_dir="smoke_test_output",
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                max_steps=MAX_STEPS,
                learning_rate=2e-4,
                logging_steps=1,
                optim="adamw_8bit",
                max_seq_length=MAX_SEQ_LENGTH,
                report_to="none",
            ),
        )
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|turn>user\n",
            response_part="<|turn>model\n",
        )
        trainer.train()
    except Exception as e:
        if "out of memory" in str(e).lower():
            print(f"\n[OOM ATTESO] Memoria esaurita: {e}")
            print("Questo e' plausibile con 4GB di VRAM (Gemma 4 E2B ne richiede")
            print("ufficialmente 8-10GB). NON e' necessariamente un problema di")
            print("codice -- se sei arrivato fin qui (import, caricamento modello,")
            print("chat template, LoRA, avvio del training tutti riusciti), il")
            print("codice e' verosimilmente corretto e il job HPC (con piu' VRAM)")
            print("dovrebbe funzionare.")
            sys.exit(0)
        else:
            print(f"[ERRORE] Training fallito per un motivo diverso da OOM: {e}")
            traceback.print_exc()
            sys.exit(1)
    print("[OK] Training di prova completato senza errori.\n")

    print("--- Passo 8: provo una generazione (verifica use_cache) ---")
    try:
        FastModel.for_inference(model)
        messages = [{"role": "user", "content": sample[0]["prompt"]}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        output = model.generate(**inputs, max_new_tokens=64, do_sample=False, use_cache=True)
        text = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        print(f"Output generato: {text[:200]}")
    except Exception as e:
        print(f"[ERRORE] Generazione fallita: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\n=== Tutti i passi completati senza errori di codice. ===")
    print("=== Il job HPC dovrebbe partire senza problemi di questo tipo. ===")


if __name__ == "__main__":
    main()
