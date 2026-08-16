"""
Stesso identico schema di codice di smoke_test_gpu.py (LoRA, training,
generazione) ma con un modello minuscolo (Qwen2.5 0.5B) invece di Gemma 4
E2B, cosi' la GTX 1650 non va in out-of-memory e possiamo verificare per
davvero che la LOGICA del training funzioni -- LoRA, mascheramento della
loss sul prompt, ciclo di training, generazione con use_cache.

Se questo test passa, lo stesso pattern di codice (che e' quello usato
davvero in train_lora.py, solo con un modello diverso) dovrebbe funzionare
anche con Gemma 4 E2B su HPC, dove la memoria non e' un vincolo.

NOTA: i marcatori del chat template qui sono quelli di Qwen (ChatML,
"<|im_start|>..."), DIVERSI da quelli di Gemma 4 usati in train_lora.py
("<|turn>..."). E' normale e atteso -- stiamo testando la logica del
codice, non la stringa esatta (quella l'abbiamo gia' verificata a parte
con smoke_test_tokenizer.py).

Uso: python smoke_test_gpu_tiny_model.py
"""
import json
import sys
import traceback

MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct"
N_SAMPLE_ROWS = 20
MAX_STEPS = 5
MAX_SEQ_LENGTH = 512
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"


def main():
    print("--- Passo 1: import di Unsloth ---")
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import train_on_responses_only
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset
        import torch
    except ImportError as e:
        sys.exit(f"[ERRORE] Import fallito: {e}")
    print("[OK]\n")

    print("--- Passo 2: preparo un mini-campione dai dati reali ---")
    try:
        rows = [json.loads(l) for l in open("train_augmented.jsonl", encoding="utf-8") if l.strip()]
    except FileNotFoundError:
        sys.exit("[ERRORE] 'train_augmented.jsonl' non trovato in questa cartella.")
    sample = [r for r in rows if not r.get("fallback")][:N_SAMPLE_ROWS]
    if not sample:
        sys.exit("[ERRORE] Nessuna riga utilizzabile trovata nel dataset.")
    print(f"[OK] Uso {len(sample)} righe.\n")

    print(f"--- Passo 3: carico {MODEL_NAME} (piccolo, dovrebbe entrare comodo in 4GB) ---")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
        )
    except Exception as e:
        print(f"[ERRORE] Caricamento modello fallito: {e}")
        traceback.print_exc()
        sys.exit(1)
    print("[OK]\n")

    print("--- Passo 4: verifico i marcatori del chat template ---")
    sample_formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": "test"}, {"role": "assistant", "content": "risposta"}],
        tokenize=False, add_generation_prompt=False,
    )
    print(sample_formatted)
    if INSTRUCTION_PART not in sample_formatted or RESPONSE_PART not in sample_formatted:
        sys.exit(f"[ERRORE] Marcatori attesi non trovati. Guarda l'output sopra "
                  f"e correggi INSTRUCTION_PART/RESPONSE_PART in questo script.")
    print("[OK]\n")

    print("--- Passo 5: aggiungo gli adapter LoRA ---")
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
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
            for convo in examples["conversations"]
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    print("[OK]\n")

    print(f"--- Passo 7: {MAX_STEPS} passi di training vero (batch=1) ---")
    bf16_supported = torch.cuda.is_bf16_supported()
    print(f"    (bf16 supportato: {bf16_supported})")
    try:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=SFTConfig(
                dataset_text_field="text",
                output_dir="smoke_test_tiny_output",
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                max_steps=MAX_STEPS,
                learning_rate=2e-4,
                logging_steps=1,
                optim="adamw_8bit",
                bf16=bf16_supported,
                fp16=not bf16_supported,
                max_seq_length=MAX_SEQ_LENGTH,
                report_to="none",
            ),
        )
        trainer = train_on_responses_only(
            trainer,
            instruction_part=INSTRUCTION_PART,
            response_part=RESPONSE_PART,
        )
        trainer.train()
    except Exception as e:
        print(f"[ERRORE] Training fallito: {e}")
        traceback.print_exc()
        sys.exit(1)
    print("[OK] Training completato senza errori.\n")

    print("--- Passo 8: generazione con use_cache=True ---")
    try:
        FastLanguageModel.for_inference(model)
        messages = [{"role": "user", "content": sample[0]["prompt"]}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        output = model.generate(**inputs, max_new_tokens=64, do_sample=False, use_cache=True)
        text = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        print(f"Output generato: {text[:300]}")
    except Exception as e:
        print(f"[ERRORE] Generazione fallita: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\n=== Tutti i passi completati senza errori. ===")
    print("=== La logica di LoRA/training/generazione e' verificata. ===")
    print("=== Il codice equivalente in train_lora.py dovrebbe funzionare su HPC. ===")


if __name__ == "__main__":
    main()
