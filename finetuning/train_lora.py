"""
Training LoRA/QLoRA per Gemma 4 E2B sul dataset text-to-SPARQL.
Segue la ricetta ufficiale Unsloth per Gemma 4 (unsloth.ai/docs/models/gemma-4/train),
non una configurazione generica: nome modello, classe di caricamento, chat template,
mascheramento della loss sul prompt e fix del doppio token BOS sono tutti presi da li'.

Pensato per girare senza supervisione su HPC: salva checkpoint periodici e non
rifa' il training se e' gia' stato completato in un run precedente.
"""
import json
import sys
from pathlib import Path

from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

MODEL_NAME = "unsloth/gemma-4-E2B-it"
MAX_SEQ_LENGTH = 2048  # il massimo reale nei dati e' 1770 token (verificato con smoke_test_tokenizer.py); margine incluso
OUTPUT_DIR = "gemma4-sparql-lora"
FINAL_DIR = f"{OUTPUT_DIR}-final"


def require_file(path: str):
    if not Path(path).exists():
        sys.exit(f"ERRORE: manca il file richiesto '{path}'. "
                  f"Verifica di averlo copiato nella cartella prima di lanciare il job.")


def convert_to_chat_format(input_path: str, output_path: str):
    """Rigenera sempre da zero, non riusa un file esistente: evita il
    rischio di allenare su una versione vecchia se il dataset e' cambiato
    ma il file convertito era rimasto da un tentativo precedente."""
    require_file(input_path)
    n_written, n_skipped = 0, 0
    with open(input_path, encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("fallback"):
                n_skipped += 1
                continue
            chat_row = {"conversations": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["query"]},
            ]}
            f_out.write(json.dumps(chat_row, ensure_ascii=False) + "\n")
            n_written += 1
    print(f"{input_path} -> {output_path}: {n_written} righe, {n_skipped} fallback escluse")
    if n_written == 0:
        sys.exit(f"ERRORE: '{input_path}' non ha prodotto nessuna riga valida.")


def main():
    # Se un run precedente ha gia' completato il training, non rifarlo:
    # evita di sovrascrivere un modello buono se il job viene rilanciato
    # per errore.
    if Path(f"{FINAL_DIR}/TRAINING_COMPLETE").exists():
        print(f"Training gia' completato in precedenza ({FINAL_DIR}). "
              f"Cancella quella cartella se vuoi rifarlo da capo. Esco senza fare nulla.")
        return

    convert_to_chat_format("train_augmented.jsonl", "train_chat.jsonl")
    convert_to_chat_format("val_augmented.jsonl", "val_chat.jsonl")

    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        full_finetuning=False,
    )

    # Gemma 4 esiste in variante "thinking" e "non-thinking": per un compito
    # di traduzione diretta prompt->query (nessun ragionamento step-by-step
    # da mostrare), la doc ufficiale raccomanda la variante non-thinking
    # per i modelli piccoli come E2B.
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

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

    dataset = load_dataset("json", data_files={
        "train": "train_chat.jsonl",
        "validation": "val_chat.jsonl",
    })

    def formatting_prompts_func(examples):
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            ).removeprefix("<bos>")  # evita il doppio token BOS
            for convo in examples["conversations"]
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # Controllo di sicurezza: se questi marcatori non compaiono davvero nel
    # testo formattato, train_on_responses_only mascera la loss nel modo
    # sbagliato SENZA sollevare un errore -- meglio bloccarsi qui con un
    # messaggio chiaro che scoprirlo dopo ore di training su HPC.
    instruction_part = "<|turn>user\n"
    response_part = "<|turn>model\n"
    sample_text = dataset["train"][0]["text"]
    if instruction_part not in sample_text or response_part not in sample_text:
        sys.exit(
            f"ERRORE: i marcatori del chat template non corrispondono a quelli attesi.\n"
            f"Atteso instruction_part={instruction_part!r}, response_part={response_part!r}\n"
            f"Testo formattato reale:\n{sample_text}\n"
            f"Lancia prima smoke_test_tokenizer.py in locale per vedere il formato "
            f"esatto e correggi le due variabili qui sopra di conseguenza."
        )

    resume = Path(OUTPUT_DIR).exists() and any(Path(OUTPUT_DIR).glob("checkpoint-*"))

    # bf16 non e' supportato bene su GPU Turing (es. RTX 2080, stessa
    # generazione della GTX 1650 usata nei test locali) -- rilevo il
    # supporto invece di darlo per scontato, per evitare lo stesso problema
    # di 'Using float16 precision... won't work!' visto in locale.
    import torch
    bf16_supported = torch.cuda.is_bf16_supported()
    print(f"Supporto bf16 rilevato su questo hardware: {bf16_supported} "
          f"(se False, uso fp16 al suo posto)")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=SFTConfig(
            dataset_text_field="text",
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=2e-4,
            warmup_ratio=0.03,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            bf16=bf16_supported,
            fp16=not bf16_supported,
            max_seq_length=MAX_SEQ_LENGTH,
            seed=3407,
            report_to="none",
        ),
    )

    # Allena solo sulla risposta dell'assistente (la query), non sul
    # prompt dell'utente -- altrimenti il modello spreca capacita' anche
    # a "imparare" a riprodurre le domande, non solo le risposte.
    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruction_part,
        response_part=response_part,
    )

    print("NOTA: per Gemma 4 E2B/E4B una loss iniziale/di plateau intorno "
          "a 13-15 e' normale (particolarita' nota di questi modelli "
          "multimodali), non un segno di malfunzionamento. E' un problema "
          "solo se la loss e' molto piu' alta (es. 100+).")

    trainer.train(resume_from_checkpoint=resume)

    model.save_pretrained(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)

    # File sentinella: dice allo script SLURM che il training e' DAVVERO
    # completo, non solo che il job e' terminato (per timeout, es.).
    Path(f"{FINAL_DIR}/TRAINING_COMPLETE").touch()
    print("Training completato con successo.")


if __name__ == "__main__":
    main()