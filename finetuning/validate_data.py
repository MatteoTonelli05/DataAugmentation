"""
Controllo dei tre file prima di mandare qualunque cosa su HPC.
Nessuna dipendenza esterna (solo libreria standard) -- lanciabile ovunque,
anche sul portatile, in pochi secondi.

Uso: python validate_data.py
"""
import json
import sys
from pathlib import Path

FILES = ["train_augmented.jsonl", "val_augmented.jsonl", "test_augmented.jsonl"]
REQUIRED_KEYS = {"prompt", "query"}


def validate_file(path: str) -> bool:
    if not Path(path).exists():
        print(f"[ERRORE] '{path}' non esiste in questa cartella.")
        return False

    ok = True
    n_lines = 0
    n_fallback = 0
    empty_prompt = 0
    empty_query = 0
    missing_keys = 0
    bad_json = []
    source_ids = set()
    duplicate_prompts = {}

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                bad_json.append((i, str(e)))
                continue

            if not REQUIRED_KEYS.issubset(row.keys()):
                missing_keys += 1
                continue
            if not row["prompt"].strip():
                empty_prompt += 1
            if not row["query"].strip():
                empty_query += 1
            if row.get("fallback"):
                n_fallback += 1

            sid = row.get("source_id")
            if sid is not None:
                source_ids.add(sid)

            duplicate_prompts[row["prompt"]] = duplicate_prompts.get(row["prompt"], 0) + 1

    n_duplicates = sum(1 for c in duplicate_prompts.values() if c > 1)

    print(f"--- {path} ---")
    print(f"  righe totali: {n_lines}")
    print(f"  record sorgente distinti: {len(source_ids)}")
    print(f"  righe fallback (verranno escluse dal training): {n_fallback}")
    print(f"  prompt duplicati: {n_duplicates}")

    if bad_json:
        ok = False
        print(f"  [ERRORE] {len(bad_json)} righe con JSON non valido:")
        for line_no, err in bad_json[:5]:
            print(f"    riga {line_no}: {err}")
        if len(bad_json) > 5:
            print(f"    ... e altre {len(bad_json) - 5}")

    if missing_keys:
        ok = False
        print(f"  [ERRORE] {missing_keys} righe senza le chiavi richieste {REQUIRED_KEYS}")

    if empty_prompt:
        ok = False
        print(f"  [ERRORE] {empty_prompt} righe con prompt vuoto")

    if empty_query:
        ok = False
        print(f"  [ERRORE] {empty_query} righe con query vuota")

    if n_lines == 0:
        ok = False
        print(f"  [ERRORE] il file e' vuoto")

    print()
    return ok


def main():
    all_ok = True
    for f in FILES:
        if not validate_file(f):
            all_ok = False

    print("=" * 50)
    if all_ok:
        print("Tutti i controlli superati. I dati sono pronti.")
    else:
        print("Trovati problemi -- risolvili prima di mandare qualunque cosa su HPC.")
        sys.exit(1)


if __name__ == "__main__":
    main()
