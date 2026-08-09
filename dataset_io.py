import json
from pathlib import Path

from logging_utils import LINE
from styles import opener

def load_records(input_path: Path) -> list:
    return [json.loads(l) for l in input_path.read_text(encoding="utf-8").splitlines() if l.strip()]

def done_ids(output_path: Path) -> set:
    if not output_path.exists():
        return set()
    with output_path.open(encoding="utf-8") as f:
        return {
            entry["source_id"]
            for line in f
            if line.strip()
            for entry in [json.loads(line)]
        }

def append_entries(out_f, entries: list, source_id: int, query: str) -> None:
    for text, style_tag, used_fallback in entries:
        out_f.write(json.dumps({
            "prompt": text,
            "query": query,
            "source_id": source_id,
            "style": style_tag,
            "fallback": used_fallback,
        }, ensure_ascii=False) + "\n")
    out_f.flush()

def diversity_report(output_path: Path) -> None:
    if not output_path.exists():
        return
    entries = [json.loads(l) for l in output_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    openers = [opener(e["prompt"]) for e in entries]
    distinct_ratio = len(set(openers)) / len(openers) if openers else 0

    style_counts = {}
    for e in entries:
        style_counts[e.get("style", "unknown")] = style_counts.get(e.get("style", "unknown"), 0) + 1

    print(f"\n{LINE}")
    print("FINAL REPORT")
    print(LINE)
    print(f"Total generated lines : {len(entries)}")
    print(f"Distinct openings     : {len(set(openers))}/{len(openers)}  ({distinct_ratio:.1%})")
    if distinct_ratio < 0.4:
        print("  -> LOW: the augmentation is re-templating, consider revisiting the styles")
    print("Style distribution    :")
    for style, count in sorted(style_counts.items(), key=lambda x: -x[1]):
        print(f"  {style:<16} {count}")