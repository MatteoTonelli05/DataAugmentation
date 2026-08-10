import argparse
import time
from pathlib import Path

from dataset_io import append_entries, diversity_report, done_ids, load_records
from generation import generate_variant
from logging_utils import LINE, log_record_footer, log_record_header, log_rejected
from masking import mask, strip_label_repetition, unmask
from qa_stats import QAStats
from styles import StyleTracker, opener
from validation import find_similar_text, validation_reason

def main():
    ap = argparse.ArgumentParser(description="Local data augmentation for text-to-SPARQL dataset, via Ollama")
    ap.add_argument("--input", required=True, help="input .jsonl file, with a value_spans column")
    ap.add_argument("--output", required=True, help="output .jsonl file (append, resumes where it left off)")
    ap.add_argument("--model", default="gemma4:e2b", help="Ollama model tag to use")
    ap.add_argument("--n-per-example", type=int, default=5, help="how many variants to generate per example")
    ap.add_argument("--max-attempts", type=int, default=5, help="max attempts for each piece of text")
    ap.add_argument("--base-temperature", type=float, default=0.6, help="temperature of the first attempt")
    ap.add_argument("--max-temperature", type=float, default=0.95, help="temperature of the last attempt")
    ap.add_argument("--split-threshold", type=int, default=12, help="above how many values to split the example in half")
    ap.add_argument("--language", default="English", help="generation language, e.g. English or Italian")
    ap.add_argument("--stats-output", default=None,
                     help="json file where the run QA stats are saved "
                          "(default: <output>.stats.json). Updated after each record, "
                          "and reloaded if it already exists (useful when resuming an interrupted run).")
    args = ap.parse_args()

    input_path, output_path = Path(args.input), Path(args.output)
    stats_path = Path(args.stats_output) if args.stats_output else output_path.with_suffix(".stats.json")

    records = load_records(input_path)

    already = done_ids(output_path)
    todo = [r for r in records if r["id"] not in already]
    print(LINE)
    print(f"Dataset: {len(records)} total  |  {len(already)} already done  |  {len(todo)} to process")

    tracker = StyleTracker()
    stats = QAStats()
    stats.load(stats_path)

    with output_path.open("a", encoding="utf-8") as out_f:
        for record in todo:
            masked_text, mapping = mask(record["prompt"], record.get("value_spans", []))
            if not mapping:
                continue
            masked_text = strip_label_repetition(masked_text)

            log_record_header(record["id"], len(mapping), args.n_per_example)

            record_start = time.perf_counter()
            accepted = []

            for slot in range(args.n_per_example):
                stats.record_slot()

                outer_attempts = 3
                avoid_text, avoid_reason = None, None
                slot_succeeded = False
                slot_attempts_log = []

                for _outer in range(outer_attempts):
                    style_tag, style_desc = tracker.next_style()
                    avoid_history = tracker.recent_texts(style_tag)
                    slot_label = f"slot {slot + 1}/{args.n_per_example} [{style_tag}]"
                    stats.record_attempt(style_tag)

                    candidate, _elapsed, used_fallback, opening_text = generate_variant(
                        args.model, masked_text, mapping, style_tag, style_desc,
                        args.split_threshold, args.max_attempts,
                        args.base_temperature, args.max_temperature,
                        avoid_history, slot_label,
                        avoid_text, avoid_reason, args.language,
                    )

                    reason = validation_reason(candidate, mapping, [t for t, _s, _f in accepted])
                    if reason is None and tracker.opener_used_recently(style_tag, opening_text):
                        reason = (
                            f"this opening (\"{opener(opening_text)}...\") was already used recently "
                            f"for the '{style_tag}' style - start with different words"
                        )
                    if reason is None:
                        accepted.append((candidate, style_tag, used_fallback))
                        tracker.remember(style_tag, candidate)
                        stats.record_accepted(style_tag, used_fallback)
                        slot_succeeded = True
                        break
                    else:
                        log_rejected(1, f"{slot_label} (combined)", reason, candidate)
                        slot_attempts_log.append({"style": style_tag, "reason": reason})
                        similar_text, ratio = find_similar_text(candidate, [t for t, _s, _f in accepted])
                        if similar_text:
                            avoid_text = similar_text
                            avoid_reason = f"too similar ({ratio:.0%} match) to this previously accepted variant for the same request"
                        else:
                            avoid_text, avoid_reason = candidate, reason

                if not slot_succeeded:
                    stats.record_slot_failure(record["id"], slot, slot_attempts_log)

            record_elapsed = time.perf_counter() - record_start

            final_entries = [
                (unmask(text, mapping), style_tag, used_fallback)
                for text, style_tag, used_fallback in accepted
            ]
            append_entries(out_f, final_entries, record["id"], record["query"])

            log_record_footer(record["id"], len(accepted), args.n_per_example, record_elapsed)
            stats.save(stats_path)

    stats.print_summary()
    diversity_report(output_path)
    print(f"\nQA stats saved to: {stats_path}")

if __name__ == "__main__":
    main()