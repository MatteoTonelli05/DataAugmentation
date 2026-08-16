import argparse
import json
import random
import time
from pathlib import Path

from dataset_io import append_entries, diversity_report, done_ids, load_records
from generation import generate_variant, generate_variant_no_mask
from logging_utils import LINE, log_record_footer, log_record_header, log_rejected
from masking import mask, simplify_boolean_phrasing, strip_label_repetition, unmask, unmask_with_hints
from qa_stats import QAStats
from styles import STYLES, StyleTracker, opener
from validation import find_similar_text, validation_reason, validation_reason_with_hints
from value_hints import build_hint_map


def target_variants_for(n_values: int, args) -> int:
    # Simple records (few placeholders) have little real room to vary and
    # tend to burn most attempts on rejected near-duplicates. Scale the
    # target down for them instead of forcing the same ceiling on everything.
    return max(
        args.min_variants,
        min(args.n_per_example, args.min_variants + n_values // args.values_per_extra_variant),
    )


def augment_record(record: dict, args, tracker: StyleTracker, stats: QAStats) -> list[tuple[str, str, bool]]:
    if args.no_masking:
        spans = record.get("value_spans", [])
        if not spans:
            return []
        prompt_text = record["prompt"]
        values = [prompt_text[s:e] for s, e in spans]
        masked_text = prompt_text
        mapping = {v: v for v in values}
        hints = build_hint_map(mapping, model=args.model)
    else:
        masked_text, mapping = mask(record["prompt"], record.get("value_spans", []))
        if not mapping:
            return []
        masked_text = strip_label_repetition(masked_text)
        masked_text, mapping = simplify_boolean_phrasing(masked_text, mapping)
        hints = build_hint_map(mapping, model=args.model)

    n_variants = target_variants_for(len(mapping), args)
    log_record_header(record["id"], len(mapping), n_variants)

    record_start = time.perf_counter()
    accepted: list[tuple[str, str, bool]] = []

    for slot in range(n_variants):
        stats.record_slot()

        avoid_text, avoid_reason = None, None
        slot_succeeded = False
        slot_attempts_log = []
        opener_only_reject = None

        for _outer in range(args.outer_attempts):
            style_tag, style_desc = tracker.next_style()
            avoid_history = tracker.recent_texts(style_tag)
            slot_label = f"slot {slot + 1}/{n_variants} [{style_tag}]"
            stats.record_attempt(style_tag)

            if args.no_masking:
                candidate, _elapsed, used_fallback, opening_text = generate_variant_no_mask(
                    args.model, masked_text, list(mapping.keys()), style_tag, style_desc,
                    args.max_attempts, args.base_temperature, args.max_temperature,
                    avoid_history, slot_label,
                    avoid_text, avoid_reason, args.language,
                    hints=hints,
                )
            else:
                candidate, _elapsed, used_fallback, opening_text = generate_variant(
                    args.model, masked_text, mapping, style_tag, style_desc,
                    args.max_attempts, args.base_temperature, args.max_temperature,
                    avoid_history, slot_label,
                    avoid_text, avoid_reason, args.language,
                )

            if args.no_masking:
                reason = validation_reason_with_hints(candidate, list(mapping.keys()), hints, [t for t, _s, _f in accepted])
            else:
                reason = validation_reason(candidate, mapping, [t for t, _s, _f in accepted])

            if reason is None and tracker.opener_used_recently(style_tag, opening_text):
                # Content is genuinely fine, only the opening repeats. Keep it
                # as a safety net instead of discarding a valid variant, but
                # keep trying for something with a fresh opening too.
                if opener_only_reject is None:
                    opener_only_reject = (candidate, style_tag, used_fallback)
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

        if not slot_succeeded and opener_only_reject is not None:
            candidate, style_tag, used_fallback = opener_only_reject
            accepted.append((candidate, style_tag, used_fallback))
            tracker.remember(style_tag, candidate)
            stats.record_accepted(style_tag, used_fallback)
            slot_succeeded = True

        if not slot_succeeded:
            stats.record_slot_failure(record["id"], slot, slot_attempts_log)

    record_elapsed = time.perf_counter() - record_start
    log_record_footer(record["id"], len(accepted), n_variants, record_elapsed)

    def _final_text(text, used_fallback):
        if args.no_masking:
            return text
        if used_fallback:
            return unmask(text, mapping)
        return unmask_with_hints(text, mapping, hints, args.synonym_probability, random)

    return [
        (_final_text(text, used_fallback), style_tag, used_fallback)
        for text, style_tag, used_fallback in accepted
    ]


def run_batch(args) -> None:
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
            final_entries = augment_record(record, args, tracker, stats)
            append_entries(out_f, final_entries, record["id"], record["query"])
            stats.save(stats_path)

    stats.print_summary()
    diversity_report(output_path)
    print(f"\nQA stats saved to: {stats_path}")


def run_single(args) -> None:
    records = load_records(Path(args.input))
    record = next((r for r in records if r["id"] == args.id), None)
    if record is None:
        raise SystemExit(f"id={args.id} not found in {args.input}")

    style_pool = STYLES if args.style is None else [
        (tag, desc) for tag, desc in STYLES if tag == args.style
    ] or [(args.style, args.style)]

    tracker = StyleTracker(styles=style_pool)
    stats = QAStats()

    print(f"ORIGINAL: {record['prompt']}")
    print(f"STYLE: {args.style or 'all styles, cycling'}")
    print(LINE)

    entries = augment_record(record, args, tracker, stats)

    print(LINE)
    for i, (text, style_tag, used_fallback) in enumerate(entries):
        tag = " (fallback: original text)" if used_fallback else ""
        print(f"[{i}] [{style_tag}]{tag} {text}")


def main():
    ap = argparse.ArgumentParser(description="Data augmentation for the text-to-SPARQL dataset, via Ollama")
    ap.add_argument("--input", required=True, help="input .jsonl file, with a value_spans column")
    ap.add_argument("--output", default=None,
                     help="output .jsonl file (append, resumes where it left off). "
                          "Required unless --id is given for a single-record test run.")
    ap.add_argument("--id", type=int, default=None,
                     help="if given, only process this one record id and print the result "
                          "to the console instead of running the full batch")
    ap.add_argument("--style", default=None,
                     help="in single-record mode (--id), restrict generation to this one style "
                          "instead of cycling through all styles")
    ap.add_argument("--model", default="gemma4:e2b", help="Ollama model tag to use")
    ap.add_argument("--n-per-example", type=int, default=5, help="ceiling on variants per example, for complex records")
    ap.add_argument("--min-variants", type=int, default=2, help="floor on variants per example, for simple records")
    ap.add_argument("--values-per-extra-variant", type=int, default=3,
                     help="how many extra literal values earn one more variant slot, scaling the target "
                          "between --min-variants and --n-per-example instead of using a fixed count for everything")
    ap.add_argument("--outer-attempts", type=int, default=2,
                     help="how many times to retry a slot with a new style before giving up on it")
    ap.add_argument("--max-attempts", type=int, default=5, help="max attempts for each piece of text")
    ap.add_argument("--base-temperature", type=float, default=0.6, help="temperature of the first attempt")
    ap.add_argument("--max-temperature", type=float, default=0.95, help="temperature of the last attempt")
    ap.add_argument("--language", default="English", help="generation language, e.g. English or Italian")
    ap.add_argument("--synonym-probability", type=float, default=0.3,
                     help="for values with a natural-language equivalent (dates, small numbers), "
                          "probability of showing that instead of the literal value, e.g. "
                          "'May 3rd, 2010' instead of '2010-05-03'. 0 disables this entirely.")
    ap.add_argument("--no-masking", action="store_true",
                     help="experimental comparison mode: skip placeholder masking entirely, "
                          "send the raw text with literal values inline, always single-shot "
                          "(no chunking for complex records). For comparing output quality "
                          "against the normal masked pipeline, not meant for production runs.")
    ap.add_argument("--stats-output", default=None,
                     help="json file where the run QA stats are saved "
                          "(default: <output>.stats.json). Updated after each record, "
                          "and reloaded if it already exists (useful when resuming an interrupted run).")
    args = ap.parse_args()

    if args.id is not None:
        run_single(args)
    else:
        if not args.output:
            raise SystemExit("--output is required unless --id is given for a single-record test run")
        run_batch(args)


if __name__ == "__main__":
    main()