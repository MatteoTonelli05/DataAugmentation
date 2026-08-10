import argparse
import difflib
import json
import re

from llm_client import call_ollama
from styles import STYLES
from validation import similarity_threshold_for
from num2words import num2words

STYLE_MAP = dict(STYLES)

TEMPLATE = """Say this the way a real person would naturally ask it out loud, in a {style} way. Vary the wording where you can, without changing the meaning of the sentence - but if a value is a literal identifier or code (not an ordinary word), keep it exactly as given, don't try to find a synonym for it. If the question describes a chain of intermediate steps or relations, feel free to express the overall real-world outcome directly instead of listing every intermediate step, as long as the same facts are preserved. Give exactly ONE version, not multiple options.
{value_hints}
Question: {question}
{avoid_note}
Natural version:"""

TEMPERATURE = 0.9

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
NUMBER_RE = re.compile(r"^\d+(\.\d+)?$")
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def opener(text: str, n: int = 2) -> str:
    return " ".join(text.split()[:n]).lower().strip(",.?!\"'")


def body(text: str, n: int = 3) -> str:
    return " ".join(text.split()[n:]).lower()


BODY_SIMILARITY_THRESHOLD = 0.75


LEADING_JUNK_RE = re.compile(r'^[\*\-\u2022"\'\s]+')
TRAILING_JUNK_RE = re.compile(r'["\'\s]+$')


def clean(raw: str) -> str:
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if not lines:
        return ""
    text = lines[-1]
    text = LEADING_JUNK_RE.sub("", text)
    text = TRAILING_JUNK_RE.sub("", text)
    return text


def load_record(path: str, record_id: int) -> tuple[str, str, list]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("id") == record_id:
                return r["prompt"], r["query"], r.get("value_spans", [])
    raise ValueError(f"id={record_id} not found in {path}")


def extract_values(question: str, value_spans: list) -> list[str]:
    return list({question[s:e] for s, e in value_spans})


def library_hint(value: str) -> str | None:
    if DATE_RE.match(value):
        y, m, d = (int(x) for x in value.split("-"))
        return f"{MONTHS[m - 1].capitalize()} {num2words(d, to='ordinal')}, {num2words(y, to='year')}"
    if NUMBER_RE.match(value) and "." not in value:
        return num2words(int(value))
    return None


def suggest_synonym(model: str, value: str) -> str | None:
    prompt = (
        f'Is there a common, everyday English word or short phrase that means the same as "{value}", '
        f'the way an ordinary person would say it (not a database/clinical term)? '
        f'If yes, reply with ONLY that word or phrase, nothing else. '
        f'If "{value}" is a code, identifier, date, number, or has no natural everyday synonym, reply with exactly: NONE'
    )
    raw, _elapsed = call_ollama(model, prompt, temperature=0.3, num_predict=20)
    answer = clean(raw)
    if not answer or answer.strip().upper() == "NONE":
        return None
    return answer


def build_value_hints(model: str, question: str, value_spans: list) -> tuple[str, list[str]]:
    values = extract_values(question, value_spans)
    hints = []
    known_values = []
    for value in values:
        known_values.append(value)
        synonym = library_hint(value)
        if synonym is None:
            synonym = suggest_synonym(model, value)
        if synonym:
            known_values.append(synonym)
            hints.append(f'  - "{value}" -> could naturally be said as "{synonym}"')
    hints_text = ""
    if hints:
        hints_text = "\nSome values in this question have natural everyday equivalents you can use if they fit:\n" + "\n".join(hints) + "\n"
    return hints_text, known_values


def strip_known_values(text: str, known_values: list[str]) -> str:
    result = text
    for v in sorted((v for v in known_values if v), key=len, reverse=True):
        result = re.sub(re.escape(v), "VALUE", result, flags=re.IGNORECASE)
    return result


def generate_variants(model: str, question: str, query: str, value_spans: list, style_tag: str, n: int,
                       max_retries: int = 5) -> list[str]:
    style_desc = STYLE_MAP.get(style_tag, style_tag)
    variants: list[str] = []

    value_hints, known_values = build_value_hints(model, question, value_spans)
    if value_hints:
        print("SUGGESTED SYNONYMS (computed automatically):")
        print(value_hints)

    for _slot in range(n):
        if variants:
            history_note = (
                "\nThese reformulations are already used - don't repeat their wording or structure, "
                "use synonyms and different phrasing for the same values where possible:\n"
                + "\n".join(f'  - "{v}"' for v in variants) + "\n"
            )
        else:
            history_note = ""
        avoid_note = history_note

        for _retry in range(max_retries):
            prompt = TEMPLATE.format(style=style_desc, question=question, query=query, value_hints=value_hints, avoid_note=avoid_note)
            raw, elapsed = call_ollama(model, prompt, temperature=TEMPERATURE, num_predict=200)
            candidate = clean(raw)
            candidate_cmp = strip_known_values(candidate, known_values)

            threshold = similarity_threshold_for(candidate)
            too_similar = next(
                ((prev, similarity(candidate_cmp, strip_known_values(prev, known_values))) for prev in variants if similarity(candidate_cmp, strip_known_values(prev, known_values)) >= threshold),
                None,
            )
            body_too_similar = next(
                ((prev, similarity(body(candidate_cmp), body(strip_known_values(prev, known_values)))) for prev in variants if similarity(body(candidate_cmp), body(strip_known_values(prev, known_values))) >= BODY_SIMILARITY_THRESHOLD),
                None,
            )
            candidate_opener = opener(candidate)
            opener_reused = any(opener(v) == candidate_opener for v in variants)

            if too_similar is None and body_too_similar is None and not opener_reused:
                variants.append(candidate)
                print(f"[{_slot}] (attempt {_retry + 1}/{max_retries}, T={TEMPERATURE:.2f}, {elapsed:.1f}s) ACCEPTED: {candidate}")
                break

            if body_too_similar is not None and too_similar is None:
                prev_text, ratio = body_too_similar
                print(f"    slot {_slot} attempt {_retry + 1}/{max_retries} (T={TEMPERATURE:.2f}, {elapsed:.1f}s) REJECTED "
                      f"(sentence body {ratio:.0%} similar, only the opening changes): {candidate!r}")
                avoid_note = history_note + (
                    f'\nYour last attempt only changed the opening words but reused almost the same structure and content as this already-used variant: '
                    f'"{prev_text}". This time, restructure the whole sentence, not just the beginning.\n'
                )
                continue

            if opener_reused and too_similar is None:
                print(f"    slot {_slot} attempt {_retry + 1}/{max_retries} (T={TEMPERATURE:.2f}, {elapsed:.1f}s) REJECTED "
                      f"(opening '{candidate_opener}...' already used): {candidate!r}")
                avoid_note = history_note + (
                    f'\nYour last attempt started the same way ("{candidate_opener}...") as an already-used variant. '
                    f'Start with completely different opening words this time.\n'
                )
                continue

            prev_text, ratio = too_similar
            print(f"    slot {_slot} attempt {_retry + 1}/{max_retries} (T={TEMPERATURE:.2f}, {elapsed:.1f}s) REJECTED "
                  f"({ratio:.0%} match, threshold {threshold:.0%}): {candidate!r}  <- too similar to: {prev_text!r}")
            avoid_note = history_note + (
                f'\nYour last attempt was too similar ({ratio:.0%} match, threshold {threshold:.0%} for this length) to this already-accepted variant: '
                f'"{prev_text}". Write something meaningfully different - different wording and sentence structure.\n'
            )
        else:
            print(f"  [!] slot {_slot}: exhausted {max_retries} attempts without finding a variant different enough, skipping")

    return variants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="patient_with_values.jsonl", help="source .jsonl file (from dataset_generation)")
    ap.add_argument("--id", type=int, default=160, help="id of the record to use")
    ap.add_argument("--style", default="compact", help="a known style (question, conversational, keyword, ...) or free-form description")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--model", default="gemma4:e2b")
    args = ap.parse_args()

    question, query, value_spans = load_record(args.input, args.id)

    print("ORIGINAL:", question)
    print("STYLE:", args.style)
    print("-" * 70)

    variants = generate_variants(args.model, question, query, value_spans, args.style, args.n)

    for i, v in enumerate(variants):
        print(f"[{i}] {v}")


if __name__ == "__main__":
    main()