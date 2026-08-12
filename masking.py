import random
import re

TOKEN_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"

VAL_RE = re.compile(r"\[VAL_[A-Z0-9]{4}\]")
LOOSE_VAL_RE = re.compile(r"\[VAL_[^\]]{0,20}\]")

CLAUSE_SEP = " and "
BETWEEN_RE = re.compile(r"(between \[VAL_[A-Z0-9]{4}\]) and (\[VAL_[A-Z0-9]{4}\])")
BETWEEN_SENTINEL = " \x00BETWEEN_AND\x00 "

def generate_placeholder_tokens(n: int) -> list:
    tokens = set()
    while len(tokens) < n:
        tokens.add("[VAL_" + "".join(random.choices(TOKEN_ALPHABET, k=4)) + "]")
    return list(tokens)

def mask(text: str, spans: list) -> tuple[str, dict]:
    spans = sorted(spans)
    tokens = generate_placeholder_tokens(len(spans))
    mapping, pieces, cursor = {}, [], 0
    for (start, end), placeholder in zip(spans, tokens):
        pieces.append(text[cursor:start])
        mapping[placeholder] = text[start:end]
        pieces.append(placeholder)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces), mapping

def unmask(text: str, mapping: dict) -> str:
    for placeholder, value in mapping.items():
        text = text.replace(placeholder, value)
    return text


def unmask_with_hints(text: str, mapping: dict, hints: dict, hint_probability: float, rng) -> str:
    for placeholder, value in mapping.items():
        replacement = value
        if placeholder in hints and rng.random() < hint_probability:
            replacement = hints[placeholder]
        text = text.replace(placeholder, replacement)
    return text

def submapping(text: str, mapping: dict) -> dict:
    keys = VAL_RE.findall(text)
    return {k: mapping[k] for k in dict.fromkeys(keys)}

_REPEAT_LABEL_RE = re.compile(r"([\w][\w \-()]{1,80}?)\s*\(\1\)", re.IGNORECASE)

def strip_label_repetition(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = _REPEAT_LABEL_RE.sub(r"\1", text)
    return text

_BOOL_VALUES = {"true", "false"}

def simplify_boolean_phrasing(text: str, mapping: dict) -> tuple[str, dict]:
    new_mapping = dict(mapping)
    for placeholder, value in mapping.items():
        if value.lower() not in _BOOL_VALUES:
            continue
        is_true = value.lower() == "true"
        repl = r"with \1" if is_true else r"without \1"
        esc = re.escape(placeholder)

        stop = r"(?:with|which|and)"
        which_re = re.compile(rf"\bwhich ((?:(?!{stop}\b)\w+ ?){{1,5}}) is {esc}", re.IGNORECASE)
        with_re = re.compile(rf"\bwith ((?:(?!{stop}\b)\w+ ?){{1,5}}) {esc}", re.IGNORECASE)

        if which_re.search(text):
            text = which_re.sub(repl, text)
            del new_mapping[placeholder]
        elif with_re.search(text):
            text = with_re.sub(repl, text)
            del new_mapping[placeholder]

    return text, new_mapping

def _protect_between(text: str) -> str:
    return BETWEEN_RE.sub(lambda m: f"{m.group(1)}{BETWEEN_SENTINEL}{m.group(2)}", text)

def _restore_between(text: str) -> str:
    return text.replace(BETWEEN_SENTINEL, " and ")

def split_into_chunks(masked_text: str, mapping: dict, max_values_per_chunk: int) -> list[str]:
    clauses = _protect_between(masked_text).split(CLAUSE_SEP)
    chunks: list[list[str]] = [[]]
    count_in_chunk = 0
    for clause in clauses:
        n_values = len(VAL_RE.findall(clause))
        if count_in_chunk > 0 and count_in_chunk + n_values > max_values_per_chunk:
            chunks.append([])
            count_in_chunk = 0
        chunks[-1].append(clause)
        count_in_chunk += n_values
    return [_restore_between(CLAUSE_SEP.join(c)) for c in chunks if c]