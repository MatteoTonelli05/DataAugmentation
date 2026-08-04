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

def _protect_between(text: str) -> str:
    return BETWEEN_RE.sub(lambda m: f"{m.group(1)}{BETWEEN_SENTINEL}{m.group(2)}", text)

def _restore_between(text: str) -> str:
    return text.replace(BETWEEN_SENTINEL, " and ")

def split_in_half(masked_text: str, mapping: dict) -> tuple[str, str]:
    clauses = _protect_between(masked_text).split(CLAUSE_SEP)
    half = len(mapping) / 2
    first, second, count, in_first = [], [], 0, True
    for clause in clauses:
        if in_first and count >= half:
            in_first = False
        (first if in_first else second).append(clause)
        count += len(VAL_RE.findall(clause))
    return _restore_between(CLAUSE_SEP.join(first)), _restore_between(CLAUSE_SEP.join(second))