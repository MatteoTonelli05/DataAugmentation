import difflib

from masking import LOOSE_VAL_RE

def similarity_threshold_for(text: str, base: float = 0.85, lenient_cap: float = 0.97, short_words: int = 6, long_words: int = 20) -> float:
    n_words = len(text.split())
    if n_words <= short_words:
        return lenient_cap
    if n_words >= long_words:
        return base
    frac = (long_words - n_words) / (long_words - short_words)
    return base + frac * (lenient_cap - base)

def validation_reason(candidate: str, mapping: dict, accepted_so_far: list, max_similarity: float = None) -> str:
    missing = [p for p in mapping if p not in candidate]
    if missing:
        return f"you dropped the placeholder(s) {', '.join(missing)} - every single one must appear unchanged"

    malformed = sorted(set(m for m in LOOSE_VAL_RE.findall(candidate) if m not in mapping))
    if malformed:
        return f"you wrote malformed or invented placeholder-like text: {', '.join(malformed)} - placeholders must be copied exactly, never altered"

    threshold = max_similarity if max_similarity is not None else similarity_threshold_for(candidate)
    for prev in accepted_so_far:
        ratio = difflib.SequenceMatcher(None, candidate.lower(), prev.lower()).ratio()
        if ratio >= threshold:
            return f"this wording is too similar ({ratio:.0%} match, threshold {threshold:.0%} for this length) to a variant already produced for this same request"

    return None

def find_similar_text(candidate: str, accepted_so_far: list, max_similarity: float = None) -> tuple:
    threshold = max_similarity if max_similarity is not None else similarity_threshold_for(candidate)
    for prev in accepted_so_far:
        ratio = difflib.SequenceMatcher(None, candidate.lower(), prev.lower()).ratio()
        if ratio >= threshold:
            return prev, ratio
    return None, 0.0

def is_valid(candidate: str, mapping: dict, accepted_so_far: list, max_similarity: float = None) -> bool:
    return validation_reason(candidate, mapping, accepted_so_far, max_similarity) is None


def missing_values_with_hints(candidate: str, values: list, hints: dict) -> list:
    missing = []
    for v in values:
        if v in candidate:
            continue
        hint = hints.get(v)
        if hint and hint in candidate:
            continue
        missing.append(v)
    return missing


def validation_reason_with_hints(candidate: str, values: list, hints: dict, accepted_so_far: list, max_similarity: float = None) -> str:
    missing = missing_values_with_hints(candidate, values, hints)
    if missing:
        return (
            f"you dropped or altered the value(s) {', '.join(missing)} - each one, "
            f"either as originally given or in the suggested natural phrasing shown to you, must appear"
        )

    threshold = max_similarity if max_similarity is not None else similarity_threshold_for(candidate)
    for prev in accepted_so_far:
        ratio = difflib.SequenceMatcher(None, candidate.lower(), prev.lower()).ratio()
        if ratio >= threshold:
            return f"this wording is too similar ({ratio:.0%} match, threshold {threshold:.0%} for this length) to a variant already produced for this same request"

    return None