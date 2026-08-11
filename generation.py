import re

from logging_utils import log_call, log_piece_fallback, log_rejected
from masking import CLAUSE_SEP, split_into_chunks, submapping
from prompting import FRAGMENT_TEMPLATE, PROMPT_TEMPLATE, build_avoid_history_section, build_avoid_section
from llm_client import call_ollama, extract_candidate
from validation import validation_reason

TERMINAL_PUNCT_RE = re.compile(r"[?.!]+\s*$")
QUESTION_STYLES = {"question", "conversational", "exploratory", "cohort"}

def temperature_for_attempt(attempt: int, max_attempts: int, base: float, mx: float) -> float:
    step = (mx - base) / max(1, max_attempts - 1)
    return min(mx, base + attempt * step)

def num_predict_for(mapping: dict, base: int = 60, per_value: int = 20, cap: int = 320) -> int:
    return min(cap, base + per_value * len(mapping))

def paraphrase_one(model: str, text: str, mapping: dict, style_desc: str, temperature: float, language: str = "English",
                    avoid_text: str = None, avoid_reason: str = None, avoid_history: list = None) -> tuple[str, float]:
    prompt = PROMPT_TEMPLATE.format(
        question=text, style=style_desc, language=language,
        avoid_section=build_avoid_section(avoid_text, avoid_reason),
        avoid_history_section=build_avoid_history_section(avoid_history),
    )
    raw, elapsed = call_ollama(model, prompt, temperature, num_predict_for(mapping))
    return extract_candidate(raw, mapping), elapsed

def paraphrase_fragment(model: str, fragment_text: str, mapping: dict, temperature: float, language: str = "English",
                         avoid_text: str = None, avoid_reason: str = None) -> tuple[str, float]:
    prompt = FRAGMENT_TEMPLATE.format(
        fragment=fragment_text, language=language,
        avoid_section=build_avoid_section(avoid_text, avoid_reason),
    )
    raw, elapsed = call_ollama(model, prompt, temperature, num_predict_for(mapping))
    return extract_candidate(raw, mapping), elapsed

def generate_piece(model: str, piece_text: str, piece_mapping: dict, style_desc: str,
                    max_attempts: int, base_temp: float, max_temp: float,
                    avoid_history: list, indent: int, label: str,
                    avoid_text: str = None, avoid_reason: str = None,
                    as_fragment: bool = False, language: str = "English") -> tuple[str, float, bool]:
    total_elapsed = 0.0

    for attempt in range(max_attempts):
        temperature = temperature_for_attempt(attempt, max_attempts, base_temp, max_temp)
        attempt_label = f"{label} try {attempt + 1}/{max_attempts}"

        if as_fragment:
            candidate, elapsed = paraphrase_fragment(
                model, piece_text, piece_mapping, temperature, language,
                avoid_text, avoid_reason,
            )
        else:
            candidate, elapsed = paraphrase_one(
                model, piece_text, piece_mapping, style_desc, temperature, language,
                avoid_text, avoid_reason,
                avoid_history if attempt == 0 else None,
            )
        total_elapsed += elapsed

        reason = validation_reason(candidate, piece_mapping, [])
        if reason is None:
            log_call(indent, attempt_label, temperature, elapsed)
            return candidate, total_elapsed, False

        log_rejected(indent, attempt_label, reason, candidate)
        avoid_text, avoid_reason = candidate, reason

    log_piece_fallback(indent, label, max_attempts)
    return piece_text, total_elapsed, True

# The one budget left in the system: how many literal values a single LLM
# call can juggle well before quality degrades. Everything else (how many
# pieces a given record needs, if any) is derived from this per record,
# instead of a fixed split-or-not threshold that either wastes effort on
# simple records or is not enough for very complex ones.
MAX_VALUES_PER_CHUNK = 5

def generate_variant(model: str, masked_text: str, mapping: dict, style_tag: str, style_desc: str,
                      max_attempts: int, base_temp: float, max_temp: float,
                      avoid_history: list, slot_label: str,
                      avoid_text: str = None, avoid_reason: str = None, language: str = "English") -> tuple[str, float, bool, str]:
    chunks = split_into_chunks(masked_text, mapping, MAX_VALUES_PER_CHUNK)

    total_elapsed = 0.0
    any_fallback = False
    pieces: list[str] = []
    first_piece_text = ""

    for i, chunk_text in enumerate(chunks):
        chunk_map = submapping(chunk_text, mapping)
        is_first = i == 0
        text, elapsed, used_fallback = generate_piece(
            model, chunk_text, chunk_map, style_desc,
            max_attempts, base_temp, max_temp,
            avoid_history if is_first else None,
            indent=1, label=f"{slot_label} part {i + 1}/{len(chunks)}",
            avoid_text=avoid_text if is_first else None,
            avoid_reason=avoid_reason if is_first else None,
            as_fragment=not is_first,
            language=language,
        )
        total_elapsed += elapsed
        any_fallback = any_fallback or used_fallback
        if is_first:
            first_piece_text = text
        pieces.append(text)

    if len(pieces) == 1:
        return pieces[0], total_elapsed, any_fallback, pieces[0]

    final_punct = "?" if style_tag in QUESTION_STYLES else "."
    cleaned = [TERMINAL_PUNCT_RE.sub("", p).rstrip() for p in pieces]
    combined = CLAUSE_SEP.join(cleaned) + final_punct
    return combined, total_elapsed, any_fallback, first_piece_text