import re

from logging_utils import log_call, log_piece_fallback, log_rejected
from masking import CLAUSE_SEP, split_in_half, submapping
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

def generate_variant(model: str, masked_text: str, mapping: dict, style_tag: str, style_desc: str,
                      split_threshold: int, max_attempts: int, base_temp: float, max_temp: float,
                      avoid_history: list, slot_label: str,
                      avoid_text: str = None, avoid_reason: str = None, language: str = "English") -> tuple[str, float, bool, str]:
    if len(mapping) <= split_threshold:
        text, elapsed, used_fallback = generate_piece(
            model, masked_text, mapping, style_desc,
            max_attempts, base_temp, max_temp, avoid_history,
            indent=1, label=slot_label,
            avoid_text=avoid_text, avoid_reason=avoid_reason, language=language,
        )
        return text, elapsed, used_fallback, text

    first_half, second_half = split_in_half(masked_text, mapping)
    map1, map2 = submapping(first_half, mapping), submapping(second_half, mapping)

    text1, elapsed1, fallback1 = generate_piece(
        model, first_half, map1, style_desc,
        max_attempts, base_temp, max_temp, avoid_history,
        indent=1, label=f"{slot_label} half 1/2",
        avoid_text=avoid_text, avoid_reason=avoid_reason, language=language,
    )
    text2, elapsed2, fallback2 = generate_piece(
        model, second_half, map2, style_desc,
        max_attempts, base_temp, max_temp, None,
        indent=1, label=f"{slot_label} half 2/2",
        as_fragment=True, language=language,
    )

    final_punct = "?" if style_tag in QUESTION_STYLES else "."
    text1_clean = TERMINAL_PUNCT_RE.sub("", text1).rstrip()
    text2_clean = TERMINAL_PUNCT_RE.sub("", text2).rstrip()
    combined = f"{text1_clean}{CLAUSE_SEP}{text2_clean}{final_punct}"
    return combined, elapsed1 + elapsed2, fallback1 or fallback2, text1_clean