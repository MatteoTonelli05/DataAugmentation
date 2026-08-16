import re

from num2words import num2words

from llm_client import call_ollama

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:T\d{2}:\d{2}:\d{2})?$")
NUMBER_RE = re.compile(r"^\d+$")

_URL_RE = re.compile(r"^https?://")
_CODE_PREFIX_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9]*:")
_JUNK_SYMBOLS_RE = re.compile(r"[?+\]\[/><;*!^=\\|{}]")
_GENDER_VALUES = {"male", "female", "unknown"}
_CLINICAL_GRADE_RE = re.compile(r"^(grade\s+\d+|[ivx]+\s*\(AJCC\)|stage\s+[ivx\d]+)$", re.IGNORECASE)

_SYNONYM_CACHE: dict[str, str | None] = {}


def natural_date(value: str) -> str | None:
    m = DATE_RE.match(value)
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= mo <= 12) or not (1 <= d <= 31):
        return None
    return f"{MONTHS[mo - 1]} {num2words(d, to='ordinal')}, {num2words(y, to='year')}"


def natural_number(value: str) -> str | None:
    if not NUMBER_RE.match(value):
        return None
    n = int(value)
    if n > 999:
        return None
    return num2words(n)


def natural_compound_label(value: str) -> str | None:
    if " - " not in value:
        return None
    prefix, _, suffix = value.partition(" - ")
    suffix = suffix.strip()
    if not suffix or len(suffix) >= len(prefix):
        return None
    return suffix


def _plausible_synonym_candidate(value: str) -> bool:
    if not (2 <= len(value) <= 40):
        return False
    if _URL_RE.match(value) or _CODE_PREFIX_RE.match(value):
        return False
    if _JUNK_SYMBOLS_RE.search(value):
        return False
    if value.strip().lower() in _GENDER_VALUES:
        # "sex assigned at birth" is deliberately distinct from gender identity;
        # a loose synonym like "Woman"/"Man" would conflate the two.
        return False
    if _CLINICAL_GRADE_RE.match(value.strip()):
        # controlled clinical staging/grading (e.g. "Grade 5", "IV (AJCC)") has
        # a precise meaning that a colloquial synonym can silently change.
        return False
    return True


def llm_synonym(model: str, value: str) -> str | None:
    if value in _SYNONYM_CACHE:
        return _SYNONYM_CACHE[value]
    if not _plausible_synonym_candidate(value):
        _SYNONYM_CACHE[value] = None
        return None

    prompt = (
        f'Is there a common, everyday English word or short phrase that means the same as "{value}", '
        f'the way an ordinary person would say it (not a database or clinical code)? '
        f'If yes, reply with ONLY that word or phrase, nothing else. '
        f'If "{value}" is a coded identifier, a controlled clinical classification (like a stage, grade, '
        f'or diagnosis code), or anything where you are not certain the everyday word means EXACTLY '
        f'the same thing, reply with exactly: NONE. Do not guess.'
    )
    raw, _elapsed = call_ollama(model, prompt, temperature=0.3, num_predict=20)
    answer = raw.strip().strip('"').splitlines()[0].strip() if raw.strip() else ""
    result = None if not answer or answer.upper() == "NONE" else answer

    _SYNONYM_CACHE[value] = result
    return result


def build_hint_map(mapping: dict, model: str = None) -> dict:
    hints = {}
    for placeholder, value in mapping.items():
        hint = natural_date(value) or natural_number(value) or natural_compound_label(value)
        if not hint and model:
            hint = llm_synonym(model, value)
        if hint:
            hints[placeholder] = hint
    return hints