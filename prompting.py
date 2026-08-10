FRAGMENT_TEMPLATE = """Rewrite the following clause fragment in natural {language}, so it can be appended after another sentence using " and ".

Rules:
- The text below contains placeholders in square brackets, each one unique: copy each one EXACTLY as it appears, character-for-character. Never invent a new placeholder, never reuse one placeholder for a different value, never write a placeholder that is not literally present in the text below
- Every single placeholder in the original MUST appear, unchanged, in the rewritten fragment
- The original has a rigid template style with awkward repetition like "diagnosis (diagnosis)": REMOVE this repetition and phrase naturally
- Use natural, varied wording
- Do NOT end with a question mark, period, or any other punctuation
- Do NOT add a leading "and" - it gets added automatically when this is joined to the rest
{avoid_section}
Original fragment: {fragment}

Rewritten fragment:"""

PROMPT_TEMPLATE = """Rewrite the following query request as a single reformulation in {language}.

Rules:
- The text below contains placeholders in square brackets, each one unique: copy each one EXACTLY as it appears, character-for-character. Never invent a new placeholder, never reuse one placeholder for a different value, never write a placeholder that is not literally present in the text below
- Every single placeholder in the original MUST appear, unchanged, in the reformulation
- The original has a rigid template style with awkward repetition like "diagnosis (diagnosis)": REMOVE this repetition and phrase naturally
- Write it as {style}
- Use natural, varied wording - avoid generic, overused sentence openings
{avoid_history_section}- Keep all clinical/medical meaning exactly intact, do not introduce unrelated medical concepts
- Output ONLY the reformulation, nothing else: no numbering, no bullets, no preamble, no quotes
{avoid_section}
Original: {question}

Reformulation:"""

def build_avoid_history_section(previous_texts: list) -> str:
    if not previous_texts:
        return ""
    items = "\n".join(f'  - "{t}"' for t in previous_texts)
    return (
        "- Do NOT reuse the wording, structure, or vocabulary of these previous "
        f"reformulations written in this same style:\n{items}\n"
    )

def build_avoid_section(avoid_text: str, reason: str = None) -> str:
    if not avoid_text:
        return ""
    reason_line = f" Specifically, the problem was: {reason}." if reason else ""
    return (
        f"\nA previous attempt was rejected.{reason_line} Do NOT repeat this mistake, "
        f"and do NOT repeat this wording - write something meaningfully different:\n\"{avoid_text}\"\n"
    )