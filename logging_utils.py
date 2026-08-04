LINE = "=" * 70
SUB_LINE = "-" * 70

def _preview(text: str, max_len: int = 100) -> str:
    return text if len(text) <= max_len else text[: max_len - 3] + "..."

def log_record_header(record_id: int, n_values: int, target_n: int) -> None:
    print(f"\n{LINE}")
    print(f"RECORD id={record_id}   [{n_values} values]   target: {target_n} variant(s)")
    print(LINE)

def log_call(indent: int, label: str, temperature: float, elapsed: float) -> None:
    pad = "  " * indent
    print(f"{pad}{label:<28} temp={temperature:.2f}  {elapsed:5.1f}s")

def log_rejected(indent: int, label: str, reason: str, candidate: str) -> None:
    pad = "  " * indent
    print(f"{pad}{label:<28} REJECTED")
    print(f"{pad}    reason : {reason}")
    print(f"{pad}    text   : {_preview(candidate)!r}")

def log_piece_fallback(indent: int, label: str, attempts: int) -> None:
    pad = "  " * indent
    print(f"{pad}{label:<28} kept ORIGINAL text (no valid rewrite after {attempts} attempt(s))")

def log_record_footer(record_id: int, accepted_count: int, target_n: int, elapsed: float) -> None:
    status = "OK" if accepted_count == target_n else ("PARTIAL" if accepted_count else "FAILED")
    print(SUB_LINE)
    print(f"RESULT id={record_id}: {accepted_count}/{target_n} accepted   [{status}]   ({elapsed:.1f}s total)")