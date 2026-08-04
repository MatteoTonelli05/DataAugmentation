import re
import time

import ollama
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20))
def call_ollama(model: str, prompt: str, temperature: float, num_predict: int = 150) -> tuple[str, float]:
    start = time.perf_counter()
    response = ollama.generate(
        model=model, prompt=prompt, think=False,
        options={"temperature": temperature, "top_p": 0.9, "num_predict": num_predict, "num_ctx": 1024},
        keep_alive="30m",
    )
    elapsed = time.perf_counter() - start
    return response["response"], elapsed

def extract_candidate(raw: str, mapping: dict) -> str:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<\|channel\|?>?\s*thought.*?<channel\|>", "", raw, flags=re.DOTALL)
    lines = [
        re.sub(r"^\s*\d+[.)]\s*", "", line.strip("-\u2022\"' \t"))
        for line in raw.splitlines() if line.strip()
    ]
    if not lines:
        return ""
    for line in lines:
        if all(placeholder in line for placeholder in mapping):
            return line
    return lines[-1]