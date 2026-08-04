import itertools

from masking import VAL_RE

STYLES = [
    ("question", "a natural interrogative sentence; choose your own question word and phrasing, vary it each time, do not default to a generic opener"),
    ("imperative", "a command/instruction sentence; choose your own verb each time instead of a generic one"),
    ("declarative", "a plain declarative statement describing the criteria, not phrased as a question or command"),
    ("conversational", "a casual, natural-sounding way of asking, as if speaking informally"),
    ("formal", "a formal, precise clinical/administrative-register request"),
    ("keyword", "a compact, telegraphic keyword-style phrase, not a full sentence"),
    ("technical", "a technical/database-oriented framing, as if describing query filters or record counts to a fellow engineer"),
    ("reporting", "a business-reporting framing, as if requesting a volumetric or statistical report from a dashboard"),
    ("analytical", "an analytical framing, as if describing a filtering or analysis procedure applied to a dataset"),
    ("exploratory", "an exploratory framing, as if browsing or searching through the data out of curiosity"),
    ("cohort", "an age- or cohort-based framing; you may describe the relevant age range or generational cohort IN ADDITION to the literal placeholders, never as a replacement for them"),
]

def opener(text: str, n_words: int = 4) -> str:
    return " ".join(text.split()[:n_words]).lower()

class StyleTracker:

    def __init__(self, recent_size: int = 3, max_chars: int = 160, opener_window: int = 12):
        self._style_cycle = itertools.cycle(STYLES)
        self._recent_texts = {tag: [] for tag, _ in STYLES}
        self._opener_history = {tag: [] for tag, _ in STYLES}
        self._recent_size = recent_size
        self._max_chars = max_chars
        self._opener_window = opener_window

    def next_style(self) -> tuple:
        return next(self._style_cycle)

    def recent_texts(self, style_tag: str) -> list:
        return list(self._recent_texts.get(style_tag, []))

    def opener_used_recently(self, style_tag: str, text: str) -> bool:
        return opener(text) in self._opener_history.get(style_tag, [])

    def remember(self, style_tag: str, text: str) -> None:
        sanitized = VAL_RE.sub("...", text)
        preview = sanitized if len(sanitized) <= self._max_chars else sanitized[: self._max_chars - 3] + "..."
        bucket = self._recent_texts.setdefault(style_tag, [])
        bucket.append(preview)
        del bucket[: -self._recent_size]

        openers = self._opener_history.setdefault(style_tag, [])
        openers.append(opener(text))
        del openers[: -self._opener_window]