import itertools

from masking import VAL_RE

STYLES = [
    ("technical_sql", "a technical, database-oriented framing, as if describing query filters to a fellow engineer (e.g. field/value style), without literally writing SQL syntax"),
    ("clinical_formal", "a formal, precise clinical or administrative-register request, as in a clinical document"),
    ("age_based", "framed around the patient's age or birth timeframe; you may describe the relevant age range or life stage IN ADDITION to the literal placeholders, never as a replacement for them"),
    ("reporting", "a business-reporting framing, as if requesting a volumetric or statistical report from a dashboard"),
    ("direct_simple", "direct and simple, no elaboration, minimal wording"),
    ("analytical", "an analytical framing, as if describing a filtering, segmentation, or cross-referencing procedure applied to a dataset"),
    ("natural_language", "a natural, conversational spoken-style question, as a person would actually ask it out loud"),
    ("statistical", "a statistical/counting framing, asking how many records or patients match, or what the incidence is"),
    ("generational", "framed around the patient cohort or generation as a whole, describing the group rather than an individual"),
    ("exploratory", "an exploratory framing, as if browsing or searching through the data out of curiosity"),
]

def opener(text: str, n_words: int = 2) -> str:
    return " ".join(text.split()[:n_words]).lower()

class StyleTracker:

    def __init__(self, styles: list = None, recent_size: int = 3, max_chars: int = 160, opener_window: int = 12):
        self._styles = styles if styles is not None else STYLES
        self._style_cycle = itertools.cycle(self._styles)
        self._recent_texts = {tag: [] for tag, _ in self._styles}
        self._opener_history = {tag: [] for tag, _ in self._styles}
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