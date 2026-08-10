import json
from collections import defaultdict
from pathlib import Path


class QAStats:

    def __init__(self):
        self.total_slots = 0
        self.style_attempts: dict[str, int] = defaultdict(int)
        self.style_accepted: dict[str, int] = defaultdict(int)
        self.fallback_count = 0
        self.failed_slots: list[dict] = []

    def record_slot(self) -> None:
        self.total_slots += 1

    def record_attempt(self, style_tag: str) -> None:
        self.style_attempts[style_tag] += 1

    def record_accepted(self, style_tag: str, used_fallback: bool) -> None:
        self.style_accepted[style_tag] += 1
        if used_fallback:
            self.fallback_count += 1

    def record_slot_failure(self, record_id: int, slot: int, attempts: list[dict]) -> None:
        self.failed_slots.append({
            "record_id": record_id,
            "slot": slot,
            "attempts": attempts,
        })

    def to_dict(self) -> dict:
        slots_failed = len(self.failed_slots)
        yield_rate = (
            1 - (slots_failed / self.total_slots) if self.total_slots else None
        )
        return {
            "total_slots": self.total_slots,
            "slots_failed": slots_failed,
            "yield_rate": yield_rate,
            "fallback_count": self.fallback_count,
            "style_attempts": dict(self.style_attempts),
            "style_accepted": dict(self.style_accepted),
            "failed_slots": self.failed_slots,
        }

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self.total_slots = data.get("total_slots", 0)
        self.style_attempts = defaultdict(int, data.get("style_attempts", {}))
        self.style_accepted = defaultdict(int, data.get("style_accepted", {}))
        self.fallback_count = data.get("fallback_count", 0)
        self.failed_slots = data.get("failed_slots", [])

    def print_summary(self) -> None:
        d = self.to_dict()
        print("\n" + "=" * 70)
        print("QA STATS")
        print("=" * 70)
        print(f"Total slots attempted : {d['total_slots']}")
        print(f"Failed slots (0 variants produced): {d['slots_failed']}")
        if d["yield_rate"] is not None:
            print(f"Yield rate            : {d['yield_rate']:.1%}")
        print(f"Fallback variants     : {d['fallback_count']}")
        print("Attempts/accepted per style:")
        for style in sorted(d["style_attempts"]):
            att = d["style_attempts"].get(style, 0)
            acc = d["style_accepted"].get(style, 0)
            rate = acc / att if att else 0
            print(f"  {style:<16} attempted={att:<5} accepted={acc:<5} ({rate:.0%})")