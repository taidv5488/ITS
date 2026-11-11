from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from backend.plate_format import normalize_plate_text


@dataclass(frozen=True)
class OcrAnalysis:
    total: int
    exact_accuracy: float
    normalized_accuracy: float
    character_accuracy: float
    unreadable_rate: float
    confusion_pairs: dict[str, int]
    low_confidence_count: int

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "exact_accuracy": self.exact_accuracy,
            "normalized_accuracy": self.normalized_accuracy,
            "character_accuracy": self.character_accuracy,
            "unreadable_rate": self.unreadable_rate,
            "confusion_pairs": self.confusion_pairs,
            "low_confidence_count": self.low_confidence_count,
        }


def _char_accuracy(expected: str, predicted: str) -> tuple[int, int, Counter]:
    exp = normalize_plate_text(expected)
    pred = normalize_plate_text(predicted)
    max_len = max(len(exp), len(pred), 1)
    correct = 0
    pairs: Counter = Counter()
    for idx in range(max_len):
        e = exp[idx] if idx < len(exp) else "∅"
        p = pred[idx] if idx < len(pred) else "∅"
        if e == p:
            correct += 1
        else:
            pairs[f"{e}->{p}"] += 1
    return correct, max_len, pairs


def analyze_ocr_rows(rows: list[dict], low_confidence_threshold: float = 0.75) -> OcrAnalysis:
    total = len(rows)
    if total == 0:
        return OcrAnalysis(0, 0.0, 0.0, 0.0, 0.0, {}, 0)
    exact = 0
    normalized = 0
    unreadable = 0
    char_correct = 0
    char_total = 0
    confusion: Counter = Counter()
    low_conf = 0
    for row in rows:
        expected = str(row.get("expected_plate") or row.get("plate") or row.get("ground_truth") or "").strip()
        predicted = str(row.get("predicted_plate") or row.get("prediction") or row.get("ocr_text") or "").strip()
        confidence = float(row.get("confidence") or 0.0)
        if not predicted:
            unreadable += 1
        if expected and predicted and expected == predicted:
            exact += 1
        if expected and predicted and normalize_plate_text(expected) == normalize_plate_text(predicted):
            normalized += 1
        c_ok, c_total, pairs = _char_accuracy(expected, predicted)
        char_correct += c_ok
        char_total += c_total
        confusion.update(pairs)
        if confidence < low_confidence_threshold:
            low_conf += 1
    return OcrAnalysis(
        total=total,
        exact_accuracy=exact / total,
        normalized_accuracy=normalized / total,
        character_accuracy=char_correct / max(1, char_total),
        unreadable_rate=unreadable / total,
        confusion_pairs=dict(confusion.most_common(20)),
        low_confidence_count=low_conf,
    )
