"""Evaluation helpers for response quality and category balance."""

from __future__ import annotations

from collections import Counter
from statistics import mean


def category_distribution(records: list[dict]) -> dict[str, int]:
    return dict(Counter(record.get("category", "unknown") for record in records))


def average_response_length(records: list[dict], key: str = "chosen") -> float:
    lengths = [len(str(record.get(key, "")).split()) for record in records if record.get(key)]
    return mean(lengths) if lengths else 0.0


def preference_quality_summary(records: list[dict]) -> dict[str, float | int]:
    return {
        "records": len(records),
        "categories": len(category_distribution(records)),
        "avg_chosen_words": round(average_response_length(records, "chosen"), 2),
        "avg_rejected_words": round(average_response_length(records, "rejected"), 2),
        "missing_chosen": sum(1 for record in records if not record.get("chosen")),
        "missing_rejected": sum(1 for record in records if not record.get("rejected")),
    }
