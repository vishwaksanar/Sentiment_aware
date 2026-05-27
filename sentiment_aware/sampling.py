"""Sampling helpers for aligned datasets."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


def stratified_sample(
    records: list[dict[str, Any]],
    *,
    label_key: str = "category",
    sample_size: int = 1000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample records across labels as evenly as possible."""

    if sample_size <= 0 or not records:
        return []
    if len(records) <= sample_size:
        return list(records)

    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get(label_key) or "unknown")].append(record)

    for group_records in groups.values():
        rng.shuffle(group_records)

    labels = sorted(groups)
    base_quota = sample_size // len(labels)
    remainder = sample_size % len(labels)
    selected: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []

    for index, label in enumerate(labels):
        quota = base_quota + (1 if index < remainder else 0)
        group_records = groups[label]
        selected.extend(group_records[:quota])
        leftovers.extend(group_records[quota:])

    if len(selected) < sample_size:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: sample_size - len(selected)])

    selected = selected[:sample_size]
    rng.shuffle(selected)
    return selected
