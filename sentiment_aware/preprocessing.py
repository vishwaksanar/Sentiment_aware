"""Lightweight dataset preprocessing helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TypeVar

from .schemas import RawSample

SampleT = TypeVar("SampleT", bound=RawSample)


def normalize_instruction_key(instruction: str) -> str:
    """Normalize an instruction so repeated prompts collapse to one key."""

    lowered = instruction.lower()
    lowered = re.sub(r"[^a-z0-9'\s]", " ", lowered)
    return " ".join(lowered.split())


def dedupe_by_instruction(samples: Iterable[SampleT]) -> list[SampleT]:
    """Keep the first sample for each normalized instruction."""

    seen: set[str] = set()
    deduped: list[SampleT] = []
    for sample in samples:
        key = normalize_instruction_key(sample.instruction)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(sample)
    return deduped
