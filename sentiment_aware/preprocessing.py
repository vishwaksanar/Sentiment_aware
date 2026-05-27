"""Lightweight dataset preprocessing helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TypeVar

from .schemas import RawSample

SampleT = TypeVar("SampleT", bound=RawSample)

INSTRUCTION_WRAPPER_RE = re.compile(
    r"^\s*the\s+user\s+is\s+feeling\s+(?P<label>[^.]+)\.\s*"
    r"write\s+an\s+empathetic\s+and\s+supportive\s+response\.\s*"
    r"user\s*:\s*(?P<user_text>.*)$",
    re.IGNORECASE | re.DOTALL,
)


def strip_instruction_wrapper(instruction: str) -> str:
    """Remove the phase-3 instruction scaffold and keep the user text."""

    match = INSTRUCTION_WRAPPER_RE.match(instruction)
    if not match:
        return instruction
    return match.group("user_text").strip()


def extract_wrapper_label(instruction: str) -> str:
    """Extract the emotion label embedded in the phase-3 scaffold."""

    match = INSTRUCTION_WRAPPER_RE.match(instruction)
    if not match:
        return ""
    return match.group("label").strip()


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
