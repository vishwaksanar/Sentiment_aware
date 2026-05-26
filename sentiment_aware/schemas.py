"""Shared data structures for the sentiment-aware pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class RawSample:
    """A minimally normalized source record before emotional alignment."""

    instruction: str
    response: str = ""
    label: str = ""
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AlignedSample(RawSample):
    """A raw sample mapped into the unified distress-oriented taxonomy."""

    category: str = "general_support"
    confidence: float = 0.0
    safety_flags: list[str] = field(default_factory=list)
    alignment_notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ValidatedSample(AlignedSample):
    """A semantically validated sample ready for preference construction."""

    validation_score: float = 0.0
    validation_decision: str = "unchecked"
    validation_notes: list[str] = field(default_factory=list)
