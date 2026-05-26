"""Semantic validation stage for aligned samples."""

from __future__ import annotations

from collections.abc import Callable

from .alignment import normalize_text
from .schemas import AlignedSample, ValidatedSample
from .taxonomy import UNIFIED_TAXONOMY

Validator = Callable[[AlignedSample], tuple[float, str, list[str]]]


class SemanticValidator:
    """Validate aligned samples with a pluggable evaluator.

    If no external LLM evaluator is provided, a deterministic local validator is
    used. This keeps the pipeline runnable offline while preserving the report's
    LLM-check stage as an injectable boundary.
    """

    def __init__(self, threshold: float = 0.55, evaluator: Validator | None = None) -> None:
        self.threshold = threshold
        self.evaluator = evaluator or local_semantic_check

    def validate_many(self, samples: list[AlignedSample]) -> list[ValidatedSample]:
        validated: list[ValidatedSample] = []
        for sample in samples:
            result = self.validate(sample)
            if result.validation_score >= self.threshold:
                validated.append(result)
        return validated

    def validate(self, sample: AlignedSample) -> ValidatedSample:
        score, decision, notes = self.evaluator(sample)
        return ValidatedSample(
            instruction=sample.instruction,
            response=sample.response,
            label=sample.label,
            source=sample.source,
            metadata=sample.metadata,
            category=sample.category,
            confidence=sample.confidence,
            safety_flags=sample.safety_flags,
            alignment_notes=sample.alignment_notes,
            validation_score=round(score, 4),
            validation_decision=decision,
            validation_notes=notes,
        )


def local_semantic_check(sample: AlignedSample) -> tuple[float, str, list[str]]:
    text = normalize_text(f"{sample.instruction} {sample.response}")
    values = UNIFIED_TAXONOMY.get(sample.category, UNIFIED_TAXONOMY["general_support"])
    keyword_hits = sum(1 for keyword in values["keywords"] if keyword in text)
    score = sample.confidence
    notes: list[str] = []

    if keyword_hits:
        score += min(0.35, keyword_hits * 0.12)
        notes.append(f"{keyword_hits} category cue(s) found")
    if sample.response:
        score += 0.1
    if sample.safety_flags:
        score += 0.1
        notes.append("safety cue preserved")
    if len(sample.instruction.split()) < 4:
        score -= 0.2
        notes.append("instruction is very short")

    score = max(0.0, min(1.0, score))
    decision = "retain" if score >= 0.55 else "review"
    return score, decision, notes


def build_llm_validation_prompt(sample: AlignedSample) -> str:
    return (
        "Evaluate whether this mental-health support sample matches the assigned "
        "category. Return JSON with score from 0 to 1, decision retain/revise/drop, "
        "and a short reason.\n\n"
        f"Category: {sample.category}\n"
        f"User text: {sample.instruction}\n"
        f"Response: {sample.response}"
    )
