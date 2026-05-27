"""Heuristic category alignment and safety tagging."""

from __future__ import annotations

import re
from collections import Counter
from math import sqrt
from typing import Iterable

from .schemas import AlignedSample, RawSample
from .taxonomy import SAFETY_CUES, UNIFIED_TAXONOMY, canonicalize_label


class HeuristicAlignmentEngine:
    """Map noisy source labels into the unified distress taxonomy."""

    def __init__(
        self,
        min_tokens: int = 3,
        min_confidence: float = 0.18,
        conflict_margin: float = 0.05,
        semantic_weight: float = 0.45,
    ) -> None:
        self.min_tokens = min_tokens
        self.min_confidence = min_confidence
        self.conflict_margin = conflict_margin
        self.semantic_weight = semantic_weight

    def align_many(self, samples: Iterable[RawSample]) -> list[AlignedSample]:
        return [aligned for sample in samples if (aligned := self.align(sample))]

    def align(self, sample: RawSample) -> AlignedSample | None:
        text = normalize_text(f"{sample.instruction} {sample.response}")
        tokens = text.split()
        if len(tokens) < self.min_tokens:
            return None

        label_category = canonicalize_label(sample.label)
        keyword_scores = self.score_keywords(text)
        best_category, best_score = keyword_scores.most_common(1)[0]
        semantic_category, semantic_score = semantic_category_match(text)

        if best_score <= 0:
            category = semantic_category if semantic_score > 0 else label_category
            confidence = 0.35 if label_category != "general_support" else 0.15
        else:
            label_bonus = 0.4 if label_category == best_category else 0.0
            keyword_confidence = best_score / max(4, len(tokens) ** 0.5)
            if semantic_score > keyword_confidence + 0.08 and semantic_category != "general_support":
                category = semantic_category
            else:
                category = best_category
            confidence = min(
                1.0,
                ((1 - self.semantic_weight) * keyword_confidence)
                + (self.semantic_weight * semantic_score)
                + label_bonus,
            )

        notes: list[str] = []
        if label_category != category:
            notes.append(f"source label '{sample.label or 'missing'}' mapped to '{category}'")
        if semantic_score > 0:
            notes.append(
                f"semantic anchor match '{semantic_category}' score={semantic_score:.3f}"
            )

        safety_flags = detect_safety_flags(text)
        if "self_harm" in safety_flags:
            category = "self_harm_safety"
            confidence = max(confidence, 0.95)
            notes.append("self-harm safety cue detected")
        elif "immediate_danger" in safety_flags and category == "general_support":
            category = "crisis_support"
            confidence = max(confidence, 0.85)

        if self.has_conflict(keyword_scores, category):
            notes.append("conflicting emotional cues detected")
            if confidence < 0.6:
                return None

        if confidence < self.min_confidence and category == "general_support":
            return None

        return AlignedSample(
            instruction=sample.instruction,
            response=sample.response,
            label=sample.label,
            source=sample.source,
            metadata=sample.metadata,
            category=category,
            confidence=round(confidence, 4),
            safety_flags=safety_flags,
            alignment_notes=notes,
        )

    def score_keywords(self, text: str) -> Counter[str]:
        scores: Counter[str] = Counter()
        for category, values in UNIFIED_TAXONOMY.items():
            for keyword in values["keywords"]:
                if keyword in text:
                    scores[category] += 2 if " " in keyword else 1
        if not scores:
            scores["general_support"] = 0
        return scores

    def has_conflict(self, scores: Counter[str], category: str) -> bool:
        ranked = scores.most_common(2)
        if len(ranked) < 2:
            return False
        first, second = ranked
        if first[0] != category:
            return False
        return first[1] > 0 and (first[1] - second[1]) <= self.conflict_margin


def detect_safety_flags(text: str) -> list[str]:
    flags: list[str] = []
    for flag, cues in SAFETY_CUES.items():
        if any(cue in text for cue in cues):
            flags.append(flag)
    return flags


def normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9'\s]", " ", lowered)
    return " ".join(lowered.split())


def semantic_category_match(text: str) -> tuple[str, float]:
    """Match text to the category with the closest anchor sentence.

    This is a lightweight local semantic stage for the heuristic algorithm. It
    uses normalized bag-of-words cosine similarity so the pipeline has no hard
    dependency on internet downloads or embedding models. The function compares
    the input against at least five anchor sentences per taxonomy category.
    """

    text_vector = term_counts(text)
    if not text_vector:
        return "general_support", 0.0

    best_category = "general_support"
    best_score = 0.0
    for category, values in UNIFIED_TAXONOMY.items():
        anchor_scores = [
            cosine_similarity(text_vector, term_counts(normalize_text(anchor)))
            for anchor in values.get("anchors", [])
        ]
        score = max(anchor_scores) if anchor_scores else 0.0
        if score > best_score:
            best_category = category
            best_score = score
    return best_category, best_score


STOPWORDS = {
    "about",
    "after",
    "again",
    "all",
    "and",
    "any",
    "are",
    "because",
    "before",
    "being",
    "but",
    "can",
    "cannot",
    "could",
    "day",
    "does",
    "don",
    "even",
    "every",
    "feel",
    "feels",
    "for",
    "from",
    "had",
    "hard",
    "has",
    "have",
    "how",
    "into",
    "like",
    "make",
    "makes",
    "more",
    "much",
    "not",
    "now",
    "own",
    "right",
    "that",
    "the",
    "them",
    "there",
    "this",
    "through",
    "time",
    "too",
    "very",
    "want",
    "when",
    "with",
    "what",
    "will",
    "without",
}


def term_counts(text: str) -> Counter[str]:
    return Counter(
        normalize_token(token)
        for token in text.split()
        if len(token) > 2 and token not in STOPWORDS
    )


def normalize_token(token: str) -> str:
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in shared)
    left_norm = sqrt(sum(value * value for value in left.values()))
    right_norm = sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
