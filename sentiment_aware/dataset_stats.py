"""Raw dataset statistics for the first pipeline stage."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .io import load_raw_samples
from .schemas import RawSample

SUPPORT_SEEKING_PATTERNS = (
    "what should i do",
    "how can i",
    "how do i",
    "help me",
    "can you help",
    "what can i do",
    "how to",
    "any advice",
    "please help",
    "suggest",
)

SAFETY_PATTERNS = (
    "suicide",
    "kill myself",
    "end my life",
    "self harm",
    "self-harm",
    "hurt myself",
    "worthless",
    "shouldn't be here",
    "do not want to live",
    "don't want to live",
    "hopeless",
    "give up",
)

ADVICE_RESPONSE_PATTERNS = (
    "you should",
    "you can",
    "try to",
    "it may help",
    "i suggest",
    "consider",
    "talk to",
    "reach out",
    "practice",
    "seek help",
)


@dataclass(slots=True)
class DatasetStats:
    dataset: str
    samples: int
    unique_emotion_labels: int
    avg_user_message_length: float
    avg_response_length: float
    support_seeking_queries_pct: float
    safety_sensitive_signals_pct: float
    advice_style_responses_pct: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


def analyze_dataset(name: str, path: str | Path) -> tuple[DatasetStats, Counter[str]]:
    samples = load_raw_samples(path, name)
    emotion_counter: Counter[str] = Counter()
    user_lengths: list[int] = []
    response_lengths: list[int] = []
    support_seeking_count = 0
    safety_count = 0
    advice_response_count = 0

    for sample in samples:
        label = sample.label or "unknown"
        emotion_counter[label] += 1
        user_lengths.append(word_count(sample.instruction))
        response_lengths.append(word_count(sample.response))

        if contains_pattern(sample.instruction, SUPPORT_SEEKING_PATTERNS):
            support_seeking_count += 1
        if contains_pattern(sample.instruction, SAFETY_PATTERNS):
            safety_count += 1
        if contains_pattern(sample.response, ADVICE_RESPONSE_PATTERNS):
            advice_response_count += 1

    total = len(samples)
    stats = DatasetStats(
        dataset=name,
        samples=total,
        unique_emotion_labels=len(emotion_counter),
        avg_user_message_length=rounded_average(user_lengths),
        avg_response_length=rounded_average(response_lengths),
        support_seeking_queries_pct=percentage(support_seeking_count, total),
        safety_sensitive_signals_pct=percentage(safety_count, total),
        advice_style_responses_pct=percentage(advice_response_count, total),
    )
    return stats, emotion_counter


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def contains_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    return any(pattern in lowered for pattern in patterns)


def rounded_average(values: list[int]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def percentage(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round((count / total) * 100, 2)


def format_stats_table(stats: list[DatasetStats]) -> str:
    headers = (
        "Dataset",
        "Samples",
        "Unique Emotion Labels",
        "Avg User Message Length",
        "Avg Response Length",
        "Support-Seeking Queries (%)",
        "Safety-Sensitive Signals (%)",
        "Advice-Style Responses (%)",
    )
    rows = [
        (
            item.dataset,
            str(item.samples),
            str(item.unique_emotion_labels),
            f"{item.avg_user_message_length:.2f}",
            f"{item.avg_response_length:.2f}",
            f"{item.support_seeking_queries_pct:.2f}",
            f"{item.safety_sensitive_signals_pct:.2f}",
            f"{item.advice_style_responses_pct:.2f}",
        )
        for item in stats
    ]
    widths = [
        max(len(row[index]) for row in (headers, *rows))
        for index in range(len(headers))
    ]
    lines = [
        " ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        " ".join("-" * width for width in widths),
    ]
    lines.extend(
        " ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )
    return "\n".join(lines)


def format_top_emotions(emotions: dict[str, Counter[str]], limit: int = 15) -> str:
    lines: list[str] = []
    for dataset_name, counter in emotions.items():
        lines.append("")
        lines.append(dataset_name)
        lines.append("-" * 40)
        for emotion, count in counter.most_common(limit):
            lines.append(f"{emotion:25s} {count}")
    return "\n".join(lines)
