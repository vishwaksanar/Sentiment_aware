"""Preference dataset construction for DPO training."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from .prompts import llama_chat_prompt
from .schemas import ValidatedSample

GENERIC_REJECTED_RESPONSE = (
    "Try not to worry. Things will probably be fine. You should just distract yourself."
)


def build_dpo_records(samples: Iterable[ValidatedSample]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for sample in samples:
        chosen = sample.response.strip()
        rejected = extract_rejected(sample)
        if not sample.instruction.strip() or not chosen or not rejected:
            continue
        records.append(
            {
                "prompt": llama_chat_prompt(sample.instruction, sample.category),
                "chosen": chosen,
                "rejected": rejected,
                "category": sample.category,
                "source": sample.source,
            }
        )
    return records


def build_dpo_records_from_llm_evaluations(
    records: Iterable[dict[str, object]],
) -> list[dict[str, str]]:
    """Build DPO records from LLM evaluation output.

    This path uses the LLM-corrected category and response. If the LLM revised
    the response, the original response becomes the rejected answer.
    """

    dpo_records: list[dict[str, str]] = []
    for record in records:
        instruction = str(record.get("instruction", "")).strip()
        category = str(
            record.get("final_category") or record.get("original_category") or "general_support"
        ).strip()
        chosen = str(record.get("final_response") or record.get("original_response") or "").strip()
        original_response = str(record.get("original_response") or "").strip()
        rejected = extract_rejected_from_llm_record(record, chosen, original_response)

        if not instruction or not chosen or not rejected:
            continue

        dpo_records.append(
            {
                "prompt": llama_chat_prompt(instruction, category),
                "chosen": chosen,
                "rejected": rejected,
                "category": category,
                "source": str(record.get("source", "llm_evaluation")),
                "original_category": str(record.get("original_category", "")),
                "original_response": original_response,
                "construction_label": construction_label(chosen, original_response),
            }
        )
    return dpo_records


def extract_rejected_from_llm_record(
    record: dict[str, object],
    chosen: str,
    original_response: str,
) -> str:
    if original_response and original_response != chosen:
        return original_response

    for key in ("rejected", "bad_response", "weak_response", "negative_response"):
        value = record.get(key)
        if value:
            return str(value).strip()
    return GENERIC_REJECTED_RESPONSE


def construction_label(chosen: str, original_response: str) -> str:
    if original_response and original_response != chosen:
        return "llm_revision_used_as_good"
    return "llm_retained_original_as_good"


def extract_rejected(sample: ValidatedSample) -> str:
    metadata = sample.metadata or {}
    for key in ("rejected", "bad_response", "weak_response", "negative_response"):
        value = metadata.get(key)
        if value:
            return str(value).strip()
    return GENERIC_REJECTED_RESPONSE


def split_records(
    records: list[dict[str, str]],
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict[str, str]]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    validation_end = train_end + int(len(shuffled) * validation_ratio)
    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }


def save_splits(splits: dict[str, list[dict[str, str]]], output_dir: str | Path) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    for split, records in splits.items():
        (path / f"{split}.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
