"""Input/output helpers for raw and processed datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .schemas import RawSample


def load_raw_samples(path: str | Path, source: str | None = None) -> list[RawSample]:
    """Load JSON, JSONL, or CSV records into a common raw sample shape."""

    dataset_path = Path(path)
    suffix = dataset_path.suffix.lower()
    if suffix == ".json":
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
        records = data if isinstance(data, list) else data.get("data", [])
    elif suffix == ".jsonl":
        records = [
            json.loads(line)
            for line in dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif suffix == ".csv":
        with dataset_path.open("r", encoding="utf-8", newline="") as handle:
            records = list(csv.DictReader(handle))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")

    return [normalize_record(record, source or dataset_path.stem) for record in records]


def normalize_record(record: dict, source: str = "unknown") -> RawSample:
    """Accept common mental-health dataset field names and normalize them."""

    instruction = first_present(
        record,
        "instruction",
        "prompt",
        "text",
        "utterance",
        "question",
        "input",
        "user",
        "context",
    )
    response = first_present(
        record,
        "response",
        "chosen",
        "good_response",
        "answer",
        "assistant",
        "output",
        "target",
    )
    label = first_present(
        record,
        "category",
        "label",
        "emotion",
        "sentiment",
        "intent",
    )

    return RawSample(
        instruction=clean_text(instruction),
        response=clean_text(response),
        label=clean_text(label),
        source=source,
        metadata={k: v for k, v in record.items() if k not in {"instruction", "response"}},
    )


def save_jsonl(records: Iterable[object], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            if hasattr(record, "to_dict"):
                payload = record.to_dict()
            elif hasattr(record, "__dict__"):
                payload = record.__dict__
            else:
                payload = record
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def clean_text(value: object) -> str:
    return " ".join(str(value or "").replace("\n", " ").split()).strip()


def first_present(record: dict, *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return ""
