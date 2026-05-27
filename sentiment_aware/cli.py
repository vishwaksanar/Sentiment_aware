"""Command-line interface for the sentiment-aware pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .alignment import HeuristicAlignmentEngine
from .evaluation import preference_quality_summary
from .io import load_raw_samples, save_jsonl
from .preference import (
    build_dpo_records,
    build_dpo_records_from_llm_evaluations,
    save_splits,
    split_records,
)
from .preprocessing import dedupe_by_instruction
from .validation import SemanticValidator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sentiment-aware chatbot data pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    align = subparsers.add_parser("align", help="Run heuristic alignment")
    align.add_argument("input", help="Raw JSON, JSONL, or CSV dataset")
    align.add_argument("output", help="Aligned JSONL output path")
    align.add_argument("--source", default=None, help="Optional source dataset name")
    align.add_argument(
        "--dedupe-instruction",
        action="store_true",
        help="Keep only one sample per normalized instruction before alignment",
    )

    validate = subparsers.add_parser("validate", help="Run local semantic validation")
    validate.add_argument("input", help="Raw JSON, JSONL, or CSV dataset")
    validate.add_argument("output", help="Validated JSONL output path")
    validate.add_argument("--source", default=None, help="Optional source dataset name")
    validate.add_argument("--threshold", type=float, default=0.55)
    validate.add_argument(
        "--dedupe-instruction",
        action="store_true",
        help="Keep only one sample per normalized instruction before validation",
    )

    dpo = subparsers.add_parser("build-dpo", help="Build DPO train/validation/test JSON files")
    dpo.add_argument("input", help="Raw JSON, JSONL, or CSV dataset")
    dpo.add_argument("output_dir", help="Directory for split JSON files")
    dpo.add_argument("--source", default=None, help="Optional source dataset name")
    dpo.add_argument("--threshold", type=float, default=0.55)
    dpo.add_argument(
        "--dedupe-instruction",
        action="store_true",
        help="Keep only one sample per normalized instruction before DPO construction",
    )

    llm_dpo = subparsers.add_parser(
        "build-dpo-from-llm",
        help="Build DPO splits from LLM evaluation JSONL output",
    )
    llm_dpo.add_argument("input", help="LLM evaluation JSONL file")
    llm_dpo.add_argument("output_dir", help="Directory for split JSON files")

    preprocess = subparsers.add_parser("preprocess", help="Preprocess raw samples")
    preprocess.add_argument("input", help="Raw JSON, JSONL, or CSV dataset")
    preprocess.add_argument("output", help="Preprocessed JSONL output path")
    preprocess.add_argument("--source", default=None, help="Optional source dataset name")
    preprocess.add_argument(
        "--dedupe-instruction",
        action="store_true",
        help="Keep only one sample per normalized instruction",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "build-dpo-from-llm":
        records = [
            json.loads(line)
            for line in Path(args.input).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        dpo_records = build_dpo_records_from_llm_evaluations(records)
        splits = split_records(dpo_records)
        save_splits(splits, args.output_dir)
        summary = preference_quality_summary(dpo_records)
        Path(args.output_dir, "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2))
        return 0

    samples = load_raw_samples(args.input, getattr(args, "source", None))
    if getattr(args, "dedupe_instruction", False):
        original_count = len(samples)
        samples = dedupe_by_instruction(samples)
        print(f"deduped={len(samples)} removed={original_count - len(samples)}")

    if args.command == "preprocess":
        save_jsonl(samples, args.output)
        print(f"preprocessed={len(samples)}")
        return 0

    engine = HeuristicAlignmentEngine()
    aligned = engine.align_many(samples)

    if args.command == "align":
        save_jsonl(aligned, args.output)
        print(f"aligned={len(aligned)}")
        return 0

    validator = SemanticValidator(threshold=getattr(args, "threshold", 0.55))
    validated = validator.validate_many(aligned)

    if args.command == "validate":
        save_jsonl(validated, args.output)
        print(f"validated={len(validated)}")
        return 0

    records = build_dpo_records(validated)
    splits = split_records(records)
    save_splits(splits, args.output_dir)
    summary = preference_quality_summary(records)
    Path(args.output_dir, "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
