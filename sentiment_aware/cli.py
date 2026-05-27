"""Command-line interface for the sentiment-aware pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .alignment import HeuristicAlignmentEngine
from .dataset_stats import analyze_dataset, format_stats_table, format_top_emotions
from .evaluation import preference_quality_summary
from .inference import InferenceConfig, compare_base_and_adapter
from .io import load_raw_samples, save_jsonl
from .preference import (
    build_dpo_records,
    build_dpo_records_from_llm_evaluations,
    save_splits,
    split_records,
)
from .preprocessing import dedupe_by_instruction
from .sampling import stratified_sample
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

    stats = subparsers.add_parser("stats", help="Summarize raw input datasets")
    stats.add_argument(
        "datasets",
        nargs="+",
        help="Dataset specs as name=path, for example counselling=data.jsonl",
    )
    stats.add_argument("--output", default=None, help="Optional JSON summary output path")
    stats.add_argument("--top-n", type=int, default=15, help="Top emotion labels to print")

    infer = subparsers.add_parser(
        "compare-inference",
        help="Compare base and DPO-adapter Llama responses",
    )
    infer.add_argument("user_text", help="User message to answer")
    infer.add_argument("--category", default="general_support", help="Distress category")
    infer.add_argument(
        "--model-name",
        default="unsloth/Llama-3.2-3B-Instruct",
        help="Base model name or path",
    )
    infer.add_argument("--adapter-path", required=True, help="DPO LoRA adapter path")
    infer.add_argument("--max-seq-length", type=int, default=2048)
    infer.add_argument("--max-new-tokens", type=int, default=200)
    infer.add_argument("--temperature", type=float, default=0.7)
    infer.add_argument("--top-p", type=float, default=0.9)

    sample = subparsers.add_parser(
        "sample-aligned",
        help="Stratified sample from heuristic-aligned JSONL records",
    )
    sample.add_argument("input", help="Aligned JSONL input path")
    sample.add_argument("output", help="Sampled JSONL output path")
    sample.add_argument("--size", type=int, default=1000, help="Maximum sample size")
    sample.add_argument("--label-key", default="category", help="Field to stratify by")
    sample.add_argument("--seed", type=int, default=42)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "stats":
        summaries = []
        emotions = {}
        for spec in args.datasets:
            name, path = parse_dataset_spec(spec)
            summary, emotion_counter = analyze_dataset(name, path)
            summaries.append(summary)
            emotions[name] = emotion_counter

        print("\n========== DATASET SUMMARY ==========\n")
        print(format_stats_table(summaries))
        print("\n========== TOP EMOTION LABELS ==========")
        print(format_top_emotions(emotions, limit=args.top_n))

        if args.output:
            payload = {
                "summary": [summary.to_dict() for summary in summaries],
                "top_emotions": {
                    name: dict(counter.most_common(args.top_n))
                    for name, counter in emotions.items()
                },
            }
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"\nsaved={args.output}")
        return 0

    if args.command == "compare-inference":
        config = InferenceConfig(
            model_name=args.model_name,
            adapter_path=args.adapter_path,
            max_seq_length=args.max_seq_length,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        base_response, dpo_response = compare_base_and_adapter(
            args.user_text,
            args.category,
            config,
        )
        print("=" * 80)
        print("BASE MODEL RESPONSE")
        print("=" * 80)
        print(base_response)
        print("\n" + "=" * 80)
        print("DPO-ALIGNED MODEL RESPONSE")
        print("=" * 80)
        print(dpo_response)
        return 0

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

    if args.command == "sample-aligned":
        records = [
            json.loads(line)
            for line in Path(args.input).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        sampled = stratified_sample(
            records,
            label_key=args.label_key,
            sample_size=args.size,
            seed=args.seed,
        )
        save_jsonl(sampled, args.output)
        print(f"sampled={len(sampled)} input={len(records)} label_key={args.label_key}")
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


def parse_dataset_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        path = spec
        return Path(path).stem, path
    name, path = spec.split("=", 1)
    return name.strip(), path.strip()


if __name__ == "__main__":
    raise SystemExit(main())
