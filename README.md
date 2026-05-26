# Sentiment-Aware Mental Health Chatbot

This repository now contains a modular version of the project described in the report: a data-centric pipeline for a sentiment-aware mental-health support chatbot.

## What Was Added

- `sentiment_aware/taxonomy.py` defines the unified distress-oriented category taxonomy and safety cues.
- `sentiment_aware/io.py` loads raw JSON, JSONL, or CSV datasets into a common schema.
- `sentiment_aware/alignment.py` implements the heuristic alignment engine from the report.
- The heuristic engine now has two stages: keyword/label mapping first, then semantic anchor matching against five or more example sentences per category.
- `sentiment_aware/validation.py` adds a semantic validation boundary with a local offline validator and an LLM prompt helper.
- `sentiment_aware/preference.py` builds DPO-ready prompt/chosen/rejected records.
- `sentiment_aware/training.py` wraps the Unsloth + TRL DPO training flow from the notebook.
- `sentiment_aware/evaluation.py` provides lightweight dataset quality summaries.
- `sentiment_aware/cli.py` exposes the pipeline from the command line.

## Basic Usage

Run heuristic alignment:

```powershell
python -m sentiment_aware.cli align data/raw.json outputs/aligned.jsonl
```

Run validation:

```powershell
python -m sentiment_aware.cli validate data/raw.json outputs/validated.jsonl
```

Build DPO train, validation, and test files:

```powershell
python -m sentiment_aware.cli build-dpo data/raw.json outputs/dpo
```

The training module keeps heavy GPU dependencies optional. Use `sentiment_aware.training.train_dpo` in Colab or another GPU environment after installing `unsloth`, `trl`, `peft`, `accelerate`, and `bitsandbytes`.
