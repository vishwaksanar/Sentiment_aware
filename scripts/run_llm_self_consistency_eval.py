"""Run LLM self-consistency validation on aligned JSONL samples.

The script uses an OpenAI-compatible Chat Completions endpoint. Configure it
with OPENAI_API_KEY and OPENAI_MODEL, or pass --api-key and --model.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests


SYSTEM_PROMPT = """You are an expert evaluator for mental-health support data.
Evaluate whether the assistant response is safe, empathetic, and aligned with
the assigned category. Return only JSON with keys: score, decision, reason,
safety_concern. score must be 0.0 to 1.0. decision must be retain, revise, or
drop. safety_concern must be true or false."""


def build_user_prompt(sample: dict[str, Any]) -> str:
    return (
        f"Assigned category: {sample.get('category', '')}\n"
        f"Safety flags: {sample.get('safety_flags', [])}\n\n"
        f"User/instruction:\n{sample.get('instruction', '')}\n\n"
        f"Assistant response:\n{sample.get('response', '')}\n\n"
        "Judge the sample for DPO dataset quality. Prefer retain only when the "
        "response is clearly supportive, relevant, and safe for the category."
    )


def load_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
            if len(records) >= limit:
                break
    return records


def extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_vote(payload: dict[str, Any]) -> dict[str, Any]:
    score = float(payload.get("score", 0.0))
    decision = str(payload.get("decision", "revise")).lower().strip()
    if decision not in {"retain", "revise", "drop"}:
        decision = "revise"
    return {
        "score": max(0.0, min(1.0, score)),
        "decision": decision,
        "reason": str(payload.get("reason", "")).strip(),
        "safety_concern": bool(payload.get("safety_concern", False)),
    }


def call_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    sample: dict[str, Any],
    temperature: float,
    timeout: int,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.post(
        url,
        headers=headers,
        json={
            "model": model,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(sample)},
            ],
        },
        timeout=timeout,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return normalize_vote(extract_json(content))


def aggregate_votes(votes: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = Counter(vote["decision"] for vote in votes)
    majority_decision, agreement_count = decisions.most_common(1)[0]
    mean_score = sum(vote["score"] for vote in votes) / len(votes)
    return {
        "mean_score": round(mean_score, 4),
        "majority_decision": majority_decision,
        "agreement_count": agreement_count,
        "num_votes": len(votes),
        "safety_concern_votes": sum(1 for vote in votes if vote["safety_concern"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--votes", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", ""))
    parser.add_argument("--sleep", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model:
        raise SystemExit("Missing model. Set OPENAI_MODEL or pass --model.")
    if "api.openai.com" in args.base_url and not args.api_key:
        raise SystemExit("Missing API key. Set OPENAI_API_KEY or pass --api-key.")

    samples = load_jsonl(args.input, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples):
            votes = []
            for vote_index in range(args.votes):
                vote = call_chat_completion(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    sample=sample,
                    temperature=args.temperature,
                    timeout=args.timeout,
                )
                vote["vote_index"] = vote_index
                votes.append(vote)
                if args.sleep:
                    time.sleep(args.sleep)
            result = {
                "sample_index": index,
                "source": sample.get("source"),
                "category": sample.get("category"),
                "heuristic_confidence": sample.get("confidence"),
                "safety_flags": sample.get("safety_flags", []),
                "instruction": sample.get("instruction", ""),
                "response": sample.get("response", ""),
                "votes": votes,
                "self_consistency": aggregate_votes(votes),
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"saved={args.output} samples={len(samples)} votes_per_sample={args.votes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
