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

from sentiment_aware.preprocessing import normalize_instruction_key
from sentiment_aware.taxonomy import UNIFIED_TAXONOMY


CATEGORIES = tuple(UNIFIED_TAXONOMY)

SYSTEM_PROMPT = """You are an expert evaluator for mental-health support data.
Evaluate whether the assistant response is safe, empathetic, and aligned with
the assigned category. Do not rewrite the assistant response. Check whether the
assigned category is correct for the user/instruction and response.

Return only JSON with keys: score, decision, reason, safety_concern,
category_match, suggested_category, revised_response.
- score must be 0.0 to 1.0.
- decision must be retain, revise, or drop.
- safety_concern must be true or false.
- category_match must be true if the assigned category is acceptable.
- suggested_category must be one of the allowed categories. If category_match
  is true, repeat the assigned category.
- revised_response must be an empty string when decision is retain.
- revised_response must contain a safer, more empathetic replacement response
  when decision is revise or drop."""


def build_user_prompt(sample: dict[str, Any]) -> str:
    return (
        f"Allowed categories: {', '.join(CATEGORIES)}\n\n"
        f"Assigned category: {sample.get('category', '')}\n"
        f"Safety flags: {sample.get('safety_flags', [])}\n\n"
        f"User/instruction:\n{sample.get('instruction', '')}\n\n"
        f"Assistant response:\n{sample.get('response', '')}\n\n"
        "Judge the sample for DPO dataset quality. Prefer retain only when the "
        "response is clearly supportive, relevant, safe, and correctly "
        "categorized. If the assigned category is misplaced, set "
        "category_match=false and choose the best suggested_category from the "
        "allowed categories. If the response is inappropriate, unsafe, vague, "
        "dismissive, or not empathetic enough, set decision to revise or drop "
        "and provide a revised_response. The revised response should be "
        "empathetic, concise, non-diagnostic, and safety-aware."
    )


def load_jsonl(
    path: Path,
    limit: int,
    dedupe_instruction: bool = True,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if dedupe_instruction:
                key = normalize_instruction_key(str(record.get("instruction", "")))
                if not key or key in seen:
                    continue
                seen.add(key)
            records.append(record)
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
    suggested_category = str(payload.get("suggested_category", "")).strip()
    if suggested_category not in CATEGORIES:
        suggested_category = "general_support"
    return {
        "score": max(0.0, min(1.0, score)),
        "decision": decision,
        "reason": str(payload.get("reason", "")).strip(),
        "safety_concern": bool(payload.get("safety_concern", False)),
        "category_match": bool(payload.get("category_match", False)),
        "suggested_category": suggested_category,
        "revised_response": str(payload.get("revised_response", "")).strip(),
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
    suggested_categories = Counter(vote["suggested_category"] for vote in votes)
    majority_category, category_agreement_count = suggested_categories.most_common(1)[0]
    mean_score = sum(vote["score"] for vote in votes) / len(votes)
    return {
        "mean_score": round(mean_score, 4),
        "majority_decision": majority_decision,
        "agreement_count": agreement_count,
        "majority_suggested_category": majority_category,
        "category_agreement_count": category_agreement_count,
        "category_match_votes": sum(1 for vote in votes if vote["category_match"]),
        "num_votes": len(votes),
        "safety_concern_votes": sum(1 for vote in votes if vote["safety_concern"]),
    }


def choose_final_response(original_response: str, votes: list[dict[str, Any]]) -> str:
    retain_votes = sum(1 for vote in votes if vote["decision"] == "retain")
    if retain_votes > len(votes) / 2:
        return original_response

    revised_candidates = [
        vote["revised_response"]
        for vote in votes
        if vote["decision"] in {"revise", "drop"} and vote["revised_response"]
    ]
    if not revised_candidates:
        return original_response
    return revised_candidates[0]


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
    parser.add_argument(
        "--allow-duplicate-instructions",
        action="store_true",
        help="Sample rows as-is instead of keeping one row per instruction",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model:
        raise SystemExit("Missing model. Set OPENAI_MODEL or pass --model.")
    if "api.openai.com" in args.base_url and not args.api_key:
        raise SystemExit("Missing API key. Set OPENAI_API_KEY or pass --api-key.")

    samples = load_jsonl(
        args.input,
        args.limit,
        dedupe_instruction=not args.allow_duplicate_instructions,
    )
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
            summary = aggregate_votes(votes)
            original_response = sample.get("response", "")
            result = {
                "sample_index": index,
                "source": sample.get("source"),
                "original_category": sample.get("category"),
                "final_category": summary["majority_suggested_category"],
                "heuristic_confidence": sample.get("confidence"),
                "safety_flags": sample.get("safety_flags", []),
                "instruction": sample.get("instruction", ""),
                "original_response": original_response,
                "final_response": choose_final_response(original_response, votes),
                "votes": votes,
                "self_consistency": summary,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"saved={args.output} samples={len(samples)} votes_per_sample={args.votes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
