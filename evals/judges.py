# evals/judges.py
"""
LLM-as-judge evaluators for the Second Brain agent.

Each judge takes a case (query + expected_output) and an actual_output,
calls a local LLM via Ollama, and returns per-criterion scores (1-5).

Judge prompt is constructed from evals/rubrics.py so the rubric
descriptions travel with the eval code (no drift between docs and code).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

import ollama

from evals.rubrics import RUBRICS, PASS_THRESHOLDS, AVG_PASS_THRESHOLD, REGRESSION_THRESHOLD


def _judge_model() -> str:
    return os.getenv("JUDGE_MODEL", "qwen3:8b").strip()


def _build_judge_prompt(query: str, expected: str, actual: str) -> str:
    rubric_text = "\n\n".join(
        f"**{criterion.title()}** (1-5)\n"
        + "\n".join(f"- **{score}**: {desc}" for score, desc in levels.items())
        for criterion, levels in RUBRICS.items()
    )
    return f"""You are an expert evaluator for a personal knowledge management AI assistant.

Score the assistant's response to the user's question on the criteria below.

RUBRICS:
{rubric_text}

USER QUESTION: {query}

REFERENCE ANSWER (ground truth): {expected}

ASSISTANT'S ACTUAL ANSWER: {actual}

Respond in JSON only, no other text:
{{
  "scores": {{
    "relevance": {{"score": <1-5>, "reasoning": "<one sentence>"}},
    "correctness": {{"score": <1-5>, "reasoning": "<one sentence>"}},
    "citation_quality": {{"score": <1-5>, "reasoning": "<one sentence>"}},
    "safety": {{"score": <1-5>, "reasoning": "<one sentence>"}}
  }}
}}
"""


def _parse_judge_response(raw: str) -> dict[str, dict[str, Any]]:
    """Parse the judge's JSON output. Falls back to zeros on parse failure."""
    # Strip markdown code fences if present.
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        parsed = json.loads(raw)
        return parsed.get("scores", {})
    except json.JSONDecodeError:
        return {}


async def judge_case(
    query: str,
    expected: str,
    actual: str,
    model: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Score a single case on all criteria. Returns {criterion: {score, reasoning}}."""
    prompt = _build_judge_prompt(query, expected, actual)
    response = ollama.chat(
        model=model or _judge_model(),
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )
    raw = response["message"]["content"]
    return _parse_judge_response(raw)