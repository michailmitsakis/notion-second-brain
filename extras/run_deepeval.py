# evals/run_deepeval.py
# NOT OPERATIONAL AS IS
"""
DeepEval showcase — brief eval script using DeepEval's Pydantic AI integration.

This is a *showcase* file, not the canonical eval runner. It demonstrates:
  1. Auto-instrumenting pydantic-ai's Agent via DeepEvalInstrumentationSettings
     (via the new `capabilities=[Instrumentation(...)]` form).
  2. Running the same golden test set as evals/cases.py against two built-in
     DeepEval metrics (AnswerRelevancyMetric, FaithfulnessMetric).
  3. Scoring trace-level (end-to-end agent behavior).

It does NOT:
  - Replace evals/run.py or evals/run_pydantic_evals.py (those own the anchored
    rubric and per-criterion reasoning).
  - Export traces anywhere — runs purely local, no Confident AI cloud.

For observability of the agent in your dev workflow, you could instrument
separately with phoenix.otel.register. Phoenix and DeepEval both wrap
pydantic-ai's OTel hooks, so use one OR the other at a time.

Usage:
    python -m evals.run_deepeval
    python -m evals.run_deepeval --tier golden
    python -m evals.run_deepeval --output evals/results/deepeval_last_run.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

# Ensure repo root on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv()

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate.configs import AsyncConfig
from deepeval.integrations.pydantic_ai import DeepEvalInstrumentationSettings
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from pydantic_ai import Agent

from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.models.openai import OpenAIResponsesModel

from evals.cases import GOLDEN_CASES, ADVERSARIAL_CASES, DISTRIBUTION_CASES
from assistant.agent import SYSTEM_PROMPT
from assistant.tools import retrieve_knowledge, fetch_notion_page


# ── Helpers ──────────────────────────────────────────────────────────────────

TIER_POOLS = {
    "golden": GOLDEN_CASES,
    "adversarial": ADVERSARIAL_CASES,
    "distribution": DISTRIBUTION_CASES,
}

def _build_agent() -> Agent:
    model_name = os.getenv("OLLAMA_MODEL", "gemma4:latest")
    if model_name.startswith("ollama:"):
        model_name = model_name.split(":", 1)[1]
    provider = OllamaProvider(base_url="http://localhost:11434/v1")
    model = OpenAIResponsesModel(model_name=model_name, provider=provider)
    return Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[retrieve_knowledge, fetch_notion_page],
        instrument=DeepEvalInstrumentationSettings(name="second_brain_agent"),
    )

def _to_goldens(raw_cases: list[dict]) -> list[Golden]:
    """Convert evals/cases.py dicts to deepeval Golden objects.

    Skips placeholders and empty inputs — deepeval's metrics require
    non-empty actual_output, and our adv_empty_input case legitimately
    returns None.
    """
    out = []
    for c in raw_cases:
        query = (c.get("query") or "").strip()
        if not query or query.startswith("REPLACE_WITH"):
            continue
        out.append(Golden(
            input=query,
            expected_output=c["expected_output"],
            context=c.get("context"),
        ))
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

async def run_async(args: argparse.Namespace) -> int:
    pool = TIER_POOLS.get(args.tier, GOLDEN_CASES)
    goldens = _to_goldens(pool)
    if not goldens:
        print(f"No runnable cases in tier '{args.tier}'.")
        return 1

    print(f"Running {len(goldens)} {args.tier} case(s) via DeepEval...")

    agent = _build_agent()

    metrics = [
        AnswerRelevancyMetric(threshold=0.7, verbose_mode=True),
        # FaithfulnessMetric(threshold=0.7, verbose_mode=True),
    ]

    dataset = EvaluationDataset(goldens=goldens)

    async def run_one(prompt: str) -> str:
        result = await agent.run(prompt)
        return result.output

    # Iterate goldens. Await each task before evaluating so the agent's
    # output is fully produced and deepeval can read task.result().
    for golden in dataset.evals_iterator(
        async_config=AsyncConfig(run_async=True),
        metrics=metrics,
    ):
        task = asyncio.create_task(run_one(golden.input))
        _ = await task
        dataset.evaluate(task)

    # Pull scored results off the dataset.
    raw_results: Any = None
    for attr in ("test_results", "evaluations", "results", "_test_results"):
        if hasattr(dataset, attr):
            val = getattr(dataset, attr)
            if val:
                raw_results = val
                break

    if raw_results is None:
        scored_goldens = [g for g in dataset.goldens if getattr(g, "scores", None)]
        raw_results = scored_goldens if scored_goldens else dataset.goldens

    results: list[dict] = []
    for r in raw_results:
        metric_scores: dict[str, Any] = {}
        attached_metrics = (
            getattr(r, "metrics", None) or getattr(r, "scores", None) or []
        )
        for m in attached_metrics:
            name = m.__class__.__name__
            metric_scores[name] = {
                "score": getattr(m, "score", None),
                "reason": getattr(m, "reason", ""),
                "passed": (
                    m.is_successful()
                    if hasattr(m, "is_successful") and callable(m.is_successful)
                    else None
                ),
            }
        results.append({
            "input": getattr(r, "input", ""),
            "expected_output": getattr(r, "expected_output", ""),
            "actual_output": getattr(r, "actual_output", ""),
            "metric_scores": metric_scores,
        })

    # Aggregate.
    per_metric_means: dict[str, list[float]] = {}
    for r in results:
        for name, ms in r["metric_scores"].items():
            s = ms["score"]
            if isinstance(s, (int, float)):
                per_metric_means.setdefault(name, []).append(float(s))

    summary = {
        "n_cases": len(results),
        "per_metric_mean": {
            name: (mean(v) if v else 0.0) for name, v in per_metric_means.items()
        },
    }

    print("\n" + "=" * 70)
    print("DEEPEVAL SHOWCASE — RUN SUMMARY")
    print("=" * 70)
    print(f"Cases run: {summary['n_cases']}")
    for name, m in summary["per_metric_mean"].items():
        print(f"  {name:<28} {m:.3f}")
    print("=" * 70)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "framework": "deepeval",
            "framework_version": "pydantic_ai_integration_local",
            "config": {
                "tier": args.tier,
                "agent_model": os.getenv("OLLAMA_MODEL", "gemma4:latest"),
                "judge_model": "deepeval_builtin",
                "metrics": [m.__class__.__name__ for m in metrics],
            },
            "summary": summary,
            "results": results,
        }
        out.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nArtifact written: {out}")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Second Brain eval suite via DeepEval (showcase)."
    )
    parser.add_argument(
        "--tier",
        choices=list(TIER_POOLS.keys()),
        default="golden",
        help="Which tier of cases to run (default: golden).",
    )
    parser.add_argument(
        "--output",
        default="evals/results/deepeval_evals.json",
        help="Optional path to write JSON results.",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run_async(args)))


if __name__ == "__main__":
    main()