# evals/run_pydantic_evals.py
"""
Second Brain eval suite — pydantic_evals-native implementation.

Uses Evaluator + EvaluationReason per the official custom-evaluator contract:
https://pydantic.dev/docs/ai/evals/evaluators/custom/

Each criterion (relevance, correctness, citation_quality, safety) is returned
as a single EvaluationReason(value=<score>, reason=<text>) so the framework
handles storage and report rendering natively.

The actual judging logic lives in evals/judges.py::judge_case() so the rubric
text stays in evals/rubrics.py (single source of truth, no drift).

Usage:
    python -m evals.run_pydantic_evals                 # runs all tiers
    python -m evals.run_pydantic_evals --tier golden  # one tier
    python -m evals.run_pydantic_evals --tier adversarial
    python -m evals.run_pydantic_evals --tier distribution
    python -m evals.run_pydantic_evals --output evals/results/pydantic_evals.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

# Ensure repo root on sys.path so `import assistant.*` works.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel, Field
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    EvaluationReason,
    Evaluator,
    EvaluatorContext,
    EvaluatorOutput,
)

from evals.cases import (
    GOLDEN_CASES,
    ADVERSARIAL_CASES,
    DISTRIBUTION_CASES,
)
from evals.rubrics import RUBRICS, PASS_THRESHOLDS, AVG_PASS_THRESHOLD
from evals.judges import judge_case
from assistant.agent import run_agent
from assistant.memory import read_context


ALL_TIERS = ["golden", "adversarial", "distribution"]
TIER_POOLS = {
    "golden": GOLDEN_CASES,
    "adversarial": ADVERSARIAL_CASES,
    "distribution": DISTRIBUTION_CASES,
}


# ── Case & Output types ──────────────────────────────────────────────────────

@dataclass
class AgentCase(Case):
    """Eval case for the Second Brain agent."""
    name: str
    inputs: str
    expected_output: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentOutput(BaseModel):
    """What the agent produces for a case."""
    response: str


# ── Convert raw case dicts to AgentCase objects ──────────────────────────────

def _to_pydantic_cases(raw_cases: list[dict]) -> list[AgentCase]:
    return [
        AgentCase(
            name=c["name"],
            inputs=c["query"],
            expected_output=c["expected_output"],
            metadata={
                "tier": c["tier"],
                "category": c.get("category", "general"),
                "tags": c.get("tags", []),
                "priority": c.get("priority", "medium"),
            },
        )
        for c in raw_cases
    ]


# ── Custom evaluator: returns EvaluationReason per criterion ─────────────────

@dataclass
class AgentRubricEvaluator(Evaluator):
    """One async evaluator call -> N EvaluationReason results (one per criterion).

    Each criterion gets its own key in the returned dict; the value is an
    EvaluationReason(value=<score 1-5>, reason=<judge text>). The framework
    stores these in case_result.scores as EvaluationReason objects.
    """

    async def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput:
        response_text = (
            ctx.output.response
            if isinstance(ctx.output, AgentOutput)
            else str(ctx.output)
        )
        scores = await judge_case(
            query=ctx.inputs,
            expected=ctx.expected_output or "",
            actual=response_text,
        )

        out: dict[str, EvaluationReason] = {}
        for criterion, score_obj in scores.items():
            if not isinstance(score_obj, dict) or "score" not in score_obj:
                continue
            out[criterion] = EvaluationReason(
                value=float(score_obj["score"]),
                reason=str(score_obj.get("reasoning", "") or ""),
            )
        return out


# ── Dataset definition ───────────────────────────────────────────────────────

def _build_dataset(tier: str) -> Dataset:
    """Build a pydantic_evals Dataset for the given tier."""
    return Dataset(
        name=f"second_brain_{tier}",
        cases=_to_pydantic_cases(TIER_POOLS[tier]),
        evaluators=[AgentRubricEvaluator()],
    )


# ── Run the agent (the task under test) ──────────────────────────────────────

def _build_memory_block() -> str:
    if os.getenv("ENABLE_MEMORY", "true").lower() not in ("1", "true", "yes", "on"):
        return ""
    days = int(os.getenv("RECENT_LOG_DAYS", "2"))
    return read_context(days=days)


async def _run_agent_on_case(query: str) -> AgentOutput:
    memory_block = _build_memory_block()
    response = await run_agent(query, memory_block=memory_block)
    return AgentOutput(response=response)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _coerce_score(s: Any) -> float | None:
    """Read a finite float from an EvaluationReason / number / EvaluationResult.

    Returns None for NaN, inf, missing, or non-numeric — those are treated as
    'no score recorded' (e.g. EvaluatorFailure from pydantic_evals when a
    judge returns a non-finite value).
    """
    if isinstance(s, (int, float)):
        v = float(s)
        return v if math.isfinite(v) else None
    if hasattr(s, "value"):
        v = s.value
        if isinstance(v, (int, float)):
            v = float(v)
            return v if math.isfinite(v) else None
    return None


# ── Reporter ─────────────────────────────────────────────────────────────────

def _custom_reporter(report: Any, tier: str) -> None:
    """Print a per-tier summary compatible with evals/run.py output."""
    print("\n" + "=" * 70)
    print(f"PYDANTIC EVALS — {tier.upper()} TIER")
    print("=" * 70)

    cases = report.cases
    print(f"Cases: {len(cases)}")

    per_criterion_means: dict[str, list[float]] = {c: [] for c in RUBRICS}
    for case_result in cases:
        for criterion in RUBRICS:
            v = _coerce_score(case_result.scores.get(criterion))
            if v is not None:
                per_criterion_means[criterion].append(v)

    print(f"\nPer-criterion means:")
    blocking_failures = []
    for criterion, scores in per_criterion_means.items():
        if not scores:
            continue
        m = mean(scores)
        threshold = PASS_THRESHOLDS[criterion]
        status = "PASS" if m >= threshold else "FAIL"
        if m < threshold and threshold >= 4.0:
            blocking_failures.append(criterion)
        print(f"  {criterion:<20} {m:.2f} / 5.0  (threshold: {threshold:.1f})  {status}")

    nonempty = [s for s in per_criterion_means.values() if s]
    overall = mean([m for s in nonempty for m in s]) if nonempty else 0.0

    print(f"\nTier overall mean: {overall:.2f} / 5.0")
    print(f"Blocking failures: {blocking_failures or 'none'}")
    print("=" * 70)


# ── Per-case result extraction ───────────────────────────────────────────────

def _extract_results(report: Any, dataset: Dataset, tier: str) -> list[dict]:
    """Walk report.cases and pair each with its original case for full context.

    pydantic_evals strips query / expected_output / actual_output from the
    report object, so we zip by index with the original dataset.cases (which
    evaluate_sync processes sequentially).
    """
    original_cases = list(dataset.cases)
    results: list[dict] = []

    for i, case_result in enumerate(report.cases):
        orig = original_cases[i] if i < len(original_cases) else None

        actual_text = ""
        if hasattr(case_result, "output") and case_result.output is not None:
            actual_text = (
                case_result.output.response
                if isinstance(case_result.output, AgentOutput)
                else str(case_result.output)
            )

        scores_dict: dict[str, Any] = {}
        for criterion, score_obj in case_result.scores.items():
            v = _coerce_score(score_obj)
            if v is None:
                continue
            reason_text = str(getattr(score_obj, "reason", "") or "")
            scores_dict[criterion] = {"score": v, "reasoning": reason_text}

        results.append({
            "name": case_result.name,
            "tier": tier,
            "query": orig.inputs if orig else "",
            "expected_output": orig.expected_output if orig else "",
            "actual_output": actual_text,
            "scores": scores_dict,
        })

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Second Brain eval suite via pydantic_evals (native API)."
    )
    parser.add_argument(
        "--tier",
        choices=ALL_TIERS + ["all"],
        default="all",
        help="Which tier to run, or 'all' for every tier in one pass (default).",
    )
    parser.add_argument(
        "--output",
        default="evals/results/pydantic_evals.json",
        help="Optional path to write JSON results.",
    )
    args = parser.parse_args()

    tiers = ALL_TIERS if args.tier == "all" else [args.tier]

    all_results: list[dict] = []
    for tier in tiers:
        dataset = _build_dataset(tier)
        print(f"\n>>> Running {len(dataset.cases)} {tier} case(s) via pydantic_evals...")
        report = dataset.evaluate_sync(_run_agent_on_case)
        _custom_reporter(report, tier)
        all_results.extend(_extract_results(report, dataset, tier))

    # Combined summary across all selected tiers.
    nonempty_means: dict[str, list[float]] = {c: [] for c in RUBRICS}
    for r in all_results:
        for c in RUBRICS:
            if c in r["scores"]:
                nonempty_means[c].append(r["scores"][c]["score"])
    overall = mean([m for s in nonempty_means.values() for m in s]) if any(nonempty_means.values()) else 0.0

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "framework": "pydantic_evals",
            "framework_version": "native_EvaluationReason",
            "config": {
                "tier": args.tier,
                "judge_model": os.getenv("JUDGE_MODEL", "qwen3:8b"),
                "agent_model": os.getenv("OLLAMA_MODEL", "gemma4:latest"),
            },
            "thresholds": {
                "avg_pass": AVG_PASS_THRESHOLD,
                "per_criterion": PASS_THRESHOLDS,
            },
            "summary": {
                "n_cases": len(all_results),
                "overall_mean": overall,
                "per_criterion_mean": {
                    c: (mean(s) if s else 0.0) for c, s in nonempty_means.items()
                },
                "passes_avg_threshold": overall >= AVG_PASS_THRESHOLD,
            },
            "results": all_results,
        }
        out.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nArtifact written: {out}  ({len(all_results)} cases across {len(tiers)} tier(s))")


if __name__ == "__main__":
    main()