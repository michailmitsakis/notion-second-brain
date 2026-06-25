# evals/run_evals.py
"""
CLI entry point for running the eval suite.

Usage:
    python -m evals.run_evals
    python -m evals.run_evals --tier golden
    python -m evals.run_evals --case golden_citation_required
    python -m evals.run_evals --output evals/results/last_run.json

Outputs per-case scores + a summary table to stdout, and writes a JSON
artifact with full results (for CI / trend analysis).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

# Ensure repo root on sys.path so `import evals.*` and `import assistant.*` work.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv()

from evals.cases import ALL_CASES, GOLDEN_CASES, ADVERSARIAL_CASES, DISTRIBUTION_CASES
from evals.judges import judge_case
from evals.rubrics import PASS_THRESHOLDS, AVG_PASS_THRESHOLD, REGRESSION_THRESHOLD
from assistant.agent import run_agent
from assistant.memory import read_context


TIERS = {
    "golden": GOLDEN_CASES,
    "adversarial": ADVERSARIAL_CASES,
    "distribution": DISTRIBUTION_CASES,
    "all": ALL_CASES,
}


def _filter_cases(tier: str | None, case_name: str | None) -> list[dict]:
    pool = TIERS.get(tier or "all", ALL_CASES)
    if case_name:
        return [c for c in pool if c["name"] == case_name]
    return list(pool)


def _build_memory_block() -> str:
    """Build the memory context for the agent (same as production)."""
    from dotenv import load_dotenv as _ld
    _ld(override=False)
    import os
    if os.getenv("ENABLE_MEMORY", "true").lower() not in ("1", "true", "yes", "on"):
        return ""
    days = int(os.getenv("RECENT_LOG_DAYS", "2"))
    return read_context(days=days)


async def _run_one(case: dict, memory_block: str) -> dict:
    """Execute a single case: call the agent, judge the output, return results."""
    query = case["query"]
    if not query or query.startswith("REPLACE_WITH"):
        return {
            "name": case["name"],
            "tier": case["tier"],
            "skipped": True,
            "reason": "placeholder case, fill in real query/expected_output",
            "scores": {},
        }

    # Call the agent.
    try:
        actual = await run_agent(query, memory_block=memory_block)
    except Exception as e:
        return {
            "name": case["name"],
            "tier": case["tier"],
            "skipped": False,
            "error": str(e),
            "scores": {},
        }

    # Judge the output.
    scores = await judge_case(
        query=query,
        expected=case["expected_output"],
        actual=actual,
    )

    return {
        "name": case["name"],
        "tier": case["tier"],
        "category": case.get("category"),
        "tags": case.get("tags", []),
        "priority": case.get("priority"),
        "query": query,
        "expected_output": case["expected_output"],
        "actual_output": actual,
        "scores": scores,
    }


def _aggregate(results: list[dict]) -> dict:
    """Compute per-criterion means and a pass/fail summary."""
    criteria = list(PASS_THRESHOLDS.keys())
    per_criterion: dict[str, list[float]] = {c: [] for c in criteria}

    for r in results:
        if r.get("skipped") or r.get("error"):
            continue
        for c in criteria:
            score_obj = r.get("scores", {}).get(c, {})
            score = score_obj.get("score") if isinstance(score_obj, dict) else None
            if isinstance(score, (int, float)):
                per_criterion[c].append(float(score))

    means = {c: (mean(v) if v else 0.0) for c, v in per_criterion.items()}
    overall = mean(means.values()) if means else 0.0

    blocking_failures = [
        c for c, threshold in PASS_THRESHOLDS.items()
        if means[c] < threshold
    ]

    return {
        "n_cases": len([r for r in results if not r.get("skipped")]),
        "n_skipped": len([r for r in results if r.get("skipped")]),
        "n_errors": len([r for r in results if r.get("error")]),
        "per_criterion_mean": means,
        "overall_mean": overall,
        "passes_avg_threshold": overall >= AVG_PASS_THRESHOLD,
        "passes_per_criterion": {
            c: means[c] >= PASS_THRESHOLDS[c] for c in criteria
        },
        "blocking_failures": blocking_failures,
    }


def _print_summary(results: list[dict], summary: dict) -> None:
    print("\n" + "=" * 70)
    print("EVAL SUMMARY")
    print("=" * 70)
    print(f"Cases run: {summary['n_cases']}  |  "
          f"skipped: {summary['n_skipped']}  |  "
          f"errors: {summary['n_errors']}")
    print(f"Overall mean: {summary['overall_mean']:.2f} / 5.0  "
          f"(threshold: {AVG_PASS_THRESHOLD:.2f})  "
          f"{'PASS' if summary['passes_avg_threshold'] else 'FAIL'}")
    print()
    print("Per-criterion means:")
    for c, m in summary["per_criterion_mean"].items():
        threshold = PASS_THRESHOLDS[c]
        status = "PASS" if m >= threshold else "FAIL"
        print(f"  {c:<20} {m:.2f} / 5.0  (threshold: {threshold:.1f})  {status}")
    print()
    if summary["blocking_failures"]:
        print("BLOCKING FAILURES (must fix before ship):")
        for c in summary["blocking_failures"]:
            print(f"  - {c}")
    else:
        print("No blocking failures.")
    print()
    print("Per-case detail:")
    for r in results:
        if r.get("skipped"):
            print(f"  [SKIP] {r['name']}: {r.get('reason', '')}")
            continue
        if r.get("error"):
            print(f"  [ERR ] {r['name']}: {r['error'][:80]}")
            continue
        scores_str = "  ".join(
            f"{c}={r['scores'].get(c, {}).get('score', '?')}"
            for c in PASS_THRESHOLDS
        )
        print(f"  [{r['tier'][:4]}] {r['name']:<40} {scores_str}")
    print("=" * 70)


async def main_async(args: argparse.Namespace) -> int:
    cases = _filter_cases(args.tier, args.case)
    if not cases:
        print(f"No cases matched (tier={args.tier}, case={args.case})")
        return 1

    print(f"Running {len(cases)} eval case(s)...")
    memory_block = _build_memory_block()

    results = []
    for case in cases:
        print(f"  -> {case['name']} ({case['tier']})")
        result = await _run_one(case, memory_block)
        results.append(result)

    summary = _aggregate(results)
    _print_summary(results, summary)

    # Write JSON artifact.
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "tier": args.tier,
                "case": args.case,
                "judge_model": __import__("evals.judges", fromlist=["_judge_model"])._judge_model(),
                "agent_model": __import__("os").getenv("OLLAMA_MODEL", "gemma4:latest"),
            },
            "thresholds": {
                "avg_pass": AVG_PASS_THRESHOLD,
                "per_criterion": PASS_THRESHOLDS,
                "regression": REGRESSION_THRESHOLD,
            },
            "summary": summary,
            "results": results,
        }
        out_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nArtifact written: {out_path}")

    # Exit code: 0 if passes, 1 if blocking failures.
    return 0 if not summary["blocking_failures"] and summary["passes_avg_threshold"] else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Second Brain eval suite.")
    parser.add_argument(
        "--tier",
        choices=list(TIERS.keys()),
        default="all",
        help="Which tier of cases to run (default: all).",
    )
    parser.add_argument(
        "--case",
        default=None,
        help="Run a single case by name (overrides --tier).",
    )
    parser.add_argument(
        "--output",
        default="evals/results/evals.json",
        help="Path to write JSON results artifact.",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()