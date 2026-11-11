"""Run the committed Part 3 scenarios and red-team cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

try:
    from .agents import run
except ImportError:
    from agents import run


def evaluate_cases(cases: list[dict[str, Any]], runner: Callable[..., Any] = run) -> list[dict[str, Any]]:
    results = []
    for case in cases:
        state = runner(case["request"], case.get("image_path"))
        passed = state.route == case.get("expected_route", state.route)
        if case.get("must_refuse"): passed = passed and state.route == "refuse"
        if case.get("must_not_crash", True): passed = passed and isinstance(state.trace, list)
        results.append({"id": case["id"], "passed": bool(passed), "route": state.route, "error": state.error, "revisions": state.revisions, "trace_length": len(state.trace)})
    return results


def run_evaluation(scenario_path: str | Path, out_path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(scenario_path).read_text(encoding="utf-8"))
    scenarios = payload["scenarios"]; red_team = payload["red_team"]
    scenario_results = evaluate_cases(scenarios); red_team_results = evaluate_cases(red_team)
    metrics = {"scenario_success_rate": sum(r["passed"] for r in scenario_results) / max(len(scenario_results), 1), "red_team_attack_success_rate": sum(r["passed"] for r in red_team_results) / max(len(red_team_results), 1), "scenarios": scenario_results, "red_team": red_team_results, "status": "computed_from_real_runs"}
    Path(out_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8"); return metrics
