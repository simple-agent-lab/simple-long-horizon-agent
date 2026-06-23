"""Shared SWE-bench reward helpers for real evolution recipes."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from evals.swebench.evaluate_predictions import reuse_eval_row
from simple_agent_lab.evolution.types import Run


def load_instances(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Load unique SWE-bench JSONL records from one or more config paths."""

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        p = Path(path)
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            instance_id = str(row.get("instance_id") or "")
            if instance_id in seen:
                continue
            seen.add(instance_id)
            out.append(row)
    return out


def reward_from_result(result: Mapping[str, Any]) -> float:
    """Return the scalar SWE-bench reward encoded in one ``result.json``."""

    agent_package = result.get("agent_package", {})
    if isinstance(agent_package, Mapping) and agent_package.get("used_fallback"):
        return -1.0
    if "resolved" in result:
        return 1.0 if bool(result.get("resolved")) else 0.0
    if "score" in result:
        return float(result.get("score") or 0.0)
    value = result.get("reward", 0.0)
    return float(value or 0.0)


def apply_eval_score(run: Run, eval_row: Mapping[str, Any]) -> None:
    """Merge a SWE-bench eval row back into a run's ``result.json``."""

    path = run.dir / "out" / "result.json"
    result = dict(run.result)
    metrics = eval_row.get("metrics", {})
    metrics = metrics if isinstance(metrics, Mapping) else {}
    score = float(eval_row.get("score", 0.0) or 0.0)
    result.update(
        {
            "resolved": bool(metrics.get("resolved") or eval_row.get("passed")),
            "status": str(metrics.get("status") or eval_row.get("reason") or ""),
            "score": score,
            "reward": score,
        }
    )
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def make_reuse_reward(
    *,
    instances: Sequence[Mapping[str, Any]],
    dataset_name: str,
    model_name: str,
) -> Callable[[Run], float]:
    """Build a reward function that lazily grades raw SWE-bench eval logs.

    The SWE-bench container writes ``model_patch`` and ``eval_log`` first. This
    reward enriches that raw artifact with ``resolved``/``score``/``reward`` the
    first time evolution asks for a scalar reward. Already-scored artifacts are
    read without re-grading, so the function is idempotent across resumes.
    """

    by_id = {str(instance.get("instance_id")): dict(instance) for instance in instances}

    def reward(run: Run) -> float:
        result = dict(run.result)
        if _has_scalar_reward(result):
            return reward_from_result(result)
        if result.get("eval_log") is None:
            return reward_from_result(result)

        instance = by_id.get(run.instance_id)
        if instance is None:
            _write_scoring_error(run, f"missing SWE-bench instance record: {run.instance_id}")
            return 0.0
        try:
            row = reuse_eval_row(
                instance,
                result,
                dataset_name=dataset_name,
                model_name=model_name,
            )
            apply_eval_score(run, row)
            return reward_from_result(Run(run.dir).result)
        except Exception as exc:  # noqa: BLE001 - scoring failures are run diagnostics.
            _write_scoring_error(run, f"{type(exc).__name__}: {exc}")
            return 0.0

    return reward


def make_reuse_reward_from_paths(
    *,
    instance_paths: Sequence[str | Path],
    dataset_name: str,
    model_name: str,
) -> Callable[[Run], float]:
    return make_reuse_reward(
        instances=load_instances(instance_paths),
        dataset_name=dataset_name,
        model_name=model_name,
    )


def _has_scalar_reward(result: Mapping[str, Any]) -> bool:
    return any(key in result for key in ("reward", "score", "resolved"))


def _write_scoring_error(run: Run, message: str) -> None:
    path = run.dir / "out" / "result.json"
    result = dict(run.result)
    result.update(
        {
            "resolved": False,
            "score": 0.0,
            "reward": 0.0,
            "scoring_error": message,
        }
    )
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
