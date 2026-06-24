from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path


def ahe_root(run_root: Path) -> Path:
    return run_root / "ahe"


def round_dir(run_root: Path, round_index: int) -> Path:
    path = ahe_root(run_root) / "rounds" / f"round_{round_index:03d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(data), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path, default: object) -> object:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def update_task_history(
    run_root: Path,
    round_index: int,
    scores: Mapping[str, Mapping[str, float]],
) -> dict[str, list[dict[str, object]]]:
    path = ahe_root(run_root) / "task_history.json"
    history = read_json(path, default={})
    if not isinstance(history, dict):
        history = {}

    for instance_id, score_map in scores.items():
        entries = history.setdefault(str(instance_id), [])
        if not isinstance(entries, list):
            entries = []
            history[str(instance_id)] = entries
        reward = float(score_map.get("reward", 0.0))
        entries.append(
            {
                "round": round_index,
                "scores": dict(score_map),
                "passed": reward > 0.0,
            }
        )

    write_json(path, history)
    return history


def evaluate_manifest_predictions(
    manifest: Mapping[str, object],
    baseline_scores: Mapping[str, Mapping[str, float]],
    candidate_scores: Mapping[str, Mapping[str, float]],
) -> dict[str, object]:
    fixed_tasks = sorted(
        instance_id
        for instance_id in baseline_scores
        if float(baseline_scores[instance_id].get("reward", 0.0)) <= 0.0
        and float(candidate_scores.get(instance_id, {}).get("reward", 0.0)) > 0.0
    )
    regressed_tasks = sorted(
        instance_id
        for instance_id in baseline_scores
        if float(baseline_scores[instance_id].get("reward", 0.0)) > 0.0
        and float(candidate_scores.get(instance_id, {}).get("reward", 0.0)) <= 0.0
    )
    fixed_task_set = set(fixed_tasks)
    regressed_task_set = set(regressed_tasks)

    change_evaluations: list[dict[str, object]] = []
    for change in manifest.get("changes", []):
        if not isinstance(change, Mapping):
            continue
        predicted_fixes = [
            str(item) for item in change.get("predicted_fixes", []) if str(item)
        ]
        risk_tasks = [str(item) for item in change.get("risk_tasks", []) if str(item)]
        expected_fixes_verified = sorted(
            task for task in predicted_fixes if task in fixed_task_set
        )
        false_predictions = sorted(
            task for task in predicted_fixes if task not in fixed_task_set
        )
        regressions_observed = sorted(
            task for task in risk_tasks if task in regressed_task_set
        )
        unexpected_fixes = sorted(
            task for task in fixed_tasks if task not in set(predicted_fixes)
        )
        if regressions_observed and not expected_fixes_verified:
            verdict = "harmful"
        elif regressions_observed and expected_fixes_verified:
            verdict = "mixed"
        elif expected_fixes_verified and not regressions_observed:
            verdict = "keep"
        else:
            verdict = "ineffective"
        change_evaluations.append(
            {
                "id": str(change.get("id", "")),
                "expected_fixes_verified": expected_fixes_verified,
                "false_predictions": false_predictions,
                "regressions_observed": regressions_observed,
                "unexpected_fixes": unexpected_fixes,
                "verdict": verdict,
            }
        )

    return {
        "round": manifest.get("round"),
        "fixed_tasks": fixed_tasks,
        "regressed_tasks": regressed_tasks,
        "change_evaluations": change_evaluations,
    }


def update_best_ever(
    run_root: Path,
    round_index: int,
    version_hash: str,
    scores: Mapping[str, Mapping[str, float]],
) -> dict[str, object]:
    path = ahe_root(run_root) / "best_ever.json"
    total = len(scores)
    reward_mean = (
        sum(float(score_map.get("reward", 0.0)) for score_map in scores.values())
        / total
        if total
        else 0.0
    )
    current = {
        "round": round_index,
        "version": version_hash,
        "reward_mean": reward_mean,
        "total": total,
    }
    best = read_json(path, default={})
    if not isinstance(best, dict):
        best = {}
    previous_mean = float(best.get("reward_mean", float("-inf")))
    if reward_mean > previous_mean:
        write_json(path, current)
        return current
    return best


def append_history(run_root: Path, text: str) -> None:
    path = ahe_root(run_root) / "history.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = text if text.endswith("\n") else f"{text}\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(suffix)
