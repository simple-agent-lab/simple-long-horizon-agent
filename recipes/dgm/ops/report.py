"""Summarize DGM SWE-bench performance artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from simple_agent_lab.trace.jsonl import read_jsonl  # noqa: E402

DIAGNOSTIC_FIELDS = (
    "completed",
    "agent_load_failed",
    "agent_build_failed",
    "container_failed",
    "missing_result",
    "scoring_failed",
)


def summarize(
    run_root: str | Path, *, test_dataset: str | Path | None = None
) -> dict[str, Any]:
    run_root = Path(run_root)
    metrics_path = run_root / "generation_metrics.jsonl"
    rows = read_jsonl(metrics_path) if metrics_path.is_file() else []
    monitor = monitor_summary(run_root, rows, test_dataset=test_dataset)
    if not rows:
        return {
            "generations": 0,
            "best_generation": None,
            "best_version": "",
            "best_resolved_rate": 0.0,
            "selector_counts": {},
            "monitor": monitor,
        }

    best = max(rows, key=lambda row: float(row.get("resolved_rate", 0.0)))
    selector_counts = Counter(str(row.get("parent_selection", "")) for row in rows)
    return {
        "generations": len(rows),
        "best_generation": best.get("generation"),
        "best_version": str(best.get("version", "")),
        "best_resolved_rate": float(best.get("resolved_rate", 0.0)),
        "best_resolved": int(best.get("resolved", 0)),
        "best_total": int(best.get("total", 0)),
        "tokens": sum(int(row.get("tokens", 0)) for row in rows),
        "selector_counts": dict(selector_counts),
        "monitor": monitor,
    }


def monitor_summary(
    run_root: Path,
    metrics_rows: list[dict[str, Any]],
    *,
    test_dataset: str | Path | None = None,
) -> dict[str, Any]:
    decisions_path = run_root / "decisions.jsonl"
    if not decisions_path.is_file():
        decisions_path = run_root / "evolution" / "decisions.jsonl"
    decisions = read_jsonl(decisions_path) if decisions_path.is_file() else []
    accepted = [row for row in decisions if bool(row.get("accepted"))]
    latest_accepted = accepted[-1] if accepted else {}
    promoted_child = str(_current_pointer(run_root) or "")
    if promoted_child:
        current_version = promoted_child
    elif latest_accepted:
        current_version = str(_nested(latest_accepted, "candidate", "hash") or "")
    elif decisions:
        current_version = str(_nested(decisions[-1], "candidate", "hash") or "")
    else:
        current_version = ""
    best_train_score = 0.0
    for row in decisions:
        best_train_score = max(
            best_train_score,
            float(_nested(row, "candidate", "scores", "reward") or 0.0),
        )
    latest_metric = metrics_rows[-1] if metrics_rows else {}
    latest_test_score = float(
        latest_metric.get("test_resolved_rate")
        or latest_metric.get("heldout_resolved_rate")
        or 0.0
    )
    return {
        "decision_count": len(decisions),
        "accepted": len(accepted),
        "rejected": len(decisions) - len(accepted),
        **_archive_child_counts(decisions),
        **_diagnostic_counts(decisions),
        "promoted_child": current_version,
        "current_version": current_version,
        "best_train_score": best_train_score,
        "latest_test_score": latest_test_score,
        "test_touched_before_final_scoring": _test_touched(decisions, test_dataset),
    }


def _diagnostic_counts(decisions: list[dict[str, Any]]) -> dict[str, int]:
    counts = {f"diagnostic_{field}": 0 for field in DIAGNOSTIC_FIELDS}
    for row in decisions:
        diagnostics = _nested(row, "candidate", "diagnostics")
        if not isinstance(diagnostics, dict):
            continue
        for field in DIAGNOSTIC_FIELDS:
            counts[f"diagnostic_{field}"] += int(diagnostics.get(field, 0) or 0)
    return counts


def _archive_child_counts(decisions: list[dict[str, Any]]) -> dict[str, int]:
    valid = invalid = improved = tied = regressed = 0
    for row in decisions:
        is_valid = bool(
            _nested(row, "candidate", "valid_parent")
            or float(_nested(row, "deltas", "valid_parent") or 0.0) > 0.0
        )
        if is_valid:
            valid += 1
            delta = float(_nested(row, "deltas", "reward") or 0.0)
            if delta > 0.0:
                improved += 1
            elif delta < 0.0:
                regressed += 1
            else:
                tied += 1
        else:
            invalid += 1
    return {
        "valid_children": valid,
        "invalid_children": invalid,
        "improved_children": improved,
        "tied_children": tied,
        "regressed_children": regressed,
    }


def _current_pointer(run_root: Path) -> str:
    for path in (
        run_root / "evolution" / "pointers" / "current.json",
        run_root / "pointers" / "current.json",
    ):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        return str(data.get("hash") or "")
    return ""


def _nested(row: dict[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _test_touched(
    decisions: list[dict[str, Any]], test_dataset: str | Path | None
) -> bool:
    if test_dataset is None:
        return False
    path = Path(test_dataset)
    if not path.is_file():
        return False
    test_ids = {
        str(row.get("instance_id", ""))
        for row in read_jsonl(path)
        if str(row.get("instance_id", ""))
    }
    if not test_ids:
        return False
    decision_text = json.dumps(decisions, sort_keys=True)
    return any(instance_id in decision_text for instance_id in test_ids)


def print_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    parser.add_argument("--test-dataset")
    args = parser.parse_args()
    print_summary(summarize(args.run_root, test_dataset=args.test_dataset))


if __name__ == "__main__":
    main()
