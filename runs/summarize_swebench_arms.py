"""Summarize and compare SWE-bench arm runs side by side.

Reads each run-id's per-instance ``result.json`` (the layout the generic runner
writes: ``<run-root>/<run-id>/<instance>/out/result.json``) and prints a compact
per-arm table: how many instances ran, how many produced a non-empty patch, and
the output-token cost the arm spent (from the ``workflow`` breakdown the facade
records). If an arm's official eval-results JSONL is present it also folds in the
resolved count — the quality side of the quality-vs-cost frontier.

Usage:
    uv run python runs/summarize_swebench_arms.py --run-root evals/out/swebench_pro \
        arms-20260622-120000-baseline arms-20260622-120000-loop \
        arms-20260622-120000-pdr

Resolved counts: pass ``--eval-results <run_id>=<path.jsonl>`` for any arm whose
predictions you graded with the official harness (see
``evals/swebench/evaluate_predictions.py --pro --run-official``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_result(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8") or "{}")
    except (OSError, json.JSONDecodeError):
        return {}


def _arm_stats(run_root: Path, run_id: str) -> dict[str, Any]:
    """Aggregate one arm's per-instance results into headline numbers."""

    instances = 0
    patched = 0
    output_tokens = 0
    for result_path in sorted((run_root / run_id).glob("*/out/result.json")):
        result = _load_result(result_path)
        if not result:
            continue
        instances += 1
        if str(result.get("model_patch") or "").strip():
            patched += 1
        workflow = result.get("workflow")
        if isinstance(workflow, dict):
            output_tokens += int(workflow.get("output_tokens") or 0)
    return {
        "instances": instances,
        "patched": patched,
        "output_tokens": output_tokens,
    }


def _iter_json_records(text: str):
    """Yield top-level JSON objects from concatenated/pretty-printed JSON.

    The official-grader EvalResult file (`eval_results.jsonl`) is a stream of
    pretty-printed objects, not one-per-line, so `raw_decode` is used to walk it.
    """

    decoder = json.JSONDecoder()
    idx, n = 0, len(text)
    while idx < n:
        while idx < n and text[idx] in " \t\r\n":
            idx += 1
        if idx >= n:
            break
        obj, idx = decoder.raw_decode(text, idx)
        yield obj


def _is_resolved(row: dict[str, Any]) -> bool:
    """A record counts as resolved across the EvalResult shape variants."""

    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    status = str(row.get("status") or metrics.get("status") or "").lower()
    return (
        row.get("resolved") is True
        or metrics.get("resolved") is True
        or row.get("passed") is True
        or row.get("score") == 1.0
        or status == "resolved"
    )


def _resolved_count(path: Path) -> tuple[int, int]:
    """Count (resolved, total) from a grader EvalResult file, if present."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return (0, 0)
    resolved = total = 0
    try:
        for row in _iter_json_records(text):
            if not isinstance(row, dict):
                continue
            total += 1
            if _is_resolved(row):
                resolved += 1
    except json.JSONDecodeError:
        pass
    return (resolved, total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+", help="Arm run-ids to compare.")
    parser.add_argument("--run-root", default="evals/out/swebench_pro")
    parser.add_argument(
        "--eval-results",
        action="append",
        default=[],
        metavar="RUN_ID=PATH",
        help="Optional graded EvalResult JSONL for an arm (repeatable).",
    )
    args = parser.parse_args()

    eval_paths: dict[str, Path] = {}
    for item in args.eval_results:
        run_id, _, path = item.partition("=")
        if path:
            eval_paths[run_id] = Path(path)

    run_root = Path(args.run_root)
    header = f"{'arm (run-id)':<40} {'inst':>5} {'patched':>8} {'out_tok':>10} {'resolved':>10}"
    print(header)
    print("-" * len(header))
    for run_id in args.run_ids:
        stats = _arm_stats(run_root, run_id)
        resolved_cell = "-"
        if run_id in eval_paths:
            resolved, total = _resolved_count(eval_paths[run_id])
            resolved_cell = f"{resolved}/{total}"
        print(
            f"{run_id:<40} {stats['instances']:>5} {stats['patched']:>8} "
            f"{stats['output_tokens']:>10} {resolved_cell:>10}"
        )


if __name__ == "__main__":
    main()
