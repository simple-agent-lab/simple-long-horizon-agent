"""Export training examples from trajectories plus optional eval results."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_agent_lab.training_data import (
    eval_results_by_trace,
    examples_from_trace,
    training_example_record,
)
from simple_agent_lab.trajectory import ModelTurn, RunTrace, read_jsonl, write_jsonl


def trace_from_record(record: dict[str, Any]) -> RunTrace:
    return RunTrace(
        trace_id=str(record["trace_id"]),
        producer=str(record["producer"]),
        task=str(record.get("task") or ""),
        messages=list(record.get("messages") or []),
        events=list(record.get("events") or []),
        model_turns=[
            ModelTurn(
                step_id=str(turn["step_id"]),
                agent=str(turn.get("agent") or ""),
                input_messages=list(turn.get("input_messages") or []),
                output_message=turn.get("output_message"),
                tools=list(turn.get("tools") or []),
                meta=dict(turn.get("meta") or {}) or None,
            )
            for turn in record.get("model_turns") or []
        ],
        meta=dict(record.get("meta") or {}) or None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traces",
        default=str(ROOT / "evals/out/design_version_trajectories.jsonl"),
        help="Trajectory JSONL input.",
    )
    parser.add_argument(
        "--eval-results",
        default=str(ROOT / "evals/out/design_version_eval_results.jsonl"),
        help="Eval-result JSONL input. Missing file means unlabeled examples.",
    )
    parser.add_argument(
        "--jsonl",
        default=str(ROOT / "evals/out/design_version_training_examples.jsonl"),
        help="Training-example JSONL output.",
    )
    args = parser.parse_args()

    traces = [
        trace_from_record(record)
        for record in read_jsonl(args.traces)
        if record.get("type") == "trajectory"
    ]
    eval_records = read_jsonl(args.eval_results) if Path(args.eval_results).exists() else []
    eval_by_trace = eval_results_by_trace(eval_records)

    examples = [
        example
        for trace in traces
        for example in examples_from_trace(trace, eval_by_trace.get(trace.trace_id))
    ]
    write_jsonl(args.jsonl, [training_example_record(example) for example in examples])

    print(f"wrote {len(examples)} training examples to {args.jsonl}")
    for trace in traces:
        print(f"{trace.trace_id}: model_turns={len(trace.model_turns)}")


if __name__ == "__main__":
    main()
