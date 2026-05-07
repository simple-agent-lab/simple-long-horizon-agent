"""Training examples derived from trajectories plus optional eval labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .evaluation import EvalResult
from .trajectory import RunTrace, json_safe


SCHEMA = "simple-agent-lab.training-example.v1"


@dataclass(frozen=True)
class TrainingExample:
    trace_id: str
    step_id: str
    agent: str
    input_messages: list[Any]
    output_message: Any
    tools: list[Any]
    reward: float | None = None
    labels: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None


def examples_from_trace(
    trace: RunTrace,
    eval_result: EvalResult | None = None,
) -> list[TrainingExample]:
    reward = eval_result.score if eval_result is not None else None
    labels = None
    if eval_result is not None:
        labels = {
            "passed": eval_result.passed,
            "scorer": eval_result.scorer,
            "reason": eval_result.reason,
        }
    return [
        TrainingExample(
            trace_id=trace.trace_id,
            step_id=turn.step_id,
            agent=turn.agent,
            input_messages=turn.input_messages,
            output_message=turn.output_message,
            tools=turn.tools,
            reward=reward,
            labels=labels,
            meta=turn.meta,
        )
        for turn in trace.model_turns
    ]


def training_example_record(example: TrainingExample) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "type": "training_example",
        **json_safe(example),
    }


def eval_results_by_trace(records: list[Mapping[str, Any]]) -> dict[str, EvalResult]:
    out: dict[str, EvalResult] = {}
    for record in records:
        if record.get("type") != "eval_result":
            continue
        out[str(record["trace_id"])] = EvalResult(
            trace_id=str(record["trace_id"]),
            scorer=str(record["scorer"]),
            passed=bool(record["passed"]),
            score=float(record["score"]),
            metrics=dict(record.get("metrics") or {}),
            reason=str(record.get("reason") or ""),
            meta=dict(record.get("meta") or {}) or None,
        )
    return out
