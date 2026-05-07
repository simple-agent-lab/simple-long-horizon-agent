"""Runtime-neutral evaluation records.

Evaluation reads a trajectory and answers "how did it do?" Scorers can be
demo-specific, but this result shape is not tied to any runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .trajectory import RunTrace, json_safe


SCHEMA = "simple-agent-lab.evaluation.v1"
Scorer = Callable[[RunTrace], "EvalResult"]


@dataclass(frozen=True)
class EvalResult:
    trace_id: str
    scorer: str
    passed: bool
    score: float
    metrics: dict[str, Any]
    reason: str = ""
    meta: dict[str, Any] | None = None


def eval_result_record(result: EvalResult) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "type": "eval_result",
        **json_safe(result),
    }
