"""Runtime-neutral trajectory records.

A trajectory answers "what happened?" It should not answer whether the run was
good, bad, accepted, or useful for training. Evaluation and training labels are
attached later by separate modules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import json


SCHEMA = "simple-agent-lab.trajectory.v1"


@dataclass(frozen=True)
class ModelTurn:
    """One model-visible input/output pair captured from a run."""

    step_id: str
    agent: str
    input_messages: list[Any]
    output_message: Any
    tools: list[Any]
    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class RunTrace:
    """A provider-neutral trace produced by any runtime or script."""

    trace_id: str
    producer: str
    task: str
    messages: list[Any]
    events: list[Any]
    model_turns: list[ModelTurn]
    meta: dict[str, Any] | None = None


def trace_record(trace: RunTrace) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "type": "trajectory",
        **json_safe(trace),
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(json_safe(record), ensure_ascii=False, sort_keys=True))
            f.write("\n")


def json_safe(value: Any) -> Any:
    """Convert project dataclasses and containers into JSON-safe values."""
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
