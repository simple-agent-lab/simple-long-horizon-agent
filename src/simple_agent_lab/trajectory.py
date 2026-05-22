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

from .protocols import Event, MessageEvent, ModelRequestEvent


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


def run_trace_from_state(
    *,
    state: Any,
    trace_id: str,
    producer: str,
    meta: Mapping[str, Any] | None = None,
) -> RunTrace:
    """Build a provider-neutral trajectory from a runtime State-like object."""

    events = list(state.events)
    return RunTrace(
        trace_id=trace_id,
        producer=producer,
        task=str(state.task),
        messages=json_safe(state.messages),
        events=[event_record(event) for event in events],
        model_turns=model_turns_from_events(trace_id, events),
        meta=dict(meta or {}),
    )


def event_record(event: Event) -> dict[str, Any]:
    """Serialize one runtime event with its discriminator intact."""

    return {
        "kind": event.kind.value,
        "index": event.index,
        **json_safe(event.data),
    }


def model_turns_from_events(trace_id: str, events: Iterable[Event]) -> list[ModelTurn]:
    """Extract model-visible input/output pairs from runtime events."""

    turns: list[ModelTurn] = []
    pending: dict[str, Any] | None = None
    model_call_index = 0

    for event in events:
        if isinstance(event, ModelRequestEvent):
            model_call_index += 1
            pending = {
                "agent": str(event.agent or ""),
                "input_messages": event.llm_payload,
                "tools": event.tools,
                "request_event_index": event.index,
                "meta": {
                    "visible_count": event.visible_count,
                    "model_message_count": event.llm_message_count,
                },
            }
            continue

        if not isinstance(event, MessageEvent) or pending is None:
            continue
        message = event.message
        if message.role != "assistant":
            continue
        agent = pending["agent"] or message.sender
        if message.sender != agent:
            continue
        turns.append(
            ModelTurn(
                step_id=f"{trace_id}.model{model_call_index}",
                agent=agent,
                input_messages=json_safe(pending["input_messages"]),
                output_message=json_safe(message),
                tools=json_safe(pending["tools"]),
                meta={
                    **pending["meta"],
                    "request_event_index": pending["request_event_index"],
                    "message_event_index": event.index,
                },
            )
        )
        pending = None

    return turns


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
