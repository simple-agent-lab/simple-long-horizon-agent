"""RunTrace: the provider-neutral trace value plus its record schema.

A ``RunTrace`` bundles the raw event log and messages for one run; spans
and model turns are derived on demand from the span/training layers. The
``*_record`` functions serialize a run into the canonical JSON shape that
the writers in ``live`` persist.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..messages import (
    _token_usage_from_record,
    message_from_record,
    normalize_content,
    text_of,
)
from ..protocols import (
    AgentEndEvent,
    AgentStartEvent,
    ContextCompressionEvent,
    Event,
    MessageEvent,
    ModelRequestEvent,
    ModelResponseEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from .jsonl import json_safe, read_jsonl
from .spans import Span, merge_sub_agent_spans, span_record, spans_from_events
from .training import ModelTurn, model_turns_from_events


SCHEMA = "simple-agent-lab.trajectory.v3"

# The first line of the append-only ("event stream") on-disk shape: identity and
# metadata for the events that follow, one event_record per subsequent line. A
# finished run is materialized back into the single canonical `trace_record`
# (see `trace_record_from_jsonl`); the stream is the cheap live representation.
TRACE_HEADER_TYPE = "trace_header"


@dataclass(frozen=True)
class RunTrace:
    """Provider-neutral trace: raw events + on-demand span tree."""

    trace_id: str
    producer: str
    task: str

    events: list[Any]
    messages: list[Any]

    meta: dict[str, Any] | None = None

    def spans(self) -> list[Span]:
        """Derive the span tree from the event log (this agent only)."""
        return spans_from_events(self.trace_id, self.events)

    def merged_spans(self) -> list[Span]:
        """Derive the full span tree including sub-agent spans."""
        return merge_sub_agent_spans(
            self.trace_id,
            self.events,
            self.messages,
        )

    def model_turns(self) -> list[ModelTurn]:
        """Extract model-visible input/output pairs for training."""
        return model_turns_from_events(self.trace_id, self.events)


def run_trace_from_state(
    *,
    state: Any,
    trace_id: str,
    producer: str,
    meta: Mapping[str, Any] | None = None,
) -> RunTrace:
    """Build a RunTrace from a runtime State-like object."""

    # `state.task` is `str` or a content-block sequence (multimodal); the trace's
    # `task` field is a readable text summary, so non-text blocks (e.g. images)
    # are dropped here — the full task message is preserved in `messages`.
    task = state.task
    task_text = task if isinstance(task, str) else text_of(normalize_content(task))
    return RunTrace(
        trace_id=trace_id,
        producer=producer,
        task=task_text,
        events=list(state.events),
        messages=list(state.messages),
        meta=dict(meta or {}),
    )


def event_record(event: Event) -> dict[str, Any]:
    """Serialize one runtime event into a JSON-safe record.

    `Event.kind` is a real `Literal[...]` dataclass field, so the
    discriminator is preserved by `asdict` and no manual patching is
    needed here -- this is a thin alias over `json_safe` that keeps the
    canonical name for external callers and tests.
    """

    return json_safe(event)


def trace_record(trace: RunTrace) -> dict[str, Any]:
    """Serialize a RunTrace into a JSON-safe dict for export."""
    return {
        "schema": SCHEMA,
        "type": "trajectory",
        "trace_id": trace.trace_id,
        "producer": trace.producer,
        "task": trace.task,
        "events": [event_record(e) for e in trace.events],
        "messages": json_safe(trace.messages),
        "spans": json_safe([span_record(s) for s in trace.spans()]),
        "model_turns": json_safe(trace.model_turns()),
        "meta": trace.meta,
    }


def trace_header_record(
    *,
    trace_id: str,
    producer: str,
    task: str,
    meta: Mapping[str, Any] | None = None,
    schema: str = SCHEMA,
) -> dict[str, Any]:
    """The header line that opens an append-only event-stream trace file."""
    return {
        "type": TRACE_HEADER_TYPE,
        "schema": schema,
        "trace_id": trace_id,
        "producer": producer,
        "task": task,
        "meta": dict(meta) if meta is not None else None,
    }


def event_from_record(record: Mapping[str, Any]) -> Event | None:
    """Rebuild a typed `Event` from one serialized event record.

    The inverse of `event_record` for the events the span/training transforms
    consume (they dispatch on `isinstance`). Returns ``None`` for kinds those
    transforms ignore (``tool_execution_update``, ``hook_fired``) so a folded
    record can skip rebuilding them — they still survive verbatim in the
    reconstructed ``events`` list as their original dicts.
    """
    kind = record.get("kind")
    event: Event | None
    if kind == "message":
        event = MessageEvent(message=message_from_record(record["message"]))
    elif kind == "agent_start":
        event = AgentStartEvent()
    elif kind == "agent_end":
        event = AgentEndEvent(reason=record.get("reason", "done"))
    elif kind == "turn_start":
        event = TurnStartEvent(agent=record.get("agent", ""))
    elif kind == "turn_end":
        event = TurnEndEvent(
            agent=record.get("agent", ""),
            terminated=bool(record.get("terminated", False)),
        )
    elif kind == "model_request":
        event = ModelRequestEvent(
            agent=record.get("agent", ""),
            visible_count=int(record.get("visible_count", 0)),
            llm_message_count=int(record.get("llm_message_count", 0)),
            context_view=dict(record.get("context_view") or {}),
            tools=list(record.get("tools") or []),
            llm_payload=list(record.get("llm_payload") or []),
        )
    elif kind == "model_response":
        event = ModelResponseEvent(
            agent=record.get("agent", ""),
            output_kind=record.get("output_kind", "message"),
            target=record.get("target", "all"),
            tool_call_count=int(record.get("tool_call_count", 0)),
            usage=_token_usage_from_record(record.get("usage")),
            model=record.get("model", ""),
        )
    elif kind == "context_compression":
        event = ContextCompressionEvent(
            agent=record.get("agent", ""),
            summary_message_index=int(record.get("summary_message_index", -1)),
            compressed_message_indices=list(
                record.get("compressed_message_indices") or []
            ),
            active_context_indices=list(record.get("active_context_indices") or []),
            before_tokens=int(record.get("before_tokens", 0)),
            after_tokens=int(record.get("after_tokens", 0)),
            strategy=record.get("strategy", ""),
        )
    elif kind == "tool_execution_start":
        event = ToolExecutionStartEvent(
            tool_call_id=record.get("tool_call_id", ""),
            tool_name=record.get("tool_name", ""),
        )
    elif kind == "tool_execution_end":
        event = ToolExecutionEndEvent(
            tool_call_id=record.get("tool_call_id", ""),
            tool_name=record.get("tool_name", ""),
            is_error=bool(record.get("is_error", False)),
            terminate=bool(record.get("terminate", False)),
        )
    else:
        return None

    stamp: dict[str, Any] = {
        "index": int(record.get("index", -1)),
        "elapsed": float(record.get("elapsed", 0.0)),
    }
    if isinstance(event, MessageEvent):
        stamp["uuid"] = record.get("uuid", "")
        stamp["parent_uuid"] = record.get("parent_uuid")
    return dataclasses.replace(event, **stamp)


def events_from_records(records: Sequence[Mapping[str, Any]]) -> list[Event]:
    """Deserialize a list of event records, dropping kinds with no typed form."""
    return [event for record in records if (event := event_from_record(record))]


def trace_record_from_jsonl(path: str | Path) -> dict[str, Any]:
    """Read a trace file and return the canonical single `trace_record` dict.

    Accepts either on-disk shape transparently: a finished run already written
    as the single canonical record is returned as-is; an append-only event
    stream (header line + event lines) is folded back into the same shape,
    re-deriving spans and model turns from the events. This is the read seam
    that lets the live writer stay append-only while every reader keeps seeing
    one trajectory record.
    """
    records = read_jsonl(path)
    if records and isinstance(records[0].get("events"), list):
        return records[0]
    return _fold_event_stream(records)


def _fold_event_stream(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    header: Mapping[str, Any] = {}
    event_dicts: list[Mapping[str, Any]] = []
    for record in records:
        if record.get("type") == TRACE_HEADER_TYPE:
            header = record
        elif "kind" in record:
            event_dicts.append(record)

    trace_id = str(header.get("trace_id", ""))
    typed = events_from_records(event_dicts)
    messages = [
        record["message"]
        for record in event_dicts
        if record.get("kind") == "message" and "message" in record
    ]
    return {
        "schema": header.get("schema", SCHEMA),
        "type": "trajectory",
        "trace_id": trace_id,
        "producer": header.get("producer", ""),
        "task": header.get("task", ""),
        "events": [dict(record) for record in event_dicts],
        "messages": messages,
        "spans": json_safe(
            [span_record(s) for s in spans_from_events(trace_id, typed)]
        ),
        "model_turns": json_safe(model_turns_from_events(trace_id, typed)),
        "meta": header.get("meta"),
    }
