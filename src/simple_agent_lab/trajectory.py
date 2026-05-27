"""Three-layer trace: Event → Span → Training.

Layer 1 — **Event** (``protocols.py``): append-only runtime log.
Layer 2 — **Span** (this module): structured operations derived from events.
Layer 3 — **Training** (``trace.py``): provider-formatted fine-tuning records.

A ``Span`` is one operation the agent performed.  ``spans_from_events``
is the single extraction function that pairs start/end events into a
hierarchical span tree.  ``RunTrace`` stores the raw events and messages;
spans are computed on demand via ``RunTrace.spans()``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import json

from .protocols import (
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

__all__ = [
    "ModelTurn",
    "RunTrace",
    "Span",
    "event_record",
    "model_turns_from_events",
    "merge_sub_agent_spans",
    "run_trace_from_state",
    "span_record",
    "spans_from_events",
    "trace_record",
    "read_jsonl",
    "write_jsonl",
    "json_safe",
]


SCHEMA = "simple-agent-lab.trajectory.v3"


# ---------------------------------------------------------------------------
# Layer 2: Span
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Span:
    """One operation the agent performed, derived from runtime events."""

    id: str
    parent_id: str | None
    kind: str
    start: float
    end: float
    input: Any | None = None
    output: Any | None = None
    attributes: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Layer 3: ModelTurn (training data)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelTurn:
    """One model-visible input/output pair captured from a run."""

    step_id: str
    agent: str
    input_messages: list[Any]
    output_message: Any
    tools: list[Any]
    meta: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# RunTrace
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Extraction: State → RunTrace
# ---------------------------------------------------------------------------


def run_trace_from_state(
    *,
    state: Any,
    trace_id: str,
    producer: str,
    meta: Mapping[str, Any] | None = None,
) -> RunTrace:
    """Build a RunTrace from a runtime State-like object."""

    return RunTrace(
        trace_id=trace_id,
        producer=producer,
        task=str(state.task),
        events=list(state.events),
        messages=list(state.messages),
        meta=dict(meta or {}),
    )


# ---------------------------------------------------------------------------
# Span extraction
# ---------------------------------------------------------------------------


def spans_from_events(
    trace_id: str,
    events: Iterable[Event],
) -> list[Span]:
    """Build a span tree from runtime events.

    Uses a stack to track open spans.  Start events push onto the stack;
    end events pop and finalize the span.  The parent of each new span is
    the current top of the stack.
    """

    events_list = list(events)
    spans: list[Span] = []
    stack: list[dict[str, Any]] = []
    counters: dict[str, int] = {}

    def _next_id(kind: str) -> str:
        counters[kind] = counters.get(kind, 0) + 1
        return f"{trace_id}.{kind}{counters[kind]}"

    def _parent_id(*, skip_kinds: set[str] | None = None) -> str | None:
        if not stack:
            return None
        if skip_kinds:
            for i in range(len(stack) - 1, -1, -1):
                if stack[i]["kind"] not in skip_kinds:
                    return stack[i]["id"]
            return None
        return stack[-1]["id"]

    for event in events_list:
        if isinstance(event, AgentStartEvent):
            span_id = _next_id("run")
            stack.append(
                {
                    "id": span_id,
                    "kind": "agent_run",
                    "start": event.elapsed,
                    "input": None,
                    "attributes": {},
                }
            )

        elif isinstance(event, AgentEndEvent):
            if stack and stack[-1]["kind"] == "agent_run":
                open_span = stack.pop()
                spans.append(
                    Span(
                        id=open_span["id"],
                        parent_id=_parent_id(),
                        kind="agent_run",
                        start=open_span["start"],
                        end=event.elapsed,
                        input=open_span.get("input"),
                        output=open_span.get("output"),
                        attributes={"reason": event.reason},
                    )
                )

        elif isinstance(event, TurnStartEvent):
            span_id = _next_id("turn")
            stack.append(
                {
                    "id": span_id,
                    "parent_id": _parent_id(),
                    "kind": "turn",
                    "start": event.elapsed,
                    "attributes": {"agent": event.agent},
                }
            )

        elif isinstance(event, TurnEndEvent):
            if stack and stack[-1]["kind"] == "turn":
                open_span = stack.pop()
                spans.append(
                    Span(
                        id=open_span["id"],
                        parent_id=open_span.get("parent_id"),
                        kind="turn",
                        start=open_span["start"],
                        end=event.elapsed,
                        attributes={
                            **open_span.get("attributes", {}),
                            "terminated": event.terminated,
                        },
                    )
                )

        elif isinstance(event, ModelRequestEvent):
            span_id = _next_id("call")
            parent = _parent_id()
            stack.append(
                {
                    "id": span_id,
                    "parent_id": parent,
                    "kind": "model_call",
                    "start": event.elapsed,
                    "input": event.llm_payload,
                    "attributes": {
                        "agent": event.agent,
                        "visible_count": event.visible_count,
                        "llm_message_count": event.llm_message_count,
                        "tools": event.tools,
                    },
                }
            )

        elif isinstance(event, ModelResponseEvent):
            if stack and stack[-1]["kind"] == "model_call":
                open_span = stack.pop()
                spans.append(
                    Span(
                        id=open_span["id"],
                        parent_id=open_span.get("parent_id"),
                        kind="model_call",
                        start=open_span["start"],
                        end=event.elapsed,
                        input=json_safe(open_span.get("input")),
                        output={
                            "kind": event.output_kind,
                            "target": event.target,
                            "tool_call_count": event.tool_call_count,
                        },
                        attributes=open_span.get("attributes"),
                    )
                )

        elif isinstance(event, ToolExecutionStartEvent):
            span_id = _next_id("tool")
            parent = _parent_id(skip_kinds={"tool_call"})
            stack.append(
                {
                    "id": span_id,
                    "parent_id": parent,
                    "kind": "tool_call",
                    "start": event.elapsed,
                    "attributes": {
                        "tool_call_id": event.tool_call_id,
                        "tool_name": event.tool_name,
                    },
                }
            )

        elif isinstance(event, ToolExecutionEndEvent):
            matched = None
            for i in range(len(stack) - 1, -1, -1):
                entry = stack[i]
                if (
                    entry["kind"] == "tool_call"
                    and entry["attributes"].get("tool_call_id") == event.tool_call_id
                ):
                    matched = stack.pop(i)
                    break
            if matched is not None:
                spans.append(
                    Span(
                        id=matched["id"],
                        parent_id=matched.get("parent_id"),
                        kind="tool_call",
                        start=matched["start"],
                        end=event.elapsed,
                        attributes={
                            **matched.get("attributes", {}),
                            "is_error": event.is_error,
                            "terminate": event.terminate,
                        },
                    )
                )

        elif isinstance(event, ContextCompressionEvent):
            span_id = _next_id("compression")
            spans.append(
                Span(
                    id=span_id,
                    parent_id=_parent_id(),
                    kind="compression",
                    start=event.elapsed,
                    end=event.elapsed,
                    attributes={
                        "agent": event.agent,
                        "before_tokens": event.before_tokens,
                        "after_tokens": event.after_tokens,
                        "summary_message_index": event.summary_message_index,
                        "compressed_message_indices": event.compressed_message_indices,
                    },
                )
            )

    return _tree_sort(spans)


# ---------------------------------------------------------------------------
# Sub-agent span merging
# ---------------------------------------------------------------------------


def merge_sub_agent_spans(
    trace_id: str,
    events: Iterable[Event],
    messages: Iterable[Any],
) -> list[Span]:
    """Build a full span tree that inlines sub-agent spans.

    Finds tool_call spans whose tool results carry ``sub_events`` (set by
    ``task_tool``), builds sub-agent spans from those events, and
    re-parents each sub-agent's root span under the corresponding
    tool_call span.
    """

    events_list = list(events)
    parent_spans = spans_from_events(trace_id, events_list)

    sub_events_by_call_id = _collect_sub_events(messages)
    if not sub_events_by_call_id:
        return parent_spans

    tool_spans_by_call_id = {
        s.attributes["tool_call_id"]: s
        for s in parent_spans
        if s.kind == "tool_call" and s.attributes and "tool_call_id" in s.attributes
    }

    all_spans = list(parent_spans)
    for call_id, sub_events in sub_events_by_call_id.items():
        tool_span = tool_spans_by_call_id.get(call_id)
        if tool_span is None:
            continue

        sub_trace_id = f"{trace_id}.{call_id}"
        sub_spans = spans_from_events(sub_trace_id, sub_events)

        for sub_span in sub_spans:
            reparented = Span(
                id=sub_span.id,
                parent_id=tool_span.id
                if sub_span.parent_id is None
                else sub_span.parent_id,
                kind=sub_span.kind,
                start=sub_span.start + tool_span.start,
                end=sub_span.end + tool_span.start,
                input=sub_span.input,
                output=sub_span.output,
                attributes=sub_span.attributes,
            )
            all_spans.append(reparented)

    return _tree_sort(all_spans)


def _tree_sort(spans: list[Span]) -> list[Span]:
    """Sort spans in depth-first preorder (parent before children)."""

    by_parent: dict[str | None, list[Span]] = {}
    for s in spans:
        by_parent.setdefault(s.parent_id, []).append(s)
    for children in by_parent.values():
        children.sort(key=lambda s: (s.start, -s.end))

    result: list[Span] = []

    def walk(parent_id: str | None) -> None:
        for span in by_parent.get(parent_id, []):
            result.append(span)
            walk(span.id)

    walk(None)

    seen_ids = {s.id for s in result}
    for s in spans:
        if s.id not in seen_ids:
            result.append(s)
    return result


def _collect_sub_events(messages: Iterable[Any]) -> dict[str, list[Any]]:
    """Extract sub-agent events from tool_result messages."""

    result: dict[str, list[Any]] = {}
    for message in messages:
        if getattr(message, "kind", None) != "tool_result":
            continue
        data = getattr(message, "data", None)
        if not isinstance(data, dict):
            continue
        details = data.get("details")
        if not isinstance(details, dict):
            continue
        for call_id, call_details in details.items():
            if not isinstance(call_details, dict):
                continue
            sub_events = call_details.get("sub_events")
            if isinstance(sub_events, list) and sub_events:
                result[str(call_id)] = sub_events
    return result


# ---------------------------------------------------------------------------
# Layer 3: model turn extraction
# ---------------------------------------------------------------------------


def model_turns_from_events(
    trace_id: str,
    events: Iterable[Event],
) -> list[ModelTurn]:
    """Extract model-visible input/output pairs from runtime events.

    Each ``ModelRequestEvent`` → next assistant ``MessageEvent`` pair
    becomes one ``ModelTurn`` suitable for supervised fine-tuning.
    """

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


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def event_record(event: Event) -> dict[str, Any]:
    """Serialize one runtime event with its discriminator intact."""

    record: dict[str, Any] = {"kind": event.kind.value, **json_safe(event)}
    if isinstance(event, ContextCompressionEvent):
        record["active_message_indices"] = event.active_message_indices
    return record


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


def span_record(span: Span) -> dict[str, Any]:
    """Serialize one Span into a JSON-safe dict."""
    record: dict[str, Any] = {
        "id": span.id,
        "parent_id": span.parent_id,
        "kind": span.kind,
        "start": span.start,
        "end": span.end,
    }
    if span.input is not None:
        record["input"] = json_safe(span.input)
    if span.output is not None:
        record["output"] = json_safe(span.output)
    if span.attributes:
        record["attributes"] = json_safe(span.attributes)
    return record


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
