"""Layer 2 — Span: structured operations derived from the event log.

``spans_from_events`` is the single extraction function that pairs
start/end events into a hierarchical span tree. This module is pure: it
reads the append-only event log and produces ``Span`` values, with no IO.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..protocols import (
    AgentEndEvent,
    AgentStartEvent,
    ContextCompressionEvent,
    Event,
    ModelRequestEvent,
    ModelResponseEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from .jsonl import json_safe


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


@dataclass
class _OpenSpan:
    """A span still on the extraction stack, awaiting its end event.

    The in-flight counterpart to the frozen `Span`: a typed accumulator
    (not an untyped dict) so the start/end handlers below read `.kind` /
    `.parent_id` instead of stringly keys. `parent_id` is fixed at push
    time for every kind; fields a given kind doesn't use (e.g. `input`,
    set only by model_call) stay at their defaults. The closed `Span`'s
    `output` is built from the end event, so it has no open-span field.
    """

    id: str
    kind: str
    start: float
    parent_id: str | None = None
    input: Any | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


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
    stack: list[_OpenSpan] = []
    counters: dict[str, int] = {}

    def _next_id(kind: str) -> str:
        counters[kind] = counters.get(kind, 0) + 1
        return f"{trace_id}.{kind}{counters[kind]}"

    def _parent_id(*, skip_kinds: set[str] | None = None) -> str | None:
        skip = skip_kinds or set()
        for i in range(len(stack) - 1, -1, -1):
            if stack[i].kind not in skip:
                return stack[i].id
        return None

    for event in events_list:
        if isinstance(event, AgentStartEvent):
            stack.append(
                _OpenSpan(
                    id=_next_id("run"),
                    kind="agent_run",
                    start=event.elapsed,
                    parent_id=_parent_id(),
                )
            )

        elif isinstance(event, AgentEndEvent):
            if stack and stack[-1].kind == "agent_run":
                open_span = stack.pop()
                spans.append(
                    Span(
                        id=open_span.id,
                        parent_id=open_span.parent_id,
                        kind="agent_run",
                        start=open_span.start,
                        end=event.elapsed,
                        attributes={"reason": event.reason},
                    )
                )

        elif isinstance(event, TurnStartEvent):
            stack.append(
                _OpenSpan(
                    id=_next_id("turn"),
                    kind="turn",
                    start=event.elapsed,
                    parent_id=_parent_id(),
                    attributes={"agent": event.agent},
                )
            )

        elif isinstance(event, TurnEndEvent):
            if stack and stack[-1].kind == "turn":
                open_span = stack.pop()
                spans.append(
                    Span(
                        id=open_span.id,
                        parent_id=open_span.parent_id,
                        kind="turn",
                        start=open_span.start,
                        end=event.elapsed,
                        attributes={
                            **open_span.attributes,
                            "terminated": event.terminated,
                        },
                    )
                )

        elif isinstance(event, ModelRequestEvent):
            stack.append(
                _OpenSpan(
                    id=_next_id("call"),
                    kind="model_call",
                    start=event.elapsed,
                    parent_id=_parent_id(),
                    input=event.llm_payload,
                    attributes={
                        "agent": event.agent,
                        "visible_count": event.visible_count,
                        "llm_message_count": event.llm_message_count,
                        "tools": event.tools,
                    },
                )
            )

        elif isinstance(event, ModelResponseEvent):
            if stack and stack[-1].kind == "model_call":
                open_span = stack.pop()
                spans.append(
                    Span(
                        id=open_span.id,
                        parent_id=open_span.parent_id,
                        kind="model_call",
                        start=open_span.start,
                        end=event.elapsed,
                        input=json_safe(open_span.input),
                        # Response facts -- including the cost primitives
                        # (model + usage) -- live together in `output`;
                        # request-side metadata stays in `attributes`. Both
                        # cost fields are always present (usage may be None)
                        # and json_safe'd here like `input`, so an in-memory
                        # span never holds a raw dataclass under an output key.
                        output=json_safe(
                            {
                                "kind": event.output_kind,
                                "target": event.target,
                                "tool_call_count": event.tool_call_count,
                                "model": event.model,
                                "usage": event.usage,
                            }
                        ),
                        attributes=open_span.attributes,
                    )
                )

        elif isinstance(event, ToolExecutionStartEvent):
            stack.append(
                _OpenSpan(
                    id=_next_id("tool"),
                    kind="tool_call",
                    start=event.elapsed,
                    parent_id=_parent_id(skip_kinds={"tool_call"}),
                    attributes={
                        "tool_call_id": event.tool_call_id,
                        "tool_name": event.tool_name,
                    },
                )
            )

        elif isinstance(event, ToolExecutionEndEvent):
            matched = None
            for i in range(len(stack) - 1, -1, -1):
                entry = stack[i]
                if (
                    entry.kind == "tool_call"
                    and entry.attributes.get("tool_call_id") == event.tool_call_id
                ):
                    matched = stack.pop(i)
                    break
            if matched is not None:
                spans.append(
                    Span(
                        id=matched.id,
                        parent_id=matched.parent_id,
                        kind="tool_call",
                        start=matched.start,
                        end=event.elapsed,
                        attributes={
                            **matched.attributes,
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
                        "strategy": event.strategy,
                        "before_tokens": event.before_tokens,
                        "after_tokens": event.after_tokens,
                        "summary_message_index": event.summary_message_index,
                        "compressed_message_indices": event.compressed_message_indices,
                    },
                )
            )

    return _tree_sort(spans)


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
        sidecar = getattr(message, "sidecar", None)
        if not isinstance(sidecar, dict):
            continue
        details = sidecar.get("details")
        if not isinstance(details, dict):
            continue
        for call_id, call_details in details.items():
            if not isinstance(call_details, dict):
                continue
            sub_events = call_details.get("sub_events")
            if isinstance(sub_events, list) and sub_events:
                result[str(call_id)] = sub_events
    return result


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
