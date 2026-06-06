"""RunTrace: the provider-neutral trace value plus its record schema.

A ``RunTrace`` bundles the raw event log and messages for one run; spans
and model turns are derived on demand from the span/training layers. The
``*_record`` functions serialize a run into the canonical JSON shape that
the writers in ``live`` persist.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..messages import normalize_content, text_of
from ..pricing import PriceBook, RunCost
from ..protocols import Event
from .jsonl import json_safe
from .spans import Span, merge_sub_agent_spans, span_record, spans_from_events
from .training import ModelTurn, model_turns_from_events


SCHEMA = "simple-agent-lab.trajectory.v3"


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

    def run_cost(self, price_book: PriceBook | None = None) -> RunCost:
        """Fold this run's token usage into a per-model dollar rollup.

        Includes sub-agent calls reached through tool-result sidecars. Pass a
        `price_book` to override the built-in rate card.
        """
        return RunCost.from_run(self.events, self.messages, price_book)


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
        "cost": trace.run_cost().as_dict(),
        "meta": trace.meta,
    }
