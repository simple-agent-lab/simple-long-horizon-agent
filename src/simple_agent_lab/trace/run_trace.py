"""RunTrace: the provider-neutral trace value plus its record schema.

A ``RunTrace`` bundles the raw event log and messages for one run; spans
and model turns are derived on demand from the span/training layers. The
``*_record`` functions serialize a run into the canonical JSON shape that
the writers in ``live`` persist.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..messages import normalize_content, text_of
from ..model_metadata import PriceBook, RunCost
from ..protocols import Event
from .jsonl import json_safe
from .spans import Span, merge_sub_agent_spans, spans_from_events
from .training import ModelTurn, model_turns_from_events


SCHEMA = "simple-agent-lab.trajectory.v5"


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
    """Serialize one runtime event into a JSON-safe line for the event stream.

    Drops the reconstructable per-turn `llm_payload` (the reader rebuilds the
    request from the message events + the `agents` registry, or reads the
    verbatim wire from the external pool). The small `system_prompt` that rides
    on `agent_start` / the compressor's `model_request` is kept, so readers can
    build `agents`. `Event.kind` is a real `Literal[...]`, so the discriminator
    survives `json_safe` with no manual patching.
    """

    record = json_safe(event)
    record.pop("llm_payload", None)
    return record


RAW_REF_KEY = "raw_ref"


def split_raw_from_record(record: Any) -> tuple[Any, list[Any]]:
    """Externalize provider ``raw`` snapshots from serialized event lines.

    The provider request/response snapshot on each model output's
    ``sidecar["raw"]`` re-embeds the whole growing request history every turn —
    so inlining it balloons a trace (a single SWE-bench rollout reached ~200 MB).
    The blob is debug-only: the viewer shows it just in the on-demand "Wire
    debug" panel. So we lift it out instead of storing it inline.

    Walks the given node (the v5 event-line list, or any dict/list) and replaces
    every ``{"raw": {"request"/"response": ...}}`` with a light
    ``{"raw": {"raw_ref": <int>}}`` pointer, returning the distinct blobs as a
    content-deduplicated pool (identical blobs share a slot). Callers persist the
    slim lines beside a ``*.raw.jsonl`` pool file (one blob per line, indexed by
    ``raw_ref``); the viewer resolves the pointer against that file on demand.
    Returns ``(slim, pool)``; ``pool`` is empty when there were no raw blobs (so
    callers can skip writing the sidecar).
    """

    pool: list[Any] = []
    index: dict[str, int] = {}

    def is_raw_blob(value: Any) -> bool:
        # The provider snapshot specifically — never an arbitrary "raw" key.
        return isinstance(value, dict) and ("request" in value or "response" in value)

    def ref_for(value: Any) -> dict[str, int]:
        digest = hashlib.sha1(
            json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        slot = index.get(digest)
        if slot is None:
            slot = len(pool)
            index[digest] = slot
            pool.append(value)
        return {RAW_REF_KEY: slot}

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {
                key: (
                    ref_for(value)
                    if key == "raw" and is_raw_blob(value)
                    else walk(value)
                )
                for key, value in node.items()
            }
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    return walk(record), pool


def collect_agents(events: Any) -> dict[str, str]:
    """Build the ``agent -> system_prompt`` registry from an event stream.

    The prompt rides on the event that introduces an agent: ``agent_start`` for
    every ``run()`` agent (main and each sub-agent — the latter nested under
    message-event sidecars), and the compressor's ``model_request`` (it has no
    ``agent_start``). First non-empty entry per agent wins.

    This is a *reader* helper: the v5 file does not persist the registry — this
    and the viewer's JS twin derive it from the stream, so a system prompt is
    always the real one and never fabricated. See ADR trajectory-schema-v5.
    """

    agents: dict[str, str] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("kind") in ("agent_start", "model_request"):
                name = node.get("agent")
                prompt = node.get("system_prompt")
                if name and prompt and name not in agents:
                    agents[name] = prompt
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(events)
    return agents


def trace_header(trace: RunTrace) -> dict[str, Any]:
    """The first line of a v5 trajectory stream: run identity + metadata.

    Carries no arrays: ``messages`` / ``spans`` / ``model_turns`` / ``cost`` and
    the ``agents`` registry are all derived by readers from the event lines that
    follow (the viewer already recomputes spans/turns from events). ``task`` is a
    one-line preview only — the full task is the first event's ``task`` message, so
    the header never re-stores the whole (possibly multi-KB) task. See ADR
    trajectory-schema-v5.
    """

    return {
        "schema": SCHEMA,
        "type": "trajectory",
        "trace_id": trace.trace_id,
        "producer": trace.producer,
        "task": trace.task,
        "meta": trace.meta,
    }


def event_stream(
    trace: RunTrace,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[Any]]:
    """Serialize a RunTrace into ``(header, event_lines, raw_pool)``.

    ``event_lines`` is the append-only body (one ``event_record`` per event,
    with ``llm_payload`` dropped); ``split_raw_from_record`` externalizes each
    event's verbatim provider ``raw`` snapshot into ``raw_pool`` (written to the
    sibling ``*.raw.jsonl``), leaving a ``{raw_ref}`` pointer. Writers emit
    ``header`` then the event lines as JSONL; the live writer appends only the
    new tail each flush. See ADR trajectory-schema-v5.
    """

    lines, pool = split_raw_from_record([event_record(e) for e in trace.events])
    return trace_header(trace), lines, pool


def event_stream_bytes(trace: RunTrace) -> tuple[bytes, bytes | None]:
    """Serialize a RunTrace to the v5 on-disk bytes: ``(stream, raw_pool)``.

    ``stream`` is the header line plus one line per event; ``raw_pool`` is the
    sibling ``*.raw.jsonl`` content, or ``None`` when no event carried a raw
    provider snapshot. For byte-oriented sinks (artifact stores); path-oriented
    writers use :func:`simple_agent_lab.trace.live.write_event_stream`.
    """

    header, lines, pool = event_stream(trace)
    stream = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in (header, *lines)
    ).encode("utf-8")
    if not pool:
        return stream, None
    raw = "".join(json.dumps(blob, ensure_ascii=False) + "\n" for blob in pool).encode(
        "utf-8"
    )
    return stream, raw
