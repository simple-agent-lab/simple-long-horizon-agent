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


RAW_REF_KEY = "raw_ref"


def split_raw_from_record(
    record: dict[str, Any],
) -> tuple[dict[str, Any], list[Any]]:
    """Externalize provider ``raw`` snapshots from a serialized trace record.

    The provider request/response snapshot stashed on every assistant message's
    ``sidecar["raw"]`` is duplicated across the record's ``messages``,
    ``events`` and ``model_turns`` views, and each turn re-embeds the whole
    growing request history — so a long run's record balloons quadratically (a
    single SWE-bench rollout reached ~200 MB, an unopenable single-line JSON).
    The blob is debug-only: the viewer shows it just in the on-demand "Wire
    debug" panel. So we lift it out instead of storing it inline.

    Every ``{"raw": {"request"/"response": ...}}`` is replaced in place with a
    light ``{"raw": {"raw_ref": <int>}}`` pointer, and the distinct blobs are
    returned as a content-deduplicated pool — the identical copies across the
    three views collapse to one, and identical blobs across turns share a slot.
    Callers persist the slim record beside a ``*.raw.jsonl`` pool file (one blob
    per line, indexed by ``raw_ref``); the viewer resolves the pointer against
    that file on demand. Returns ``(slim_record, pool)``; ``pool`` is empty when
    the record carried no raw blobs (so callers can skip writing the sidecar).
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
                key: (ref_for(value) if key == "raw" and is_raw_blob(value) else walk(value))
                for key, value in node.items()
            }
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    return walk(record), pool


# Per-turn input payloads that are fully reconstructable from `messages` +
# event ordering, so they need not be persisted. Three views each re-embed the
# whole growing request history every turn, so a long run's record balloons
# ~quadratically (a ripgrep rollout reached 176 MB, of which 172 MB was these
# three lists; the real data — `messages` — was 1.5 MB).
#   * `llm_payload`   on each `model_request` event
#   * `input_messages` on each `model_turns` entry
#   * `input`         on each `model_call` span
# The trace viewer rebuilds the payload on demand from the `message` events
# (`synthesizeLlmPayload`), so emptying these costs no information the file
# didn't already carry once in `messages`. The system prompt is the lone
# exception — it lives only in `llm_payload[0]`, not in `messages`, so the
# viewer shows a synthesized ``You are {agent}.`` placeholder; the real prompt
# stays in source. Training/span views are recomputed from live in-memory
# events at write time, never read back from the slimmed file.
_RECONSTRUCTABLE_PAYLOAD_KEYS = ("llm_payload", "input_messages")


def slim_payloads_from_record(record: dict[str, Any]) -> dict[str, Any]:
    """Empty reconstructable per-turn input payloads from a serialized record.

    Returns a new record with `llm_payload` / `input_messages` lists (anywhere,
    including nested sub-agent events) and `model_call` span `input` lists
    replaced by ``[]``. The counts needed to rebuild them (`llm_message_count`,
    `request_event_index`, …) are left intact. See the module note above.
    """

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            is_model_call = node.get("kind") == "model_call"
            return {
                key: (
                    []
                    if key in _RECONSTRUCTABLE_PAYLOAD_KEYS
                    or (key == "input" and is_model_call)
                    else walk(value)
                )
                for key, value in node.items()
            }
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    return walk(record)


def trace_record(trace: RunTrace) -> dict[str, Any]:
    """Serialize a RunTrace into a JSON-safe dict for export.

    The exported record is *slim*: the per-turn input payloads that every
    writer would otherwise re-embed each turn (see `slim_payloads_from_record`)
    are emptied here, at the single canonical serialization point, so every
    consumer — eval harness, workflow sub-traces, live flushes — stays small
    without each having to remember to slim. The viewer rebuilds the payloads
    from `messages` on demand.
    """
    record = {
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
    return slim_payloads_from_record(record)
