"""RunTrace: the provider-neutral trace value plus its record schema.

A ``RunTrace`` bundles the raw event log and messages for one run; spans
and model turns are derived on demand from the span/training layers. The
``*_record`` functions serialize a run into the canonical JSON shape that
the writers in ``live`` persist.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..messages import normalize_content, text_of
from ..protocols import Event
from .jsonl import json_safe
from .spans import Span, merge_sub_agent_spans, span_record, spans_from_events
from .training import ModelTurn, model_turns_from_events


SCHEMA = "simple-agent-lab.trajectory.v3"
_DATA_IMAGE_URL_RE = re.compile(
    r"data:(?P<mime>image/[A-Za-z0-9.+-]+);base64,"
    r"(?P<data>[A-Za-z0-9+/_=-]{128,})"
)
_IMAGE_BASE64_RUN_RE = re.compile(
    r"(?<![A-Za-z0-9+/])"
    r"(?P<data>(?:iVBOR|/9j/|R0lGOD|UklGR)[A-Za-z0-9+/_=-]{512,})"
    r"(?![A-Za-z0-9+/])"
)


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
        "meta": trace.meta,
    }
    return _redact_trace_images(record)


def _redact_trace_images(value: Any) -> Any:
    """Remove base64 image payloads from the on-disk trace record.

    Redaction is intentionally trace-only. In-memory messages and model requests
    keep their original image blocks; persisted trajectories keep MIME/size
    metadata without repeatedly storing large PNG/JPEG strings.
    """

    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        image_like = _is_trace_image_mapping(value)
        for key, item in value.items():
            key_text = str(key)
            if image_like and key_text in {"data", "blob"} and isinstance(item, str):
                output[key_text] = _image_payload_placeholder(
                    mime=_mime_from_mapping(value),
                    chars=len(item),
                )
                output[f"{key_text}_redacted"] = True
                continue
            if (
                key_text in {"image_url", "url"}
                and isinstance(item, str)
                and item.startswith("data:image/")
            ):
                output[key_text] = _redact_image_data_urls(item)
                continue
            output[key_text] = _redact_trace_images(item)
        return output
    if isinstance(value, list):
        return [_redact_trace_images(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_trace_images(item) for item in value]
    if isinstance(value, str):
        return _redact_image_base64_runs(_redact_image_data_urls(value))
    return value


def _is_trace_image_mapping(value: Mapping[str, Any]) -> bool:
    kind = str(value.get("kind") or value.get("type") or "").lower()
    if kind in {"image", "input_image"}:
        return True
    mime = _mime_from_mapping(value)
    return mime.startswith("image/") and any(key in value for key in ("data", "blob"))


def _mime_from_mapping(value: Mapping[str, Any]) -> str:
    return str(
        value.get("mime_type")
        or value.get("mimeType")
        or value.get("mime")
        or "image/*"
    )


def _redact_image_data_urls(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return _image_data_url_placeholder(
            mime=match.group("mime"),
            chars=len(match.group("data")),
        )

    return _DATA_IMAGE_URL_RE.sub(replace, text)


def _redact_image_base64_runs(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return _image_payload_placeholder(
            mime="image/*", chars=len(match.group("data"))
        )

    return _IMAGE_BASE64_RUN_RE.sub(replace, text)


def _image_payload_placeholder(*, mime: str, chars: int) -> str:
    return f"[trace image base64 omitted: mime={mime}, chars={chars}]"


def _image_data_url_placeholder(*, mime: str, chars: int) -> str:
    return f"[trace image data URL omitted: mime={mime}, base64_chars={chars}]"
