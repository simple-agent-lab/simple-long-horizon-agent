"""Three-layer trace: Event → Span → Training.

Layer 1 — **Event** (``protocols.py``): append-only runtime log.
Layer 2 — **Span** (``spans.py``): structured operations derived from events.
Layer 3 — **Training** (``training.py``): model-visible input/output pairs.

The package keeps one import surface (``simple_agent_lab.trajectory``) but
splits the work by concern so each piece stays small and readable:

- ``jsonl`` — JSON-safe coercion + (atomic) JSONL read/write.
- ``spans`` — ``Span`` model and the event → span tree extraction.
- ``training`` — ``ModelTurn`` model and training-pair extraction.
- ``run_trace`` — ``RunTrace`` value plus the canonical record schema.
- ``live`` — the IO/concurrency-heavy incremental ("live") export edge.

Pure transforms (``spans``, ``training``, ``run_trace``) sit below the
genuinely dirty ``live`` writer; nothing in the core runtime imports this
package, so it stays a downstream consumer of the event log.
"""

from __future__ import annotations

from .jsonl import (
    json_safe,
    read_jsonl,
    write_jsonl,
    write_jsonl_atomic,
)
from .live import (
    LIVE_TRACE_PATH_ENV,
    IncrementalTraceWriter,
    LiveTraceSession,
    TraceMeta,
    default_stderr_flush_error,
    live_trace_path_from_env,
    run_agent_with_live_trace,
    trace_meta_from_run_trace,
    write_canonical_trace,
)
from .run_trace import (
    SCHEMA,
    RunTrace,
    event_record,
    run_trace_from_state,
    trace_record,
)
from .spans import (
    Span,
    merge_sub_agent_spans,
    span_record,
    spans_from_events,
)
from .training import (
    ModelTurn,
    model_turns_from_events,
)

__all__ = [
    "IncrementalTraceWriter",
    "LIVE_TRACE_PATH_ENV",
    "LiveTraceSession",
    "ModelTurn",
    "RunTrace",
    "SCHEMA",
    "Span",
    "TraceMeta",
    "default_stderr_flush_error",
    "event_record",
    "json_safe",
    "live_trace_path_from_env",
    "merge_sub_agent_spans",
    "model_turns_from_events",
    "read_jsonl",
    "run_agent_with_live_trace",
    "run_trace_from_state",
    "span_record",
    "spans_from_events",
    "trace_meta_from_run_trace",
    "trace_record",
    "write_canonical_trace",
    "write_jsonl",
    "write_jsonl_atomic",
]
