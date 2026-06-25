"""Three-layer trace: Event → Span → Training.

Layer 1 — **Event** (``protocols.py``): append-only runtime log.
Layer 2 — **Span** (``spans.py``): structured operations derived from events.
Layer 3 — **Training** (``training.py`` / ``openai_export.py``): model-visible
input/output pairs.

The package keeps one import surface (``simple_agent_lab.trace``) but
splits the work by concern so each piece stays small and readable:

- ``render`` — ``print_trace``, the human-readable console view of Layer 1.
- ``jsonl`` — JSON-safe coercion + (atomic) JSONL read/write.
- ``spans`` — ``Span`` model and the event → span tree extraction.
- ``training`` — ``ModelTurn`` model and provider-neutral training pairs.
- ``openai_export`` — OpenAI Chat fine-tuning JSONL export (the package's
  one provider-specific module).
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
from .openai_export import (
    append_openai_training_record,
    openai_training_record,
)
from .render import print_trace
from .run_trace import (
    SCHEMA,
    TRACE_HEADER_TYPE,
    RunTrace,
    event_from_record,
    event_record,
    events_from_records,
    run_trace_from_state,
    trace_header_record,
    trace_record,
    trace_record_from_jsonl,
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
    "TRACE_HEADER_TYPE",
    "TraceMeta",
    "append_openai_training_record",
    "default_stderr_flush_error",
    "event_from_record",
    "event_record",
    "events_from_records",
    "json_safe",
    "live_trace_path_from_env",
    "merge_sub_agent_spans",
    "model_turns_from_events",
    "openai_training_record",
    "print_trace",
    "read_jsonl",
    "run_agent_with_live_trace",
    "run_trace_from_state",
    "span_record",
    "spans_from_events",
    "trace_header_record",
    "trace_meta_from_run_trace",
    "trace_record",
    "trace_record_from_jsonl",
    "write_canonical_trace",
    "write_jsonl",
    "write_jsonl_atomic",
]
