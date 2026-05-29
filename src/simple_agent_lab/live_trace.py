"""Incremental trace export for Docker and other long-running agent hosts.

Mount a host-visible directory into the container and point
:class:`LiveTraceSession` (or :func:`run_agent_with_live_trace`) at a path
under that mount. A background thread atomically rewrites a single-record
``trajectory.jsonl`` on a fixed interval so the host trace viewer can poll it
without changing the on-disk schema.

See ``docs/agent-native/docker-live-trace.md`` for the mount + env contract.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import Agent
from .protocols import Event
from .state import State
from .trajectory import (
    SCHEMA,
    IncrementalTraceWriter,
    RunTrace,
    run_trace_from_state,
    trace_record,
    write_jsonl_atomic,
)

__all__ = [
    "LIVE_TRACE_PATH_ENV",
    "LiveTraceSession",
    "TraceMeta",
    "live_trace_path_from_env",
    "run_agent_with_live_trace",
    "write_canonical_trace",
]

# Optional standard env var: container runners read this when --traces is omitted.
LIVE_TRACE_PATH_ENV = "LIVE_TRACE_PATH"

DEFAULT_FLUSH_INTERVAL_S = 2.0


@dataclass(frozen=True)
class TraceMeta:
    """Identity and metadata for a live + final trajectory export."""

    trace_id: str
    producer: str
    meta_fn: Callable[[], Mapping[str, Any] | None] | None = None
    schema: str = SCHEMA


def live_trace_path_from_env() -> Path | None:
    """Return ``LIVE_TRACE_PATH`` when set, else ``None``."""

    raw = os.environ.get(LIVE_TRACE_PATH_ENV, "").strip()
    if not raw:
        return None
    return Path(raw)


def write_canonical_trace(
    path: str | Path,
    *,
    state: State | None = None,
    trace_meta: TraceMeta | None = None,
    record: Mapping[str, Any] | None = None,
) -> None:
    """Write the end-of-run single-record JSONL (atomic)."""

    if record is None:
        if state is None or trace_meta is None:
            raise ValueError(
                "write_canonical_trace needs state and trace_meta when record is omitted"
            )
        trace = run_trace_from_state(
            state=state,
            trace_id=trace_meta.trace_id,
            producer=trace_meta.producer,
            meta=_resolve_meta(trace_meta.meta_fn),
        )
        record = trace_record(trace)
    write_jsonl_atomic(path, [record])


class LiveTraceSession:
    """Context manager for incremental trace export while an agent runs.

    The trace file uses the same single-record JSONL shape as a finished run;
    only the write cadence differs. Call :meth:`stop` with ``final_flush=False``
    when the caller will write the canonical final record separately (typical
    for eval harnesses that enrich ``state`` after the event loop).
    """

    def __init__(
        self,
        path: str | Path,
        state: Any,
        *,
        trace_id: str,
        producer: str,
        meta_fn: Callable[[], Mapping[str, Any] | None] | None = None,
        flush_interval_s: float = DEFAULT_FLUSH_INTERVAL_S,
        on_error: Callable[[BaseException], None] | None = None,
        final_flush_on_exit: bool = False,
    ) -> None:
        self._path = Path(path)
        self._state = state
        self._trace_id = trace_id
        self._producer = producer
        self._meta_fn = meta_fn
        self._flush_interval_s = flush_interval_s
        self._on_error = on_error
        self._final_flush_on_exit = final_flush_on_exit
        self._writer: IncrementalTraceWriter | None = None
        self._started = False

    @property
    def path(self) -> Path:
        return self._path

    def start(self) -> None:
        if self._started:
            return
        self._writer = IncrementalTraceWriter(
            path=self._path,
            state=self._state,
            trace_id=self._trace_id,
            producer=self._producer,
            meta_fn=self._meta_fn,
            min_interval_s=self._flush_interval_s,
            on_error=self._on_error,
        )
        self._writer.start()
        self._started = True

    def stop(
        self,
        *,
        final_flush: bool = True,
        final_record: Mapping[str, Any] | None = None,
    ) -> None:
        """Stop the background writer and optionally flush or write a final record."""

        writer = self._writer
        if writer is None:
            return
        writer.stop(final_flush=final_flush and final_record is None)
        self._writer = None
        self._started = False
        if final_record is not None:
            write_jsonl_atomic(self._path, [final_record])

    def drain(self, events: Iterable[Event]) -> list[Event]:
        """Consume ``events`` while the session is active; return the list."""

        return list(events)

    def __enter__(self) -> LiveTraceSession:
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop(final_flush=self._final_flush_on_exit)


def run_agent_with_live_trace(
    agent: Agent,
    task: str,
    trace_path: str | Path,
    *,
    trace_meta: TraceMeta,
    max_turns: int = 10,
    flush_interval_s: float = DEFAULT_FLUSH_INTERVAL_S,
    on_flush_error: Callable[[BaseException], None] | None = None,
    build_final_record: Callable[[State], Mapping[str, Any]] | None = None,
    **agent_run_kwargs: Any,
) -> tuple[State, list[Event]]:
    """Run ``agent``, stream incremental trace snapshots, then write the final record.

    Returns ``(state, events)`` after the loop finishes. The final on-disk
    record matches :func:`trace_record` unless ``build_final_record`` overrides
    it.
    """

    state, events = agent.run(task, max_turns=max_turns, **agent_run_kwargs)
    meta_fn = trace_meta.meta_fn
    session = LiveTraceSession(
        trace_path,
        state,
        trace_id=trace_meta.trace_id,
        producer=trace_meta.producer,
        meta_fn=meta_fn,
        flush_interval_s=flush_interval_s,
        on_error=on_flush_error,
    )
    session.start()
    try:
        collected = session.drain(events)
    finally:
        session.stop(final_flush=False)
    if build_final_record is not None:
        record = build_final_record(state)
    else:
        trace = run_trace_from_state(
            state=state,
            trace_id=trace_meta.trace_id,
            producer=trace_meta.producer,
            meta=_resolve_meta(meta_fn),
        )
        record = trace_record(trace)
    write_jsonl_atomic(trace_path, [record])
    return state, collected


def default_stderr_flush_error(exc: BaseException) -> None:
    print(
        f"live trace flush failed: {type(exc).__name__}: {exc}",
        file=sys.stderr,
        flush=True,
    )


def _resolve_meta(
    meta_fn: Callable[[], Mapping[str, Any] | None] | None,
) -> dict[str, Any] | None:
    if meta_fn is None:
        return None
    meta = meta_fn()
    return dict(meta) if meta is not None else None


def trace_meta_from_run_trace(trace: RunTrace) -> TraceMeta:
    """Build :class:`TraceMeta` from an existing :class:`RunTrace`."""

    meta = dict(trace.meta or {})
    return TraceMeta(
        trace_id=trace.trace_id,
        producer=trace.producer,
        meta_fn=lambda: meta,
    )
