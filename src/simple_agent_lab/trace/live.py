"""Incremental ("live") trajectory export for Docker and long-running hosts.

This is the genuinely IO/concurrency-heavy edge of the package: a
background daemon thread atomically rewrites a single-record
``trajectory.jsonl`` on a fixed interval so a polling host viewer can tail
it without ever seeing a torn file. Beginners reading the core runtime can
skip this module entirely; nothing in ``core`` depends on it.

Mount a host-visible directory into the container and point
``LiveTraceSession`` (or ``run_agent_with_live_trace``) at a path under that
mount. See ``docs/agent-native/docker-live-trace.md`` for the mount + env
contract.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ..messages import ContentInput
from ..protocols import Event
from .jsonl import write_jsonl_atomic
from .run_trace import (
    SCHEMA,
    RunTrace,
    run_trace_from_state,
    split_raw_from_record,
    trace_record,
)

if TYPE_CHECKING:
    from ..core import Agent
    from ..state import State


# Standard env var: container runners read this when --traces is omitted.
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


class IncrementalTraceWriter:
    """Periodically dump a live ``RunTrace`` to disk so viewers can tail it.

    The trace file is the SAME single-record JSONL the final writer produces:
    one canonical :func:`trace_record` per file, atomically rewritten as the
    run progresses.  Anything reading ``trajectory.jsonl`` (the viewer,
    ``read_jsonl``, ``jq``) keeps working unchanged — they just see a record
    that grows over time.

    Cadence: a background daemon thread takes a snapshot of ``state`` every
    ``min_interval_s`` seconds and rewrites the file only when the event log
    has actually grown.  Chatty agents with hundreds of fast events per
    second therefore pay at most one rewrite per interval, not one per event.

    Lifecycle::

        writer = IncrementalTraceWriter(
            path=traces_path,
            state=state,
            trace_id=trace_id,
            producer="suite:swebench",
            meta_fn=lambda: {...},
            min_interval_s=2.0,
        )
        writer.start()
        try:
            ... drive the agent loop ...
        finally:
            writer.stop()  # also performs one final flush
    """

    def __init__(
        self,
        *,
        path: str | Path,
        state: Any,
        trace_id: str,
        producer: str,
        meta_fn: Callable[[], Mapping[str, Any] | None] | None = None,
        min_interval_s: float = 2.0,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        self._path = Path(path)
        self._state = state
        self._trace_id = trace_id
        self._producer = producer
        self._meta_fn = meta_fn
        self._min_interval_s = max(0.1, float(min_interval_s))
        self._on_error = on_error

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._write_lock = threading.Lock()
        # Tracks how many events were on disk last flush; cheap "did anything
        # change since last write" check so we don't rewrite an unchanged
        # multi-megabyte JSON record every interval.
        self._last_event_count: int = -1

    @property
    def path(self) -> Path:
        return self._path

    def start(self) -> None:
        """Spawn the background flush thread (no-op if already running)."""

        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        thread = threading.Thread(
            target=self._run,
            name="incremental-trace-writer",
            daemon=True,
        )
        self._thread = thread
        thread.start()

    def stop(self, *, final_flush: bool = True) -> None:
        """Signal the background thread to exit and (optionally) flush once more."""

        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self._min_interval_s * 2 + 1.0)
        self._thread = None
        if final_flush:
            try:
                self.flush_now(force=True)
            except Exception as exc:  # pragma: no cover - defensive
                self._report_error(exc)

    def __enter__(self) -> "IncrementalTraceWriter":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    def flush_now(self, *, force: bool = False) -> bool:
        """Serialize the current ``state`` and atomically rewrite the file.

        Returns ``True`` when a write actually happened (state grew, or
        ``force`` was passed).  ``False`` when there was nothing new to
        write.  Thread-safe via an internal lock so the background thread
        and an explicit ``flush_now`` call from the agent thread can't
        clobber each other.
        """

        with self._write_lock:
            events = list(getattr(self._state, "events", []) or [])
            count = len(events)
            if not force and count == self._last_event_count:
                return False

            meta: Mapping[str, Any] | None
            try:
                meta = self._meta_fn() if self._meta_fn is not None else None
            except Exception as exc:  # pragma: no cover - defensive
                self._report_error(exc)
                meta = None

            trace = run_trace_from_state(
                state=self._state,
                trace_id=self._trace_id,
                producer=self._producer,
                meta=meta,
            )
            try:
                write_trace_record_atomic(self._path, trace_record(trace))
            except Exception as exc:
                self._report_error(exc)
                return False
            self._last_event_count = count
            return True

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.flush_now()
            except Exception as exc:  # pragma: no cover - defensive
                self._report_error(exc)
            if self._stop_event.wait(self._min_interval_s):
                return

    def _report_error(self, exc: BaseException) -> None:
        if self._on_error is None:
            return
        try:
            self._on_error(exc)
        except Exception:  # pragma: no cover - defensive
            pass


def write_canonical_trace(
    path: str | Path,
    *,
    state: "State | None" = None,
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
    write_trace_record_atomic(path, record)


def raw_trace_path(path: str | Path) -> Path:
    """Return the provider-raw sidecar path for a trajectory JSONL file."""

    trace_path = Path(path)
    return trace_path.with_name(f"{trace_path.name}.raw.jsonl")


def write_trace_record_atomic(path: str | Path, record: Mapping[str, Any]) -> None:
    """Write a slim trace record plus a sibling raw-provider sidecar when needed."""

    trace_path = Path(path)
    slim, raw_pool = split_raw_from_record(record)
    write_jsonl_atomic(trace_path, [slim])
    sidecar = raw_trace_path(trace_path)
    if raw_pool:
        write_jsonl_atomic(sidecar, raw_pool)
    else:
        try:
            sidecar.unlink()
        except FileNotFoundError:
            pass


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

    def __enter__(self) -> "LiveTraceSession":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop(final_flush=self._final_flush_on_exit)


def run_agent_with_live_trace(
    agent: "Agent",
    task: ContentInput,
    trace_path: str | Path,
    *,
    trace_meta: TraceMeta,
    max_turns: int = 10,
    flush_interval_s: float = DEFAULT_FLUSH_INTERVAL_S,
    on_flush_error: Callable[[BaseException], None] | None = None,
    build_final_record: Callable[["State"], Mapping[str, Any]] | None = None,
    **agent_run_kwargs: Any,
) -> "tuple[State, list[Event]]":
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
        write_jsonl_atomic(trace_path, [build_final_record(state)])
    else:
        write_canonical_trace(trace_path, state=state, trace_meta=trace_meta)
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
