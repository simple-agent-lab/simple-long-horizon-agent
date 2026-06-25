"""Incremental ("live") trajectory export for Docker and long-running hosts.

This is the genuinely IO/concurrency-heavy edge of the package: a
background daemon thread *appends* new events to an append-only
``trajectory.jsonl`` (header line + one event per line) on a fixed interval
so a polling host viewer can tail it cheaply, then materializes the single
canonical record on stop. Beginners reading the core runtime can skip this
module entirely; nothing in ``core`` depends on it.

Mount a host-visible directory into the container and point
``LiveTraceSession`` (or ``run_agent_with_live_trace``) at a path under that
mount. See ``docs/agent-native/docker-live-trace.md`` for the mount + env
contract.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ..messages import ContentInput, normalize_content, text_of
from ..protocols import Event
from .jsonl import json_safe, write_jsonl_atomic
from .run_trace import (
    SCHEMA,
    RunTrace,
    event_record,
    run_trace_from_state,
    trace_header_record,
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
    """Stream a live trace to disk append-only so viewers can tail it.

    The live file is an append-only event stream: a header line
    (:func:`trace_header_record`) followed by one ``event_record`` per line, in
    record order. Each flush *appends* only the events added since the last one,
    so a long run pays O(new events) per flush instead of re-serializing and
    rewriting the whole growing record every interval (the old single-record
    shape was O(total) per flush, O(n²) over a run).

    On :meth:`stop` with ``final_flush=True`` the file is materialized into the
    single canonical :func:`trace_record` (one atomic rewrite), so a *finished*
    file is byte-for-byte what the non-live writer produces and every existing
    reader keeps working. While a run is in flight the file is the event stream;
    :func:`trace_record_from_jsonl` folds either shape back into one record.

    Crash safety differs from the old atomic rewrite: a crash mid-append can
    leave a torn trailing line, which any JSONL reader skips (it stops at the
    first undecodable tail). The trade is one possibly-lost last event instead
    of an O(n) rewrite every interval.

    Cadence: a background daemon thread appends every ``min_interval_s`` seconds
    only when the event log has actually grown, so chatty agents pay at most one
    append per interval, not one per event.

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
            writer.stop()  # materializes the canonical final record
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
        # How many events are already appended to the stream; the next flush
        # writes only `events[_written_event_count:]`. `_header_written` gates
        # the one-time header line (and selects truncate-vs-append mode).
        self._written_event_count: int = 0
        self._header_written: bool = False

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
        """Stop the background thread; if ``final_flush`` materialize the canonical record.

        ``final_flush=True`` replaces the append-only stream with the single
        canonical :func:`trace_record` so a finished file matches the non-live
        writer. ``final_flush=False`` leaves the stream in place (the caller will
        write the canonical record itself, e.g. after post-run enrichment).
        """

        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self._min_interval_s * 2 + 1.0)
        self._thread = None
        if final_flush:
            try:
                self.write_final()
            except Exception as exc:  # pragma: no cover - defensive
                self._report_error(exc)

    def __enter__(self) -> "IncrementalTraceWriter":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    def flush_now(self, *, force: bool = False) -> bool:
        """Append any new events (and the header on first write) to the stream.

        Returns ``True`` when a write actually happened (the log grew, or the
        header was not yet written), ``False`` when there was nothing new.
        Thread-safe via an internal lock so the background thread and an
        explicit ``flush_now`` from the agent thread can't interleave.
        """

        with self._write_lock:
            events = list(getattr(self._state, "events", []) or [])
            count = len(events)
            new_events = events[self._written_event_count :]
            if not new_events and self._header_written:
                return False
            try:
                self._append_stream(new_events)
            except Exception as exc:
                self._report_error(exc)
                return False
            self._written_event_count = count
            return True

    def write_final(self) -> None:
        """Materialize the canonical single-record trace (atomic), replacing the stream."""

        with self._write_lock:
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
                write_jsonl_atomic(self._path, [trace_record(trace)])
            except Exception as exc:
                self._report_error(exc)

    def _append_stream(self, new_events: list[Event]) -> None:
        """Append the header (once) and ``new_events`` as compact JSONL lines."""

        self._path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        if not self._header_written:
            lines.append(json.dumps(self._header_record(), ensure_ascii=False))
        lines.extend(
            json.dumps(json_safe(event_record(event)), ensure_ascii=False)
            for event in new_events
        )
        if not lines:
            return
        payload = "".join(f"{line}\n" for line in lines)
        # Append mode once the header exists so the stream only ever grows;
        # truncate-write on the first flush establishes a clean file.
        mode = "a" if self._header_written else "w"
        with self._path.open(mode, encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # fsync is best-effort; e.g. on tmpfs it may not be supported.
                pass
        self._header_written = True

    def _header_record(self) -> dict[str, Any]:
        task = getattr(self._state, "task", "")
        task_text = task if isinstance(task, str) else text_of(normalize_content(task))
        meta: Mapping[str, Any] | None
        try:
            meta = self._meta_fn() if self._meta_fn is not None else None
        except Exception as exc:  # pragma: no cover - defensive
            self._report_error(exc)
            meta = None
        return trace_header_record(
            trace_id=self._trace_id,
            producer=self._producer,
            task=task_text,
            meta=meta,
        )

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
    write_jsonl_atomic(path, [record])


class LiveTraceSession:
    """Context manager for incremental trace export while an agent runs.

    While the run is in flight the trace file is the append-only event stream
    (header line + one event per line); on exit it is materialized into the
    single canonical record. Call :meth:`stop` with ``final_flush=False`` when
    the caller will write the canonical final record separately (typical for
    eval harnesses that enrich ``state`` after the event loop) — the stream is
    left in place for that caller to overwrite.
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
