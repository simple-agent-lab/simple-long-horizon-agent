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

from ..config import LIVE_TRACE_PATH
from ..messages import ContentInput
from ..protocols import Event
from .jsonl import _dump_records, write_jsonl_atomic
from .run_trace import (
    RAW_REF_KEY,
    SCHEMA,
    RunTrace,
    event_record,
    event_stream,
    run_trace_from_state,
    trace_header,
)


def _is_raw_blob(value: Any) -> bool:
    return isinstance(value, dict) and ("request" in value or "response" in value)


def _externalize_into(records: list[Any], pool: list[Any]) -> list[Any]:
    """Replace each ``{"raw": blob}`` with a ``{"raw": {"raw_ref": idx}}`` pointer,
    appending the blob to ``pool`` (sequential — in v5 each blob appears once, so
    no content dedup is needed and ``raw_ref`` stays append-stable). ``pool`` is
    extended in place; returns the slimmed records.
    """

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            out: dict[str, Any] = {}
            for key, value in node.items():
                if key == "raw" and _is_raw_blob(value):
                    pool.append(value)
                    out[key] = {RAW_REF_KEY: len(pool) - 1}
                else:
                    out[key] = walk(value)
            return out
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    return [walk(record) for record in records]


def _append_lines(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Append complete compact JSONL lines to ``path`` (created if absent).

    The append-only contract: each line is one whole JSON object, flushed, so a
    tailing reader only ever skips an incomplete trailing line — no whole-file
    rewrite. Mirrors `write_jsonl_atomic`'s on-disk shape via `_dump_records`.
    """

    with path.open("a", encoding="utf-8") as f:
        _dump_records(f, records)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


if TYPE_CHECKING:
    from ..core import Agent
    from ..state import State


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

    raw = LIVE_TRACE_PATH.get()
    return Path(raw) if raw else None


class IncrementalTraceWriter:
    """Periodically append a live ``RunTrace``'s new events to disk for tailing.

    The trace file is the v5 event stream the final writer produces: a header
    line then one line per event. This writer is **append-only** — on each flush
    it writes the header once, then appends only the events added since the last
    flush (and their raw blobs to the sibling pool). A long run therefore never
    rewrites the growing file; a tailing reader only skips an incomplete trailing
    line.

    Cadence: a background daemon thread takes a snapshot of ``state`` every
    ``min_interval_s`` seconds and appends only when the event log has actually
    grown.  Chatty agents with hundreds of fast events per second therefore pay
    at most one append per interval, not one per event.

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
        # Append-only writer state: events / raw blobs already on disk, whether
        # the header line was written, and the running raw pool (so `raw_ref`
        # stays append-stable across flushes).
        self._events_written: int = 0
        self._pool: list[Any] = []
        self._pool_written: int = 0
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
        """Append the events added since the last flush (header written once).

        Returns ``True`` when an append actually happened (state grew, or
        ``force`` was passed and the header still needed writing).  ``False``
        when there was nothing new.  Thread-safe via an internal lock so the
        background thread and an explicit ``flush_now`` from the agent thread
        can't clobber each other.
        """

        with self._write_lock:
            events = list(getattr(self._state, "events", []) or [])
            count = len(events)
            if not force and count == self._events_written and self._header_written:
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
            new_events = trace.events[self._events_written :]
            new_lines = _externalize_into(
                [event_record(e) for e in new_events], self._pool
            )
            new_blobs = self._pool[self._pool_written :]
            try:
                if not self._header_written:
                    # First flush: (re)create the file with the header, clearing
                    # any stale sidecar so a re-run never resolves old raw_refs.
                    write_jsonl_atomic(self._path, [trace_header(trace)])
                    try:
                        raw_trace_path(self._path).unlink()
                    except FileNotFoundError:
                        pass
                    self._header_written = True
                if new_lines:
                    _append_lines(self._path, new_lines)
                if new_blobs:
                    _append_lines(raw_trace_path(self._path), new_blobs)
            except Exception as exc:
                self._report_error(exc)
                return False
            self._events_written = count
            self._pool_written = len(self._pool)
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


def raw_trace_path(path: str | Path) -> Path:
    """Return the provider-raw sidecar path for a trajectory JSONL file."""

    trace_path = Path(path)
    return trace_path.with_name(f"{trace_path.name}.raw.jsonl")


def write_event_stream(path: str | Path, trace: RunTrace) -> None:
    """Write the full v5 stream — header line + one line per event — plus the raw
    pool sidecar, whole-file and atomic.

    Used for the end-of-run write and any whole-file flush (a store-backed eval
    push, a workflow sub-trace). The live writer instead *appends* the new tail
    each flush; both share the `event_stream` serialization so the on-disk shape
    is identical.
    """

    header, lines, pool = event_stream(trace)
    write_jsonl_atomic(Path(path), [header, *lines])
    sidecar = raw_trace_path(path)
    if pool:
        write_jsonl_atomic(sidecar, pool)
    else:
        try:
            sidecar.unlink()
        except FileNotFoundError:
            pass


def write_canonical_trace(
    path: str | Path,
    *,
    state: "State | None" = None,
    trace_meta: TraceMeta | None = None,
) -> None:
    """Write the end-of-run v5 event stream (header + events), whole-file atomic."""

    if state is None or trace_meta is None:
        raise ValueError("write_canonical_trace needs state and trace_meta")
    trace = run_trace_from_state(
        state=state,
        trace_id=trace_meta.trace_id,
        producer=trace_meta.producer,
        meta=_resolve_meta(trace_meta.meta_fn),
    )
    write_event_stream(path, trace)


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

    def stop(self, *, final_flush: bool = True) -> None:
        """Stop the background writer (optionally appending a final flush)."""

        writer = self._writer
        if writer is None:
            return
        writer.stop(final_flush=final_flush)
        self._writer = None
        self._started = False

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
    **agent_run_kwargs: Any,
) -> "tuple[State, list[Event]]":
    """Run ``agent``, append incremental trace events, then write the final stream.

    Returns ``(state, events)`` after the loop finishes; the final on-disk file
    is the canonical v5 event stream (see :func:`write_canonical_trace`).
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
