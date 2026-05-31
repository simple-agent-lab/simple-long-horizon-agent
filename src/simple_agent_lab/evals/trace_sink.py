"""Trace sinks: where the container half pushes live trace records.

ADR 0017 moves live trace from "host tails a bind-mounted file" to "container
pushes records to a sink." The sink is the seam that makes that portable:

- `FileTraceSink` rewrites the canonical single-record ``trajectory.jsonl``
  (the exact shape `trajectory.write_canonical_trace` produces), so under a
  bind mount the host viewer keeps tailing one file — behavior-preserving.
- `HttpTraceSink` is the cloud path: the same records POSTed to a collector,
  no shared filesystem required. Defined as a stub here; see ADR 0017.

A sink receives whole canonical trace *records* (not per-event deltas) so the
file shape and the wire shape stay identical and any consumer
(`read_jsonl`, the viewer, `jq`) works against either.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..trajectory import write_jsonl_atomic


class FileTraceSink:
    """Write the latest canonical trace record to a single JSONL file.

    Matches today's incremental writer: one atomically-rewritten record per
    file. Under a bind mount this is exactly what the host viewer tails.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._last: Mapping[str, Any] | None = None

    def emit(self, record: Mapping[str, Any]) -> None:
        self._last = record
        write_jsonl_atomic(self._path, [dict(record)])

    def close(self) -> None:
        # The final canonical write is performed by the runner; nothing to
        # flush here beyond the last atomic rewrite.
        return None


class HttpTraceSink:
    """Push trace records to a collector URL. Cloud path — not implemented yet.

    Planned shape: POST each canonical record as JSON to ``url`` with a small
    bounded retry, so a remote/cloud backend with no shared filesystem can
    still feed the live viewer. Tracked by ADR 0017's transport cutover.
    """

    def __init__(self, url: str) -> None:
        self._url = url

    def emit(self, record: Mapping[str, Any]) -> None:  # pragma: no cover - stub
        raise NotImplementedError(
            "HttpTraceSink is a documented stub (ADR 0017). Use FileTraceSink "
            "with a bind mount until the cloud transport cutover lands."
        )

    def close(self) -> None:  # pragma: no cover - stub
        return None
