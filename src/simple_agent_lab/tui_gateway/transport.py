"""The wire: newline-delimited JSON-RPC 2.0 frames over stdio.

One JSON object per line, ``\\n``-terminated, UTF-8. The single hard rule is
**stdout carries protocol frames and nothing else** — a stray ``print()``
from any library would corrupt the stream and desync the UI parser. We
enforce it the same way the Hermes gateway does: at import time we capture
the real stdout, then point ``sys.stdout`` at ``sys.stderr`` so accidental
prints become harmless log noise on the UI's stderr channel.

A failed write almost always means the UI went away (its end of the pipe is
closed). That is a normal shutdown, not an error: :meth:`Transport.write`
returns ``False`` for a broken/closed peer and the caller is expected to
exit cleanly. Genuine I/O failures (a full disk, say) re-raise.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Protocol, TextIO

# Capture the real stdout *before* anything redirects it, then send Python's
# default stdout to stderr so library prints can't corrupt the frame stream.
# Order matters: this runs at first import of the package's transport.
_REAL_STDOUT: TextIO = sys.stdout
sys.stdout = sys.stderr

# Errno-ish error markers that all mean "the peer's pipe is gone". We match on
# the exception types plus a couple of message fragments because the exact
# class varies by platform (BrokenPipeError on POSIX, OSError with EINVAL or a
# "closed file" ValueError on shutdown races).
_PEER_GONE_FRAGMENTS = ("broken pipe", "closed file", "i/o operation on closed")


def _is_peer_gone(exc: BaseException) -> bool:
    if isinstance(exc, BrokenPipeError):
        return True
    text = str(exc).lower()
    return any(fragment in text for fragment in _PEER_GONE_FRAGMENTS)


class Transport(Protocol):
    """Minimal sink the server writes JSON-RPC objects to.

    Kept as a Protocol so tests can substitute an in-memory collector and
    drive the server without a real subprocess.
    """

    def write(self, obj: dict[str, Any]) -> bool:
        """Serialize ``obj`` as one JSON line. Return ``False`` if the peer
        is gone (caller should exit cleanly); ``True`` on success."""
        ...


class StdioTransport:
    """Writes frames to the captured real stdout, one JSON object per line."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else _REAL_STDOUT

    def write(self, obj: dict[str, Any]) -> bool:
        # ensure_ascii=False keeps non-ASCII compact and human-readable on the
        # wire; the UI parses UTF-8. separators trim incidental whitespace.
        line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        try:
            self._stream.write(line + "\n")
            self._stream.flush()
            return True
        except (BrokenPipeError, OSError, ValueError) as exc:
            if _is_peer_gone(exc):
                return False
            raise


class InMemoryTransport:
    """Test double: records every frame instead of writing to a pipe."""

    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []

    def write(self, obj: dict[str, Any]) -> bool:
        self.frames.append(obj)
        return True


def make_event(
    event_type: str, session_id: str | None, payload: dict[str, Any]
) -> dict[str, Any]:
    """Build a JSON-RPC *notification* envelope for a gateway → UI event.

    Notifications have no ``id`` (the UI never replies to them). Every event
    carries a ``type`` discriminator the UI switches on, an optional
    ``session_id`` for routing, and a free-form ``payload``.
    """
    params: dict[str, Any] = {"type": event_type, "payload": payload}
    if session_id is not None:
        params["session_id"] = session_id
    return {"jsonrpc": "2.0", "method": "event", "params": params}


def make_result(request_id: Any, result: Any) -> dict[str, Any]:
    """Build a JSON-RPC success response correlated by ``request_id``."""
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def make_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    """Build a JSON-RPC error response. ``id`` is ``None`` for parse errors
    where no id could be recovered."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }
