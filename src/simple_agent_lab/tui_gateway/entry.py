"""Process entry point: ``python -m simple_agent_lab.tui_gateway.entry``.

A UI process spawns this and drives it over stdio. The loop is intentionally
boring: announce readiness, then read one JSON line at a time, parse it,
dispatch it, and write any response back. The protocol invariants
(stdout-is-frames-only, broken-pipe-means-exit) live in :mod:`.transport`;
this module just wires the stdin loop to :class:`Gateway`.

Lifecycle:

  1. Emit a ``gateway.ready`` event so the UI can drop its startup timeout.
  2. For each non-blank stdin line: ``json.loads`` it; on a parse error reply
     with JSON-RPC ``-32700`` and keep going (one bad line must not kill the
     session). Otherwise dispatch and write the response if there is one.
  3. stdin EOF (UI closed its write end) ⇒ clean exit.

Any failed transport write means the UI is gone; we exit 0 rather than
spewing tracebacks into a dead pipe.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .server import Gateway
from .transport import StdioTransport, make_error

# JSON-RPC parse-error code (spec-reserved).
ERR_PARSE = -32700


def _load_dotenv(path: Path) -> None:
    """Best-effort, dependency-free ``.env`` loader for the spawning cwd.

    The gateway builds a live provider from env vars (``OPENAI_MODEL`` etc.),
    but it is spawned as a bare subprocess that does not go through
    ``uv run --env-file``, so a project ``.env`` would otherwise be ignored.
    We parse ``KEY=VALUE`` lines here so any UI that spawns the gateway from
    the project root gets credentials for free. Real environment variables
    always win — we never override something already set — so this is a
    fallback, not an authority. Malformed lines are skipped silently; this
    is convenience, not configuration validation.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        # Strip optional surrounding quotes from the value.
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> int:
    # Load a project .env from the cwd the UI spawned us in, before any
    # provider is built. Real env vars take precedence.
    _load_dotenv(Path.cwd() / ".env")

    transport = StdioTransport()
    gateway = Gateway(transport)

    # Handshake: tell the UI we're up. If the very first write fails, the UI
    # never attached — nothing to do but leave.
    if not transport.write(
        {
            "jsonrpc": "2.0",
            "method": "event",
            "params": {"type": "gateway.ready", "payload": {}},
        }
    ):
        return 0

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            if not transport.write(make_error(None, ERR_PARSE, "parse error")):
                return 0
            continue
        response = gateway.dispatch(request)
        if response is not None and not transport.write(response):
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
