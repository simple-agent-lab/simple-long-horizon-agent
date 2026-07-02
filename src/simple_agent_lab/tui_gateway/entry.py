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
import sys
from pathlib import Path

from simple_agent_lab.llm.env import load_dotenv

from .server import Gateway
from .transport import StdioTransport, make_error

# JSON-RPC parse-error code (spec-reserved).
ERR_PARSE = -32700


def main() -> int:
    # The gateway builds a live provider from env vars (``OPENAI_MODEL`` etc.),
    # but it is spawned as a bare subprocess that does not go through
    # ``uv run --env-file``, so load the project .env from the cwd the UI
    # spawned us in before any provider is built. Real env vars always win —
    # load_dotenv never overrides something already set.
    load_dotenv(Path.cwd() / ".env")

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
