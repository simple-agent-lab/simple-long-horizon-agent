"""Newline-delimited JSON-RPC gateway that fronts the agent runtime for a TUI.

This package is the *backend half* of an out-of-process TUI: a separate UI
process (e.g. a TypeScript front end built on ``@earendil-works/pi-tui``)
spawns ``python -m simple_agent_lab.tui_gateway.entry`` and talks to it over
the child's stdio using **newline-delimited JSON-RPC 2.0** (one JSON object
per line). The split mirrors the Hermes agent TUI architecture: a thin UI
that only renders, and a language-agnostic protocol on the wire so the
backend can stay pure Python.

The gateway is deliberately a thin adapter. All it does is:

  * own a small set of named *sessions*, each wrapping an :class:`Agent`
    and a persistent :class:`State`;
  * dispatch JSON-RPC *methods* the UI calls (``session.create``,
    ``prompt.submit``, ...); and
  * translate the runtime's :data:`~simple_agent_lab.protocols.Event`
    stream (yielded by :func:`simple_agent_lab.core.run`) into JSON-RPC
    *event notifications* the UI renders.

Phase A (this module) does **not** stream token deltas: each assistant
message arrives whole as a single ``message.complete`` event. Token-level
streaming (``message.delta``) is a later increment that would hook the
``llm`` streaming layer; the wire protocol leaves room for it.

See ``docs/agent-native/`` for the protocol reference. The transport and
event-mapping rules live in :mod:`.transport` and :mod:`.server`; the
process entry point is :mod:`.entry`.
"""

from __future__ import annotations

from .server import Gateway
from .transport import StdioTransport, Transport

__all__ = ["Gateway", "StdioTransport", "Transport"]
