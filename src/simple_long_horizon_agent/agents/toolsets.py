"""Toolsets: tool sources that may own a resource the runner must scope.

A plain tool (bash, read, task) is just an ``AgentTool`` value with no setup
or teardown. Some tool sources, though, need a live resource for the duration
of a run — an MCP server connection is the motivating case. A ``Toolset`` makes
that resource explicit: it is a context manager that, once entered, can hand
back the ``AgentTool``s it provides, and on exit tears the resource down.

``AgentSession`` (see ``starter.py``) opens every toolset through a single
``ExitStack`` before building the agent and closes them when the session
exits, so the connection stays alive exactly as long as the run that uses it —
the borrowed idea from ADK's ``McpToolset`` + exit-stack pattern.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from simple_long_horizon_agent.tools import AgentTool

if TYPE_CHECKING:
    from simple_long_horizon_agent.mcp import MCPConnection, MCPServerConfig


@runtime_checkable
class Toolset(Protocol):
    """A context-managed source of ``AgentTool``s.

    Implementations open any backing resource in ``__enter__``, expose their
    tools via ``tools()`` (only valid while open), and release the resource in
    ``__exit__``. Plain static tools do not need this — they are passed to a
    session directly as an ``AgentTool`` list.
    """

    def __enter__(self) -> "Toolset": ...

    def __exit__(self, *exc: object) -> None: ...

    def tools(self) -> Sequence[AgentTool]: ...


class MCPToolset:
    """Expose one MCP server's tools as a context-managed ``Toolset``.

    ``__enter__`` opens the connection (via ``connect_mcp`` by default; tests
    inject ``connect`` to use an in-memory transport), ``tools()`` wraps the
    server's tools as ``AgentTool``s, and ``__exit__`` closes the connection —
    tearing down the background session (and, for stdio, the server
    subprocess). The ``mcp`` import is deferred to ``__enter__`` so this module
    stays importable without the optional ``mcp`` extra installed.
    """

    def __init__(
        self,
        config: "MCPServerConfig",
        *,
        name_prefix: str | None = None,
        call_timeout: float = 60.0,
        connect: Callable[["MCPServerConfig"], "MCPConnection"] | None = None,
    ) -> None:
        self._config = config
        self._name_prefix = name_prefix
        self._call_timeout = call_timeout
        self._connect = connect
        self._conn: "MCPConnection | None" = None

    def __enter__(self) -> "MCPToolset":
        connect = self._connect
        if connect is None:
            from simple_long_horizon_agent.mcp import connect_mcp

            connect = connect_mcp
        self._conn = connect(self._config)
        return self

    def __exit__(self, *exc: object) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            conn.close()

    def tools(self) -> Sequence[AgentTool]:
        if self._conn is None:
            raise RuntimeError(
                "MCPToolset.tools() requires an open connection; use it as a "
                "context manager or pass it to AgentSession(toolsets=[...])"
            )
        return self._conn.agent_tools(
            name_prefix=self._name_prefix,
            call_timeout=self._call_timeout,
        )
