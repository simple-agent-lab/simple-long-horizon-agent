"""Connect to an MCP server and expose its tools as `AgentTool`s.

The MCP Python SDK is async (`ClientSession` lives inside async context
managers), but Simple Agent Lab's tool boundary is synchronous: a tool's
`execute` runs on a worker thread inside `core.dispatch_tool_calls`. This
module bridges the two with the standard pattern — a dedicated background
thread owns an asyncio event loop that keeps one `ClientSession` open for
the connection's lifetime, and each synchronous `execute` submits a
`call_tool` coroutine to that loop via `run_coroutine_threadsafe`.

Keeping one long-lived session (rather than reconnecting per call) matters
for stdio servers: a reconnect would respawn the server subprocess on
every tool call.

The connection takes a *session factory* — an async context manager that
yields a ready-to-use `ClientSession`. Production builds that factory from
an `MCPServerConfig` (stdio or HTTP); tests inject an in-memory factory, so
the whole bridge is exercised without a subprocess or a socket.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

from simple_agent_lab.messages import TextBlock
from simple_agent_lab.tools import AgentTool, ToolResult, text_result

from .config import MCPServerConfig
from .content import mcp_content_to_blocks

if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp import types as mcp_types


# An async context manager that yields a connected, initialized session.
SessionFactory = Callable[[], AbstractAsyncContextManager["ClientSession"]]


class MCPError(RuntimeError):
    """An MCP connection or tool-call failure surfaced to the caller."""


class MCPConnection:
    """A live MCP session running on a private background event loop.

    Construct via `connect_mcp(config)` for real servers, or directly with
    a `session_factory` (e.g. in tests). Call `open()` before use and
    `close()` when done; the object is also a context manager.
    """

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        name: str,
        init_timeout: float = 30.0,
    ) -> None:
        self._factory = session_factory
        self.name = name
        self._init_timeout = init_timeout

        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session: ClientSession | None = None
        self._serve_task: asyncio.Task[None] | None = None
        self._tools: tuple[mcp_types.Tool, ...] = ()
        self._ready = threading.Event()
        self._close_event: asyncio.Event | None = None
        self._error: BaseException | None = None

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> "MCPConnection":
        """Start the background loop, open the session, and discover tools.

        Atomic: if startup fails or times out, the half-started thread,
        event loop, and (for stdio) server subprocess are torn down before
        the error is raised, so a failed `open()` leaves nothing running and
        the connection can be retried.
        """

        if self._thread is not None:
            return self
        thread = threading.Thread(
            target=self._thread_main,
            name=f"mcp-{self.name}",
            daemon=True,
        )
        self._thread = thread
        thread.start()
        # Give connect + handshake + list_tools room beyond the inner timeout.
        ready = self._ready.wait(self._init_timeout + 10.0)
        if not ready or self._error is not None:
            error = self._error
            self.close()  # tear down the background loop/thread/subprocess
            if error is not None:
                raise MCPError(
                    f"MCP server {self.name!r} failed to start: {error}"
                ) from error
            raise MCPError(f"MCP server {self.name!r} did not become ready in time")
        return self

    def close(self) -> None:
        """Tear down the session, join the thread, and reset for reuse.

        Safe to call when never opened or already closed, and idempotent.
        Resetting the readiness/error state lets a closed connection be
        `open()`ed again cleanly.
        """

        loop = self._loop
        if loop is not None and not loop.is_closed():
            # Ask the serve coroutine to stop: set the close event (clean
            # path) and cancel the task (covers a connect/handshake still
            # hung inside the factory's __aenter__, which the event alone
            # cannot unstick). Both run on the loop thread.
            def _request_stop() -> None:
                if self._close_event is not None:
                    self._close_event.set()
                if self._serve_task is not None:
                    self._serve_task.cancel()

            try:
                loop.call_soon_threadsafe(_request_stop)
            except RuntimeError:
                # Loop is already shutting down; the thread will exit on its own.
                pass
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        self._thread = None
        self._loop = None
        self._session = None
        self._serve_task = None
        self._tools = ()
        self._close_event = None
        self._error = None
        self._ready.clear()

    def __enter__(self) -> "MCPConnection":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- discovery & calls -------------------------------------------------

    @property
    def tools(self) -> tuple[mcp_types.Tool, ...]:
        """The raw MCP tool definitions discovered at connect time."""

        return self._tools

    @property
    def is_connected(self) -> bool:
        """True while the background session is live and usable.

        Goes False after `close()` or when the session dies (e.g. the server
        exits while idle), so callers can tell a dead connection — where
        retrying any tool is futile — from a tool that merely returned an error.
        """

        loop = self._loop
        return self._session is not None and loop is not None and not loop.is_closed()

    def call(
        self,
        tool_name: str,
        arguments: Mapping[str, Any] | None,
        *,
        timeout: float,
        abort: Callable[[], bool] = lambda: False,
    ) -> "mcp_types.CallToolResult":
        """Invoke one MCP tool, blocking until it returns, `timeout`, or `abort`.

        The wait is polled in short slices so an `abort` that flips mid-call
        is observed promptly instead of stalling for the full `timeout`.
        """

        loop = self._loop
        session = self._session
        if loop is None or session is None:
            # Surface the captured startup/transport error (if any) instead of
            # a bare "not open", so a session that died is diagnosable.
            detail = f": {self._error}" if self._error is not None else ""
            raise MCPError(f"MCP server {self.name!r} is not connected{detail}")
        coro = session.call_tool(tool_name, dict(arguments or {}))
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        deadline = time.monotonic() + timeout
        while True:
            if abort():
                future.cancel()
                raise MCPError(f"MCP tool {tool_name!r} on {self.name!r} aborted")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                future.cancel()
                raise MCPError(
                    f"MCP tool {tool_name!r} on {self.name!r} timed out after {timeout:g}s"
                )
            try:
                return future.result(timeout=min(0.1, remaining))
            except FuturesTimeoutError:
                continue

    def agent_tools(
        self,
        *,
        name_prefix: str | None = None,
        call_timeout: float = 60.0,
    ) -> list[AgentTool]:
        """Wrap every discovered MCP tool as an `AgentTool` (see `make_mcp_tools`)."""

        return make_mcp_tools(
            self,
            name_prefix=name_prefix,
            call_timeout=call_timeout,
        )

    # -- background loop internals ----------------------------------------

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run())
        finally:
            loop.close()

    async def _run(self) -> None:
        # Run _serve as a task so close() can cancel it (cancellation, unlike
        # the close event, can unstick a connect hung in the factory).
        self._serve_task = asyncio.ensure_future(self._serve())
        try:
            await self._serve_task
        except asyncio.CancelledError:
            pass

    async def _serve(self) -> None:
        self._close_event = asyncio.Event()
        try:
            async with self._factory() as session:
                self._session = session
                listing = await asyncio.wait_for(
                    session.list_tools(), self._init_timeout
                )
                self._tools = tuple(listing.tools)
                self._ready.set()
                await self._close_event.wait()
        except asyncio.CancelledError:
            # close() cancelled us; unwind the context managers (closing the
            # transport / subprocess) and let the cancellation propagate.
            raise
        except BaseException as exc:  # noqa: BLE001 - reported back via open()
            self._error = exc
        finally:
            self._session = None
            self._ready.set()


def make_mcp_tools(
    connection: MCPConnection,
    *,
    name_prefix: str | None = None,
    call_timeout: float = 60.0,
) -> list[AgentTool]:
    """Wrap each tool discovered on `connection` as a runtime `AgentTool`.

    `name_prefix` is prepended to the model-visible tool name to keep tools
    from different servers distinct; it defaults to ``"<server>_"``. The
    wrapper always calls the server using the tool's original name, so the
    prefix never leaks onto the wire to the MCP server.
    """

    prefix = f"{connection.name}_" if name_prefix is None else name_prefix
    tools = [
        mcp_tool_to_agent_tool(
            connection,
            tool,
            name_prefix=prefix,
            call_timeout=call_timeout,
        )
        for tool in connection.tools
    ]
    # The model-visible name is the dispatch key (core's `tool_by_name` dict),
    # so a duplicate would silently shadow another tool. Fail loud instead.
    seen: set[str] = set()
    for tool in tools:
        if tool.name in seen:
            raise MCPError(
                f"duplicate MCP tool name {tool.name!r} from server "
                f"{connection.name!r}; set a distinct name_prefix to disambiguate"
            )
        seen.add(tool.name)
    return tools


def mcp_tool_to_agent_tool(
    connection: MCPConnection,
    tool: "mcp_types.Tool",
    *,
    name_prefix: str = "",
    call_timeout: float = 60.0,
) -> AgentTool:
    """Wrap a single MCP `Tool` as an `AgentTool` bound to `connection`."""

    wire_name = f"{name_prefix}{tool.name}"
    raw_name = tool.name
    description = tool.description or tool.title or f"MCP tool {tool.name!r}."
    parameters = _normalize_parameters(tool.inputSchema)

    def execute(
        call_id: str,
        args: dict[str, Any],
        abort: Callable[[], bool],
        on_update: Any,
    ) -> ToolResult:
        del call_id, on_update
        if abort():
            return text_result(
                f"MCP tool {wire_name!r} aborted before start.", is_error=True
            )
        try:
            result = connection.call(raw_name, args, timeout=call_timeout, abort=abort)
        except MCPError as exc:
            # If the connection itself is gone, retrying any MCP tool is futile;
            # `terminate` stops the run instead of burning the budget on calls
            # that cannot succeed. A timeout on a still-live connection stays
            # retryable (terminate=False).
            return text_result(
                str(exc), is_error=True, terminate=not connection.is_connected
            )
        except Exception as exc:  # noqa: BLE001 - report any call failure to the model
            return text_result(
                f"MCP tool {wire_name!r} failed: {type(exc).__name__}: {exc}",
                is_error=True,
                terminate=not connection.is_connected,
            )

        blocks = mcp_content_to_blocks(result.content)
        is_error = bool(result.isError)
        if not blocks:
            note = (
                f"MCP tool {wire_name!r} reported an error with no content."
                if is_error
                else f"MCP tool {wire_name!r} returned no content."
            )
            blocks = (TextBlock(note),)
        details: dict[str, Any] | None = None
        if result.structuredContent is not None:
            details = {"structuredContent": result.structuredContent}
        return ToolResult(content=blocks, details=details, is_error=is_error)

    return AgentTool(
        name=wire_name,
        description=description,
        parameters=parameters,
        execute=execute,
        execution_mode="parallel",
        # Bound a hung call slightly above the inner MCP timeout so the
        # runtime's own per-tool guard is the backstop, not the trigger.
        timeout_seconds=call_timeout + 5.0,
    )


def connect_mcp(config: MCPServerConfig) -> MCPConnection:
    """Open a connection to the MCP server described by `config`.

    Returns an already-`open()`ed `MCPConnection`. The caller owns the
    connection's lifetime and should `close()` it (or use it as a context
    manager) when done.
    """

    factory = _session_factory_for(config)
    connection = MCPConnection(
        factory,
        name=config.name,
        init_timeout=config.init_timeout,
    )
    return connection.open()


def _session_factory_for(config: MCPServerConfig) -> SessionFactory:
    if config.transport == "stdio":
        return _stdio_session_factory(config)
    return _http_session_factory(config)


def _stdio_session_factory(config: MCPServerConfig) -> SessionFactory:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=config.command,
        args=list(config.args),
        env=dict(config.env) if config.env else None,
        cwd=config.cwd or None,
    )

    @asynccontextmanager
    async def factory() -> AsyncIterator["ClientSession"]:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    return factory


def _http_session_factory(config: MCPServerConfig) -> SessionFactory:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    @asynccontextmanager
    async def factory() -> AsyncIterator["ClientSession"]:
        async with streamablehttp_client(
            config.url,
            headers=dict(config.headers) if config.headers else None,
        ) as (read, write, _get_session_id):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    return factory


def _normalize_parameters(input_schema: Any) -> dict[str, Any]:
    """Coerce an MCP tool's inputSchema into a JSON-Schema object dict.

    MCP guarantees an object schema, but a server could omit ``properties``;
    providers expect at least an object with a properties map, so fill in a
    minimal shell when needed.
    """

    if isinstance(input_schema, dict) and input_schema:
        schema = dict(input_schema)
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        return schema
    return {"type": "object", "properties": {}}
