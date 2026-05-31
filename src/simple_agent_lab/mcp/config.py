"""Connection settings for one MCP server.

Two transports are supported, matching the common deployment shapes:

  * ``stdio`` — launch a local server process and talk over its stdio
    pipes (e.g. ``npx @modelcontextprotocol/server-filesystem``).
  * ``http``  — connect to a remote server over Streamable HTTP.

One frozen dataclass carries both shapes. The unused fields stay empty,
and ``__post_init__`` validates that the fields the chosen transport
needs are present, so a misconfiguration fails at construction with a
clear message instead of deep inside the async client.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal


Transport = Literal["stdio", "http"]


@dataclass(frozen=True)
class MCPServerConfig:
    """How to reach one MCP server, plus per-call budgets.

    `name` is a human label used in traces and as the default tool-name
    prefix (so two servers exposing a ``search`` tool stay distinct).
    `init_timeout` bounds connect + handshake + tool discovery;
    `call_timeout` bounds a single ``tools/call``.
    """

    name: str
    transport: Transport = "stdio"

    # stdio transport
    command: str = ""
    args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None
    cwd: str | None = None

    # http transport
    url: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)

    init_timeout: float = 30.0
    call_timeout: float = 60.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("MCPServerConfig.name must be non-empty")
        if self.transport == "stdio":
            if not self.command:
                raise ValueError(
                    f"stdio MCPServerConfig requires `command` (server {self.name!r})"
                )
        elif self.transport == "http":
            if not self.url:
                raise ValueError(
                    f"http MCPServerConfig requires `url` (server {self.name!r})"
                )
        else:  # pragma: no cover - guarded by the Literal at type-check time
            raise ValueError(
                f"unknown MCP transport {self.transport!r} (server {self.name!r})"
            )
        if self.init_timeout <= 0:
            raise ValueError("init_timeout must be > 0")
        if self.call_timeout <= 0:
            raise ValueError("call_timeout must be > 0")

    @classmethod
    def stdio(
        cls,
        name: str,
        command: str,
        *args: str,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        init_timeout: float = 30.0,
        call_timeout: float = 60.0,
    ) -> "MCPServerConfig":
        """Build a stdio config: ``MCPServerConfig.stdio("fs", "npx", "-y", ...)``."""

        return cls(
            name=name,
            transport="stdio",
            command=command,
            args=tuple(args),
            env=env,
            cwd=cwd,
            init_timeout=init_timeout,
            call_timeout=call_timeout,
        )

    @classmethod
    def http(
        cls,
        name: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        init_timeout: float = 30.0,
        call_timeout: float = 60.0,
    ) -> "MCPServerConfig":
        """Build an HTTP config for a remote Streamable HTTP MCP server."""

        return cls(
            name=name,
            transport="http",
            url=url,
            headers=dict(headers or {}),
            init_timeout=init_timeout,
            call_timeout=call_timeout,
        )
