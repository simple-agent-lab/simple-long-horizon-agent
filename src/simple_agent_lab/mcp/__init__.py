"""Model Context Protocol (MCP) support for Simple Agent Lab.

Connect to an MCP server and expose its tools as ordinary `AgentTool`s,
so a model can call them through the same runtime path as the built-in
bash and task tools. Multimodal tool results (images, embedded image
resources) ride back through the existing `TextBlock | ImageBlock` tool
boundary unchanged — see `content.py` for the full mapping.

This subpackage is optional. Install the extra to use it::

    uv sync --extra mcp        # or: pip install "simple-agent-lab[mcp]"

Quick start::

    from simple_agent_lab.mcp import MCPServerConfig, connect_mcp

    config = MCPServerConfig.stdio("fs", "npx", "-y",
                                   "@modelcontextprotocol/server-filesystem", ".")
    with connect_mcp(config) as conn:
        tools = conn.agent_tools()            # list[AgentTool]
        agent = make_llm_agent(name="a", provider=p, tools=tools)
        ...

SDK-backed exports pull in the `mcp` SDK lazily; if the extra is not
installed, importing those exports raises a clear `ModuleNotFoundError`.
Pure config helpers under `simple_agent_lab.mcp.config_file` stay importable
without the extra so eval launchers can validate staged JSON before deciding
whether to open a live MCP connection.
"""

from __future__ import annotations

from typing import Any

from .config import MCPServerConfig, Transport

_CLIENT_EXPORTS = {
    "MCPConnection",
    "MCPError",
    "SessionFactory",
    "connect_mcp",
    "make_mcp_tools",
    "mcp_tool_to_agent_tool",
}
_CONTENT_EXPORTS = {"mcp_content_to_blocks"}


__all__ = [
    "MCPConnection",
    "MCPError",
    "MCPServerConfig",
    "SessionFactory",
    "Transport",
    "connect_mcp",
    "make_mcp_tools",
    "mcp_content_to_blocks",
    "mcp_tool_to_agent_tool",
]


def __getattr__(name: str) -> Any:
    if name in _CLIENT_EXPORTS:
        from . import client

        value = getattr(client, name)
    elif name in _CONTENT_EXPORTS:
        from . import content

        value = getattr(content, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
