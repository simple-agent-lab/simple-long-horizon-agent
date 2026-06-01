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

Importing this package pulls in the `mcp` SDK; if the extra is not
installed, that import raises a clear `ModuleNotFoundError`.
"""

from __future__ import annotations

from .client import (
    MCPConnection,
    MCPError,
    SessionFactory,
    connect_mcp,
    make_mcp_tools,
    mcp_tool_to_agent_tool,
)
from .config import MCPServerConfig, Transport
from .content import mcp_content_to_blocks


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
