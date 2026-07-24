# MCP integration

Connect to [Model Context Protocol](https://modelcontextprotocol.io)
servers and expose their tools as ordinary `AgentTool`s, so a model calls
them through the same runtime path as the built-in bash and task tools.
Multimodal results (images) ride back unchanged through the existing
`TextBlock | ImageBlock` tool boundary.

This subpackage is **optional**. Install the extra:

```bash
uv sync --extra mcp          # or: pip install "simple-agent-lab[mcp]"
```

## Quick start

```python
from simple_agent_lab import make_llm_agent
from simple_agent_lab.llm import Provider
from simple_agent_lab.mcp import MCPServerConfig, connect_mcp

# A local server over stdio (a subprocess), or a remote one over HTTP:
config = MCPServerConfig.stdio(
    "fs", "npx", "-y", "@modelcontextprotocol/server-filesystem", "."
)
# config = MCPServerConfig.http("remote", "https://example.com/mcp")

provider = Provider(id="claude", api="anthropic-messages",
                    model="claude-opus-4-8", api_key_env="ANTHROPIC_API_KEY")

with connect_mcp(config) as conn:
    tools = conn.agent_tools()                       # list[AgentTool]
    agent = make_llm_agent(name="a", provider=provider, tools=tools)
    state, events = agent.run("List the files in the current directory.")
    for _ in events:
        pass
```

The connection owns a long-lived MCP session on a background event loop;
use it as a context manager (or call `close()`) so the session and any
server subprocess shut down cleanly.

## What maps to what

`make_mcp_tools(conn)` (also `conn.agent_tools()`) wraps each discovered
MCP tool as an `AgentTool`:

- The MCP tool's `inputSchema` becomes the `AgentTool.parameters` JSON
  Schema.
- The model-visible name is prefixed with the server name
  (`"<server>_<tool>"`) so two servers' identically named tools stay
  distinct. The wrapper always calls the server with the tool's original
  name.

`mcp_content_to_blocks(...)` maps an MCP tool result's content to the
runtime's visible blocks:

| MCP content | Runtime block |
| --- | --- |
| text | `TextBlock` |
| image | `ImageBlock` (the multimodal path) |
| embedded text resource | `TextBlock` |
| embedded image resource | `ImageBlock` |
| audio / binary resource / resource link | `TextBlock` placeholder (kind, mime, size) |

The runtime tool boundary carries only `TextBlock` and `ImageBlock` today,
so audio and other binary artifacts become explicit text placeholders
rather than being dropped silently.

## Demo

A runnable, deterministic demo (no API key) starts a local MCP server that
returns a generated PNG and shows the image flowing back as an `ImageBlock`:

```bash
bash runs/demos/run_mcp_agent_demo.sh
bash runs/demos/run_mcp_agent_demo.sh --color gold --save-image /tmp/swatch.png
```

The server is `scripts/mcp_demo_server.py`; the client/agent is
`scripts/run_mcp_agent_demo.py`.

## Scope

Tools only. MCP resources, prompts, and sampling are out of scope for now.
Transports: stdio and Streamable HTTP.
