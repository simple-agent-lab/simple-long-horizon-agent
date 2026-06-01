# ADR 0018: MCP Servers Are a Tool Source, Wrapped at the Tool Boundary

## Status

Accepted

## Context

Model Context Protocol (MCP) servers expose tools (and resources, prompts)
over a standard JSON-RPC protocol via stdio or Streamable HTTP. Supporting
MCP lets an agent in this repo reuse the growing ecosystem of MCP servers
(filesystem, search, browsers, image renderers, …) instead of hand-writing
each tool. The immediate motivation is *multimodal* MCP: servers that
return images, which the runtime should show to the model.

Two facts about the codebase shape the decision:

1. The runtime already has a multimodal tool boundary. `ToolResult.content`
   and `ToolResultBlock.content` are `tuple[TextBlock | ImageBlock, ...]`
   (ADR 0014), so an image returned by a tool already rides back to the
   model unchanged — the bash tool's `attach` option proves the path.
2. The MCP Python SDK is async (`ClientSession` lives in async context
   managers), while the runtime's tool `execute` is synchronous, run on a
   worker thread by `core.dispatch_tool_calls`.

So the open questions were: *where* does MCP plug in, and *how* do we cross
the async/sync boundary, without disturbing the core runtime (a standing
stop condition — see ADR 0001 and ADR 0009).

## Decision

Treat an MCP server as a **tool source**, wrapped entirely at the existing
tool boundary. No core runtime, message-protocol, or LLM-adapter changes.

- A new optional subpackage `src/simple_agent_lab/mcp/` owns the
  integration. It is gated behind a `mcp` extra; the base install stays
  lean.
- `MCPConnection` holds one long-lived `ClientSession` on a private
  background asyncio event loop (one dedicated thread). Each synchronous
  tool `execute` submits a `call_tool` coroutine via
  `asyncio.run_coroutine_threadsafe`. One session per connection (not per
  call) so a stdio server's subprocess is not respawned on every call.
- Each MCP tool becomes an ordinary `AgentTool` (`make_mcp_tools`). The
  model-visible name is prefixed with the server name to avoid collisions;
  the wrapper always calls the server with the tool's original name.
- A single pure function, `mcp_content_to_blocks`, maps MCP's five content
  shapes to the runtime's two visible block types: text/image map directly
  (the multimodal path); audio, binary resources, and resource links become
  explicit text placeholders that record kind/mime/size rather than being
  dropped.
- The connection is built from an injectable *session factory* (an async
  context manager yielding a ready `ClientSession`). Production builds it
  from stdio/HTTP transports; tests inject the SDK's in-memory transport, so
  the whole bridge is covered without a subprocess or socket.

Scope is **tools only** for now: resources and prompts are out of scope.

## Consequences

- Easier: any MCP server's tools are usable through the same path as the
  built-in tools, and multimodal results work for free because the tool
  boundary was already multimodal.
- Easier to test: content mapping is pure, and the connection is tested over
  an in-memory transport — deterministic, no network, no subprocess.
- Harder / accepted cost: a real dependency (`mcp` and its server stack) and
  an async event loop on a background thread. Because the integration lives
  under `src/` and is type-checked, the CI gate now syncs `--extra mcp` (it
  is not needed for a base runtime install). A hung tool call is bounded by
  the per-call timeout and the runtime's own per-tool timeout backstop.
- Out of scope: MCP resources, prompts, sampling, notifications/streaming
  progress, and surfacing audio as a first-class block. Each can be added
  later without revisiting this boundary.

## Alternatives Considered

- **A new core "remote tool" concept / MCP-aware run loop.** Rejected: it
  would push protocol detail into the core runtime, against ADR 0001/0009.
  Wrapping as `AgentTool` keeps the core unaware that a tool is remote.
- **An MCP-specific LLM adapter.** Rejected: MCP is a tool/context protocol,
  not a model API; it does not belong in the provider-adapter layer.
- **Reconnect per tool call (no background loop).** Rejected: respawns a
  stdio server's subprocess on every call and re-runs the handshake.
- **Run MCP calls on the dispatch worker's own short-lived event loop via
  `asyncio.run`.** Rejected: the session's async context managers must stay
  open across calls; a per-call loop cannot hold them.
- **A separate, non-`ImageBlock` representation for MCP images.** Rejected:
  the runtime already has `ImageBlock`; a parallel type would fork the
  multimodal path for no benefit.
