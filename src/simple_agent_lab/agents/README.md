# Agents

Ready-to-run agent builders for Simple Agent Lab.

This package is the public starter layer for building small agents from the core
runtime. It wires a model provider to common tools such as bash, file reading,
skills, general-purpose sub-agent delegation, and MCP servers.

Most agents are plain `Agent` values that you run yourself with `agent.run(...)`.
Use `AgentSession` only when a capability owns a live resource that must be
opened and closed, such as an MCP server connection.

## Quick Start

```python
from simple_agent_lab.agents import make_agent
from simple_agent_lab.llm import Provider

provider = Provider(
    id="claude",
    api="anthropic-messages",
    model="claude-opus-4-8",
    api_key_env="ANTHROPIC_API_KEY",
)

agent = make_agent(provider, cwd=".", read=True, general_purpose=True)
state, events = agent.run("Summarize this project.")
for _ in events:
    pass
```

`agent.run(...)` returns `(state, events)`. The `events` stream is lazy, so
iterate it to advance the run.

## Plain Agents

`make_agent()` starts with the bash tool enabled and lets you add only the
resource-free pieces you need:

- `read=True` adds the read tool.
- `general_purpose=True` adds a `task` tool backed by a general-purpose
  sub-agent.
- `tools=[...]` appends custom `AgentTool` instances.
- `system_prompt=...` replaces the default composed prompt.

Named plain-agent factories are available for common shapes:

- `make_bash_agent(...)` builds a bash-capable agent.
- `make_bash_task_agent(...)` builds a bash agent with a general-purpose
  sub-agent.
- `make_skill_agent(...)` builds a bash+read agent that advertises skills on
  each `agent.run(...)`.

Skills are resource-free. `make_skill_agent(...)` installs a skills state
initializer through the core `Agent.init_state` hook, so there is no special
skills run loop and no session required:

```python
from simple_agent_lab.agents import make_skill_agent

agent = make_skill_agent(provider, cwd=".")
state, events = agent.run("Use a skill if one fits.")
for _ in events:
    pass
```

Skills require a text task, because the skills directive parser reads the task
string before initializing the conversation.

## Sessions And MCP

Use `agent_session()` when you need to scope resource-bearing toolsets. MCP is
the built-in case: the session opens the MCP connection before building the
agent and closes it when the `with` block exits.

Inside a session, skills still use the same state-initializer mechanism as
`make_skill_agent`; MCP is the part that requires the session lifetime.

## MCP Example

```python
from simple_agent_lab.agents import agent_session
from simple_agent_lab.mcp import MCPServerConfig

fs = MCPServerConfig.stdio(
    "fs",
    "npx",
    "-y",
    "@modelcontextprotocol/server-filesystem",
    ".",
)

with agent_session(provider, cwd=".", skills=True, mcp_servers=[fs]) as session:
    state, events = session.run("Use the filesystem server to inspect the repo.")
    for _ in events:
        pass
```

`session.run(...)` also returns `(state, events)`. Consume the lazy `events`
stream inside the `with` block so MCP tools remain connected for the full run.

`mcp_session(provider, mcp_servers, ...)` is a named shortcut over
`agent_session()` that presets the MCP servers and forwards the rest of the
options.

## Package Layout

- `starter.py` contains `AgentSession`, `agent_session()`, the named session
  shortcut, and the plain agent factories.
- `toolsets.py` contains the `Toolset` protocol and `MCPToolset`, which adapts
  MCP server connections into session-owned tools.
- `__init__.py` re-exports the supported public API from this package.

## Public API

The main exports are:

- `make_agent(...)`, `make_bash_agent(...)`, `make_bash_task_agent(...)`, and
  `make_skill_agent(...)`.
- `agent_session(...)`, `mcp_session(...)`, and `AgentSession`.
- `SkillConfig`.
- `Toolset` and `MCPToolset`.
- `compose_agent_system_prompt(...)`

See `starter.py` for the complete argument list and implementation details.
