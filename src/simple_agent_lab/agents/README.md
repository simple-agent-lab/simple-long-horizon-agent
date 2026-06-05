# Agents

Ready-to-run agent builders for Simple Agent Lab.

This package wires the core runtime to common tools such as bash, file reading,
skills, sub-agent exploration, and MCP servers. Use it when you want a small
agent you can run from a script without assembling the runtime pieces by hand.

## Quick Start

```python
from simple_agent_lab.agents import agent_session
from simple_agent_lab.llm import Provider

provider = Provider(
    id="claude",
    api="anthropic-messages",
    model="claude-opus-4-8",
    api_key_env="ANTHROPIC_API_KEY",
)

with agent_session(provider, cwd=".", read=True, explorer=True) as session:
    state, events = session.run("Summarize this project.")
    for _ in events:
        pass
```

The `events` stream is lazy, so consume it inside the `with` block. This matters
most when the agent uses MCP servers or other resources that need to be closed
after the run.

## Capabilities

`agent_session()` starts with the bash tool enabled and lets you add only the
pieces you need:

- `read=True` adds the read tool.
- `explorer=True` adds a `task` tool backed by a small explorer sub-agent.
- `skills=True` enables the skills loop and automatically adds `read`.
- `mcp_servers=[...]` connects MCP servers and exposes their tools.
- `tools=[...]` appends custom `AgentTool` instances.
- `system_prompt=...` replaces the default composed prompt.

Example with an MCP filesystem server:

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

### Named shortcuts

`skill_session(provider, ...)` and `mcp_session(provider, servers, ...)` are thin
wrappers over `agent_session()` that preset the `skills` / `mcp_servers`
capability and forward every other keyword, so composition still works (e.g.
`skill_session(provider, mcp_servers=[fs])`).

## Plain Agent Factories

Most callers should use `agent_session()`. If your application owns the run loop
and does not need resource-bearing tools, use the plain factories instead:

```python
from simple_agent_lab.agents import make_bash_agent

agent = make_bash_agent(provider, cwd=".")
state, events = agent.run("List the files.")
for _ in events:
    pass
```

Available factories:

- `make_bash_agent(...)` builds a bash-capable agent.
- `make_bash_task_agent(...)` builds a bash agent with an explorer sub-agent.
- `make_agent(...)` builds a resource-free agent from composable capability
  flags.

## API

The main exports are:

- `agent_session(...)` and `AgentSession`
- `skill_session(...)` and `mcp_session(...)`
- `SkillConfig`
- `Toolset` and `MCPToolset`
- `make_agent(...)`, `make_bash_agent(...)`, and `make_bash_task_agent(...)`
- `compose_agent_system_prompt(...)`

See `starter.py` for the complete argument list and implementation details.
