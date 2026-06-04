# ADR 0022: Consolidate agent presets behind one AgentSession + toolsets

## Status

Accepted

## Context

`src/simple_agent_lab/agents/` grew a subfolder per agent kind (`bash/`,
`bash_task/`, and a `skill/` variant), while two more kinds (`skill`, `mcp`)
lived only as inline code in demo scripts. All four are really the same shape —
`make_llm_agent(...)` with a different tool list — yet the wiring was duplicated
across presets, demos, evals, and tests.

Two of the kinds differ in more than their tool list:

- **skill** swaps the *run path* (`run_with_skills`, which injects a skills
  menu before the task) for the plain `Agent.run`.
- **mcp** introduces a *resource lifecycle*: an `MCPConnection` (background
  thread, and for stdio a subprocess) whose tools are bound to a live
  connection that must stay open while the run's lazy event generator is
  consumed.

So a single "vary `tools=[...]`" factory is insufficient. We wanted one
canonical builder used everywhere, without a declarative config interpreter
(which `AGENTS.md` warns against as "magic configuration").

## Decision

1. **One runner: `AgentSession`.** A context manager opens any `Toolset`s via a
   single `ExitStack`, merges their tools with the static tools, builds the
   `Agent` with the existing `make_llm_agent`, and `.run()` dispatches to
   `run_with_skills` or `Agent.run`. Exiting closes the toolsets. Definition
   (`make_llm_agent`) stays separate from execution (`AgentSession`), echoing
   ADK's agent-vs-runner split.

2. **Resource tools are self-managing `Toolset`s.** A small `Toolset` protocol
   (`agents/toolsets.py`) lets a resource-bearing tool source own its
   open/close. `MCPToolset` wraps an `MCPServerConfig`; the session opens and
   closes it. This is ADK's "drop a toolset into the list, runner owns the
   exit stack" idea, scaled to our runtime, and generalizes to multiple servers
   and future resource tools.

3. **Four kinds become composition presets** (`agents/starter.py`):
   `bash_session`, `bash_task_session` (explorer sub-agent is an ordinary
   `task` tool entry), `skill_session` (a `SkillConfig` flag), and
   `mcp_session` (`MCPToolset` entries). No per-kind class.

4. **Clean cut.** The per-kind subfolders are deleted; their factory names
   (`make_bash_agent`, `make_bash_task_agent`) survive as thin back-compat
   wrappers in `starter.py` returning a plain `Agent` for callers (evals, the
   TUI gateway) that drive their own run loop. All in-repo call sites moved to
   `simple_agent_lab.agents.starter`.

## Consequences

- One place defines how an agent is assembled and run; demos/evals/tests share
  it. Adding a kind is a new preset, not a new subfolder.
- The MCP connection lifecycle is honest and uniform (`with session:`), not a
  bespoke `with connect_mcp(...)` dance at every call site.
- The eval skills path (`src/simple_agent_lab/evals/in_container.py`) still uses
  `system_prompt_with_skills` (menu in the system prompt), which is distinct
  from the interactive `run_with_skills` path; this ADR does not unify those.
- `build_agent` from the design spec was dropped (it only forwarded to
  `make_llm_agent`); `AgentSession` and the wrappers call `make_llm_agent`
  directly.

## Alternatives Considered

- **Declarative spec + `run(spec, task)`.** Most "single entry point," but the
  most interpretive logic, conflicts with the no-magic-config guidance, and
  opening/closing an MCP connection around a lazy generator inside a plain
  function is fragile.
- **Builder only (returns a plain `Agent`).** Smallest, but does not dedup the
  run path or the MCP/skills wiring — the stated goal.
