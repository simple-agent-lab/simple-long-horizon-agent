---
title: "A Pluggable State Initializer Makes Skills a Bare-Agent Capability"
status: Accepted
date: 2026-06-05
slug: pluggable-state-init-hook
---

# A Pluggable State Initializer Makes Skills a Bare-Agent Capability

## Status

Accepted

## Context

`add-agent-skills` added agent skills as a *run path*: `run_with_skills(agent, task)`
builds the initial `State` differently from `Agent.run` — it records a skills
menu and any `/mention`ed or preloaded skill bodies *before* the task — then
drives the same `core.run` loop. `consolidate-agent-presets` then consolidated
agent construction behind `AgentSession`, whose `run` branched between
`run_with_skills` and `Agent.run`.

This left skills awkward to expose as a simple builder. Bash is a *tool*, so
`make_bash_agent` returns a bare `Agent` and `agent.run` just works. A skill is
not a tool — its menu lives in the conversation `State`, initialized per run
and gated by per-task directives. So a hypothetical `make_skill_agent`
returning a bare `Agent` would be a footgun: `agent.run(task)` would silently
skip all skills behavior, because `Agent.run` hardcoded plain single-task state
initialization and had no hook to do otherwise. The interim workarounds - a
`skill_session` wrapper, then a `SkillsAgent` runner type — both added a
parallel construct whose only job was to call `run_with_skills` instead of
`agent.run`.

The real difference between "plain run" and "skills run" is **how the initial
`State` is built**, nothing else. That is a property an `Agent` could carry.

## Decision

1. **Add an optional `init_state` hook to the core `Agent`.** `StateInitFn =
   Callable[[Agent, ContentInput], State]`. `Agent.run` uses
   `self.init_state(self, task)` when set, else a `_default_init_state` that
   records just the task message (the prior behavior). The `run` loop is
   untouched — it drives whatever `State` the initializer produced. The
   dependency points inward: `core` never imports the skills package; a higher
   layer *installs* a callable.

2. **Skills are state initialization, expressed once.** `skills.runtime`
   exposes `init_state_with_skills(agent, task, ...) -> State` (the
   menu/body-recording logic, factored out of `run_with_skills`, which now calls
   it). The agents layer adapts a `SkillConfig` into that `StateInitFn` via
   `_skills_init_state`.

3. **`make_skill_agent()` returns a bare `Agent`.** Symmetric with
   `make_bash_agent`: it builds a bash+read agent and installs the skills state
   initializer, so a plain `agent.run(task)` is skills-aware with no session, no
   runner wrapper, and no separate call. The `SkillsAgent` runner and
   `skills_agent` factory are removed.

4. **`AgentSession` keeps only what owns a live resource.** Its `run` no longer
   branches on skills — `__enter__` installs the skills state initializer on the
   built agent, and `agent_session(skills=...)`/`mcp_session(...)` reuse it. The
   session exists for MCP, whose connection must be opened and closed around the
   lazy event generator.

## Consequences

- The two axes are now explicit and orthogonal: **state initialization** (a
  build-time property of the agent — bash, read, explorer, skills) versus
  **resource lifetime** (a `with`-scoped concern — MCP). Skills sit firmly on
  the first axis.
- `make_skill_agent` is footgun-free: there is no "bare agent that forgets its
  skills," because the skill behavior rides on `agent.run` itself.
- `run_with_skills` stays as a one-call convenience for callers that drive a
  skills run directly (the interactive demo, the gateway); it is now a thin
  wrapper over `init_state_with_skills` + `core.run`, so there is a single
  implementation of the initialization logic.
- The core primitive grows one optional field and one branch in `run`. This is a
  small, inspectable addition; it gives `agent.run` a bit of non-obvious
  behavior when an initializer is set, but the initializer is an explicit
  attribute on the agent, not hidden configuration.
- Multimodal tasks are rejected by the skills state initializer (the directive
  parser needs a string); plain agents are unaffected.

## Alternatives Considered

- **Keep the `SkillsAgent` runner.** No core change, but it is a parallel
  construct whose `.run` only exists to call `run_with_skills`, and it is
  asymmetric with `make_bash_agent` — the friction that prompted this ADR.
- **Model skills as a tool.** Would let a bare agent use skills via ordinary
  tool calls, but it is a larger change to the skills subsystem and drops the
  `/mention` / `/no-skills` directive handling that state initialization gives
  for free.
- **Bake a static menu into the system prompt at build time.** Makes a bare
  agent partly skills-aware, but loses per-task `/mention` preloading and
  `/no-skills`, and splits the skills logic across two mechanisms.
