---
title: "Dynamic Workflows Use Generated JavaScript Orchestration"
status: Proposed
date: 2026-06-20
slug: dynamic-workflows-js-orchestration
---

# Dynamic Workflows Use Generated JavaScript Orchestration

## Context

Simple Agent Lab already has small static Python workflows under
`simple_agent_lab.workflow`. Those are useful teaching examples, but they are
not the same architectural idea as Claude Code Dynamic Workflows: in Dynamic
Workflows the agent writes a task-specific JavaScript harness, and a runtime
executes that script while subagents do the model/tool work.

The project needs a research-grade implementation that can run through the eval
framework without changing the core ReAct loop in `core.py`.

## Decision

Add `simple_agent_lab.dynamic_workflows` as an outer harness around the existing
agent runtime:

- A generator agent writes `workflow.js` from the task.
- A Node-based JavaScript runtime executes `workflow.js` with a restricted
  orchestration API.
- The JavaScript API exposes `agent`, `parallel`, `pipeline`, `workflow`,
  `phase`, `log`, `args`, and `budget`.
- Every `agent()` call crosses a Python bridge and runs an ordinary
  `Agent.run(...)` subagent.
- The workflow script does not receive direct filesystem, shell, environment,
  or module-loading capabilities. Subagents get tools through normal Agent
  construction.
- The Node runtime subprocess is launched with a stripped environment and
  Node's permission model enabled, so generated scripts do not inherit eval
  credentials and cannot use filesystem, child-process, worker, native-addon, or
  WASI capabilities through Node APIs.
- The runtime writes `workflow.js`, `workflow_journal.jsonl`,
  `workflow_result.json`, and per-subagent traces under `subagents/`.
- The journal stores completed agent-call results by stable call id or cache
  key so rerunning the same workflow can reuse finished calls.
- Bench integration happens through suite-specific facade agents, so the
  generic eval runner still drives one normal `Agent` and `core.py` stays
  unchanged.

## Consequences

This makes Dynamic Workflows inspectable and repeatable: the plan is executable
code saved as an artifact, intermediate orchestration state lives in JavaScript
variables, and the durable audit trail lives in the journal and traces.

The runtime is intentionally outside the core loop. That keeps the beginner
runtime small, but it means workflow-specific concepts such as phases, call
reuse, and concurrency caps live in `dynamic_workflows`, not in `State` or
`core.run`.

The first bench target is OneMillion-Bench through `LocalProcessBackend` because
it provides a deterministic no-Docker smoke path with the fake provider. Coding
bench suites reuse the same bridge with bash-capable subagents. SWE-bench may
use optional worktree isolation; ProgramBench disables worktrees, preserves its
network isolation for both the Node orchestration process and every subagent,
and writes workflow-owned artifacts outside the scored workspace.

The JavaScript `vm` context shapes the workflow API for normal scripts, while
process isolation limits the blast radius if generated code reaches Node globals.
This is still not a hardened hostile-code sandbox: OS-level process/user
isolation remains the right boundary for adversarial code.

## Alternatives Considered

- Extend the existing static Python workflow functions. Rejected because it
  keeps orchestration as prewritten host code rather than an agent-written
  workflow script.
- Put workflow concepts inside `core.py`. Rejected because the core loop should
  remain the small inspectable Agent/State/Message loop.
- Make each bench implement its own workflow runtime. Rejected because the
  generation, JavaScript execution, journaling, and bridge semantics should be
  shared across suites.
