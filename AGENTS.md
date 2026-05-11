# Agent Collaboration Guide

This file is the primary working contract for AI agents and human contributors in this repository.

## Mission

Simple Agent Lab should make agent systems easier to understand, modify, and teach. Every future implementation choice should preserve that mission.

## Working Principles

- Read this file, `README.md`, and the relevant docs before editing.
- Follow the harness engineering workflow in `docs/agent-native/harness-engineering.md`.
- Follow the nanochat-inspired code style in `docs/agent-native/code-style.md`.
- Treat tests and feedback as first-priority work; define the feedback signal before editing code.
- Prefer small, explicit modules over clever abstractions.
- Preserve beginner readability unless there is a clear reason not to.
- Document important architectural choices in `docs/decisions/`.
- Put reference architecture notes in `docs/reference-architectures/` before using them to drive implementation.
- Keep examples small and runnable once code exists.

## Agent-Native Documentation

This repo uses a small progressive-disclosure documentation system for future agents.

Read this first:

- `docs/agent-native/README.md` is the single agent-facing loading map. It
  routes future agents to doc inventory, operating rules, context docs, ADRs,
  runbooks, and tests as needed.
- `docs/decisions/` is this repo's ADR directory; do not create a parallel
  `docs/adr/` tree.

Maintenance principles:

- Code is the source of truth for behavior that is clearly visible in code. Do not rewrite obvious implementation details into docs.
- Agent-native docs are for context and decision preferences that code does not contain: user or owner guidance, source-of-truth rules, validation workflows, stop conditions, and architectural boundaries.
- Keep the documentation set small and navigable. Reuse existing topic docs whenever possible.
- Do not create tiny one-off docs for a single bug, owner answer, feature, or test unless it creates a new loading trigger.
- If doc roles, freshness, or loading triggers change, update `docs/agent-native/doc-inventory.md`.
- If future agents should load different docs, update the loading map in `docs/agent-native/README.md`.
- Put unresolved owner or external-system facts in `docs/agent-native/owner-questions.md`.
- Use ADRs only for hard-to-reverse decisions with real tradeoffs; link ADRs from the relevant agent-native doc instead of duplicating them.

## Goals

- A minimal agent loop that can be understood by inspection.
- Clear separation between model calls, tools, memory or state, and orchestration.
- Easy customization for students, workshops, and small team experiments.
- Documentation that explains why the system works, not only how to run it.

## Non-Goals

- Do not introduce a heavy framework before the first simple implementation exists.
- Do not add provider-specific complexity unless it teaches a concrete idea.
- Do not optimize for production scale before the educational path is clear.
- Do not hide core behavior behind magic configuration.

## Editing Expectations

- Make the smallest useful change.
- Use plain names and direct control flow.
- Add comments only when they clarify non-obvious intent.
- Keep public examples stable and beginner-friendly.
- Avoid implementation work until the architecture exploration phase has produced decisions.

## Suggested Agent Workflow

1. Read `README.md` and this file.
2. Use `docs/agent-native/README.md` to choose the relevant context, decision,
   or validation docs.
3. Inspect the current source of truth before editing; do not rely on chat memory alone.
4. Define the feedback signal: unit test, smoke run, eval, trace, or review checklist.
5. If the task depends on an external idea, add or update a note in `docs/reference-architectures/`.
6. If the task creates an architectural commitment, add a decision record in `docs/decisions/`.
7. Run the narrowest useful check and report the command.
