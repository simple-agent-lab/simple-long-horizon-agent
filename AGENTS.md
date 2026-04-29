# Agent Collaboration Guide

This file is the primary working contract for AI agents and human contributors in this repository.

## Mission

Simple Agent Lab should make agent systems easier to understand, modify, and teach. Every future implementation choice should preserve that mission.

## Working Principles

- Read this file, `README.md`, and the relevant docs before editing.
- Follow the harness engineering workflow in `docs/context/harness-engineering.md`.
- Follow the nanochat-inspired code style in `docs/context/code-style.md`.
- Treat tests and feedback as first-priority work; define the feedback signal before editing code.
- Prefer small, explicit modules over clever abstractions.
- Preserve beginner readability unless there is a clear reason not to.
- Document important architectural choices in `docs/decisions/`.
- Put reference architecture notes in `docs/reference-architectures/` before using them to drive implementation.
- Write task specs in `docs/tasks/` when work needs to be handed to another human or agent.
- Keep examples small and runnable once code exists.

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
2. Check `docs/context/` for product intent, code style, and harness workflow.
3. Inspect the current source of truth before editing; do not rely on chat memory alone.
4. Define the feedback signal: unit test, smoke run, eval, trace, or review checklist.
5. If the task depends on an external idea, add or update a note in `docs/reference-architectures/`.
6. If the task creates an architectural commitment, add a decision record in `docs/decisions/`.
7. If work needs handoff, write or update a task spec in `docs/tasks/`.
8. Run the narrowest useful check and report the command.
