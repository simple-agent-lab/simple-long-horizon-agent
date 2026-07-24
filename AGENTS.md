# Agent Collaboration Guide

This file is the primary working contract for AI agents and human contributors in this repository.

## Mission

Simple Agent Lab is the companion repository for *Building Reliable
Long-Horizon Agents: A Survey*. It should turn the paper's definitions,
metrics, benchmark view, and model--harness--environment--evaluation stack into
inspectable and reproducible research artifacts. The implementation should
remain easy to understand, modify, and teach so that reliability claims can be
traced to visible system choices.

## Working Principles

- Read this file, `README.md`, and the relevant docs before editing.
- Use `docs/agent-native/README.md` to load the relevant workflow, style,
  validation, and context docs for the task.
- Treat tests and feedback as first-priority work; define the feedback signal before editing code.
- Prefer small, explicit modules over clever abstractions.
- Preserve beginner readability unless there is a clear reason not to.
- Document important architectural choices in `docs/decisions/`.
- Capture reference-architecture research notes locally under `docs/reference-architectures/` before borrowing a pattern; the directory's contents are gitignored except for the README and template, so notes stay on your disk and only the durable commitment lands in an ADR.
- Keep examples small and runnable once code exists.
- Keep paper-to-code claims explicit. Distinguish what the repository
  implements now, what is planned, and what currently exists only in the paper.

## Environment and Commands

This is a `uv`-managed project (`[tool.uv] managed = true`). Run **every**
command through `uv` so it uses the project's pinned environment.

- Run Python through `uv run`, never a bare `python`/`python3`/`pip`. Examples:
  `uv run python -m unittest ...`, `uv run python scripts/foo.py`,
  `uv run ruff check`.
- Manage dependencies with `uv` (`uv add`, `uv remove`, `uv sync`), not `pip`.
  Edit `pyproject.toml` for dependency-group or optional-dependency changes,
  then `uv sync`.
- Tests use the standard-library `unittest` runner (there is no `pytest`
  dependency). Run them with `uv run python -m unittest`, e.g. a single module:
  `uv run python -m unittest tests.unit.test_programbench_suite -v`.

## Agent-Native Documentation

This repo uses a small progressive-disclosure documentation system for future agents.

Read this first:

- `docs/agent-native/README.md` is the single agent-facing loading map. It
  routes future agents to operating rules, context docs, ADRs, runbooks, and
  tests as needed.
- `docs/decisions/` is this repo's ADR directory; do not create a parallel
  ADR tree.

Maintenance principles:

- Code is the source of truth for behavior that is clearly visible in code. Do not rewrite obvious implementation details into docs.
- Agent-native docs are for context and decision preferences that code does not contain: user or owner guidance, source-of-truth rules, validation workflows, stop conditions, and architectural boundaries.
- Keep the documentation set small and navigable. Reuse existing topic docs whenever possible.
- Do not create tiny one-off docs for a single bug, owner answer, feature, or test unless it creates a new loading trigger.
- If doc roles, freshness, or loading triggers change, update the loading map
  in `docs/agent-native/README.md`.
- If future agents should load different docs, update the same loading map.
- Put unresolved owner or external-system facts in the owner-question doc named
  by the agent loading map.
- Use ADRs only for hard-to-reverse decisions with real tradeoffs; link ADRs from the relevant agent-native doc instead of duplicating them.

## Goals

- A clear paper-to-code map for reliable long-horizon agent research.
- An inspectable reference harness for controlled model, context, memory,
  verification, recovery, and orchestration interventions.
- Reproducible trajectories, benchmark artifacts, and evaluation protocols for
  studying reliable horizon under increasing task pressure.
- A minimal agent loop that can be understood by inspection.
- Clear separation between model calls, tools, memory or state, and orchestration.
- Easy customization for students, workshops, and small team experiments.
- Documentation that explains why the system works, not only how to run it.

## Non-Goals

- Do not introduce a heavy framework before the first simple implementation exists.
- Do not add provider-specific complexity unless it teaches a concrete idea.
- Do not optimize for production scale before the educational path is clear.
- Do not hide core behavior behind magic configuration.
- Do not claim a reliable-boundary shift, full paper reproduction, or empirical
  validation without matched protocols, repeated runs, uncertainty, and
  executable evidence.

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
5. If the task depends on an external idea, capture or update a local note under `docs/reference-architectures/` (the dir is gitignored; only the convention is shared). Promote the durable commitment to an ADR.
6. If the task creates an architectural commitment, add a decision record in `docs/decisions/`.
7. Run the narrowest useful check and report the command.
