# ADR 0002: Adopt Harness Engineering Workflow

## Status

Accepted

## Context

Simple Agent Lab is developed with coding agents as first-class contributors.
The project is also meant to teach agent systems, so the repository itself
should demonstrate how to make agent work legible, repeatable, and easy to
repair.

OpenAI's harness engineering writeup argues that agent-first development works
best when the repository contains the maps, constraints, feedback loops, and
checks that agents need to do reliable work. That matches this project's goal:
simple does not mean informal or invisible.

## Decision

Use harness engineering as the default development workflow for Simple Agent
Lab.

The working contract is documented in
[`docs/agent-native/harness-engineering.md`](../agent-native/harness-engineering.md).

Concretely:

- Keep `AGENTS.md` short and use it as the entry point.
- Treat `docs/` as the repository-local system of record.
- Prefer small runnable scripts and focused checks over broad explanations.
- Capture architectural commitments in ADRs.
- Capture handoff-sized work in task specs with acceptance criteria.
- When an agent struggles, improve the harness: docs, examples, scripts, tests,
  or validation commands.

## Consequences

Future changes should update the repository context when they introduce a new
rule, convention, or architecture choice. Chat-only knowledge is not enough.

The project should continue to avoid heavy framework scaffolding. Harness
engineering here means stronger feedback loops and clearer local context, not a
large process layer.

This decision also changes how we evaluate work. A change is not complete just
because files were edited. It should leave behind enough source-of-truth context
and verification evidence for another agent to continue confidently.

## Alternatives Considered

- Keep process guidance only in `AGENTS.md`. Rejected because `AGENTS.md` should
  stay compact.
- Wait until implementation grows before defining the workflow. Rejected because
  agent legibility is easiest to preserve when it is designed into the repo
  early.
