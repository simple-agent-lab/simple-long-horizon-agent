---
title: "Source-Tree Self-Evolution Replaces Wrapper-Package Simple Runs"
status: Accepted
date: 2026-06-25
slug: source-tree-self-evolution
---

# Source-Tree Self-Evolution Replaces Wrapper-Package Simple Runs

## Context

The simple SWE-bench self-evolution recipe previously exposed a small staged
Python package under `agent/` as the editable target. That was useful for early
tests, but it made `editable_components: [everything]` misleading: candidates
could change the wrapper package while leaving the real Simple Agent Lab source
unchanged.

The project now needs the user-facing simple recipe to represent true framework
self-evolution: candidates should edit the implementation that task agents
actually import from `src/simple_agent_lab/`.

## Decision

The user-facing simple recipe evolves `src/simple_agent_lab/**/*.py` through a
framework-level `source_tree` surface and an agentic `source_tree_agent`
strategy.

Candidate generation runs in an isolated copied source tree. The selected
parent version is overlaid into that tree before the meta-agent edits files.
Candidate rollout stages the selected source under
`input/source_tree/src/simple_agent_lab/` and prepends
`/agent/run/input/source_tree/src` to `PYTHONPATH` before the benchmark runner
imports `simple_agent_lab`.

The wrapper-package surface remains available as lower-level compatibility and
test support, but it is no longer the default simple recipe path.

## Consequences

Simple recipe candidates can now modify the real agent framework: tools,
starter factories, context projection, memory, LLM bridge behavior, and other
code under `src/simple_agent_lab/`.

Rollouts must treat candidate source import order as part of the execution
contract. Tests cover both artifact staging and candidate `PYTHONPATH`
propagation.

The source-tree strategy intentionally rejects edits outside
`src/simple_agent_lab/` and only stages Python source files for this phase.
Recipe policy, configs, tests, docs, and benchmark glue remain outside the
candidate edit scope.

## Alternatives Considered

- Keep wrapper-package evolution as the simple recipe default. This was rejected
  because it continued to produce misleading "whole agent" experiments that did
  not evolve the real framework source.
- Let the meta-agent edit the entire repository. This was rejected for this
  phase because recipe policy, configs, tests, docs, and output artifacts need
  different safety and review rules than framework source.
- Install candidate source as an editable package inside the container. This
  remains possible later, but prepending the staged candidate `src/` directory to
  `PYTHONPATH` is simpler and directly testable now.
