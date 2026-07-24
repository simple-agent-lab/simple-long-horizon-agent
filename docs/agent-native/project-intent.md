# Project Intent

Read when:

- A task changes audience-facing examples, teaching scope, product positioning,
  or architectural taste.
- A change would trade beginner readability for a broader abstraction.

Do not read for:

- Narrow bug fixes already covered by source code, tests, or ADRs.

## Mission

Simple Agent Lab is the companion research repository for *Building Reliable
Long-Horizon Agents: A Survey*. It turns the paper's conceptual framework into
an inspectable runtime, harness, trace, and evaluation workspace for studying
how reliability changes as task pressure increases.

The project should stay:

- Simple enough to read end to end.
- Modular enough to modify without fear.
- Practical enough to run real experiments.
- Explicit enough to attribute outcomes to model, harness, environment, and
  evaluation choices.
- Honest about which paper artifacts are available, partial, or planned.
- Documented enough for humans and AI agents to continue the work.

## Audience

### Paper Reader and Reproducer

Wants to understand the paper's reliable-horizon framework, inspect its
reference implementation, and reproduce released experiments.

Needs a paper-to-code map, exact configurations, versioned task and protocol
metadata, replayable trajectories, uncertainty-aware analysis, and clear
release status.

### Harness Researcher

Wants to compare context, memory, verification, recovery, orchestration, and
resource-budget choices under matched tasks.

Needs visible intervention points, fixed protocol boundaries, reusable
benchmark adapters, and enough evidence to distinguish model progress from
harness progress.

### Student Learner

Wants to understand how a long-running agent loop works, how actions change
state, how failures propagate, and how verification evidence is recorded.

Needs clear concepts, small deterministic examples, minimal setup, and code
that can be changed without reading a large framework.

### Agent Contributor

Uses an AI coding agent to extend the project.

Needs a stable collaboration contract, clear task specs, decision records, and
context files that connect changes to the paper rather than only listing file
locations.

## Design Principles

### Clarity First

The implementation should be understandable by reading the code directly. Prefer
explicit function calls and plain data structures.

### Hackable By Default

Users should be able to change prompts, tools, model adapters, and orchestration
logic without learning a large internal framework.

### Small Core, Visible Edges

The core agent loop should stay small. Integrations should sit at visible edges
so they can be replaced, removed, or controlled in an ablation.

### Paper-to-Code Traceability

Public artifacts should state which definition, layer, pressure axis, outcome,
or experimental claim they support. Do not use the paper as branding for
unrelated framework growth.

### Evidence Before Claims

One successful trajectory is a debugging example, not evidence of a longer
reliable horizon. Boundary-shift claims require matched tasks, fixed protocols,
repeated runs, uncertainty, and executable verification.

### Learn Before Abstracting

Add abstractions only after repeated patterns are clear. Early code should teach
the shape of the system.

### Agent-Friendly Context

The repository should make it easy for coding agents to understand intent,
constraints, and next steps before editing files.

## Current Phase

The repo now has a canonical small runtime under `src/simple_agent_lab/`, a
provider-agnostic LLM boundary under `src/simple_agent_lab/llm/`, deterministic
local examples, focused tests, append-only traces, context and memory controls,
evidence-based goal loops, and a common runner for SWE-bench, ProgramBench,
Harbor, and OneMillion-Bench adapters.

The reference harness and benchmark substrate are available. The public paper,
six-axis task annotations, matched stress paths, reliability-surface analysis,
and paper-scale repeated evaluations are not yet released. Near-term work
should make those gaps explicit, preserve the small core for attribution, and
prioritize reproducible paper artifacts over general-purpose framework breadth.
