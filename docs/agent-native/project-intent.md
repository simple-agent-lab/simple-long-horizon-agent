# Project Intent

Read when:

- A task changes audience-facing examples, teaching scope, product positioning,
  or architectural taste.
- A change would trade beginner readability for a broader abstraction.

Do not read for:

- Narrow bug fixes already covered by source code or tests.

## Mission

Simple Agent Lab exists to help people understand and modify agent systems
without starting from a large framework.

The project should stay:

- Simple enough to read end to end.
- Modular enough to modify without fear.
- Practical enough to run real experiments.
- Documented enough for humans and AI agents to continue the work.

## Audience

### Student Learner

Wants to understand how an agent loop works, how tools are called, and how state
changes over time.

Needs clear concepts, small examples, minimal setup, and code that can be
changed without reading a large framework.

### Team Explorer

Works inside a company team and wants to test whether agents can help with
internal tasks.

Needs a small base that can be adapted to local workflows, visible architecture,
simple extension points, and enough structure for team collaboration.

### Agent Contributor

Uses an AI coding agent to extend the project.

Needs a stable collaboration contract, clear task specs, and context files that
explain intent rather than only file locations.

## Design Principles

### Clarity First

The implementation should be understandable by reading the code directly. Prefer
explicit function calls and plain data structures.

### Hackable By Default

Users should be able to change prompts, tools, model adapters, and orchestration
logic without learning a large internal framework.

### Small Core, Visible Edges

The core agent loop should stay small. Integrations should sit at visible edges
so they can be replaced or removed.

### Learn Before Abstracting

Add abstractions only after repeated patterns are clear. Early code should teach
the shape of the system.

### Agent-Friendly Context

The repository should make it easy for coding agents to understand intent,
constraints, and next steps before editing files.

## Current Phase

The repo has a canonical small runtime under `src/simple_agent_lab/`, a
provider-agnostic LLM boundary with OpenAI and Anthropic adapters,
deterministic local examples, focused tests, and optional benchmark
integrations.

Near-term work should preserve the small teaching core while improving
reliability on long-horizon tasks, observability, and reproducible evaluation.
External providers and benchmark infrastructure should remain optional.
