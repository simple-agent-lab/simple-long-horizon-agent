---
title: "Make Testing And Feedback First Priority"
status: Accepted
date: 2026-04-29
slug: make-testing-and-feedback-first-priority
---

# Make Testing And Feedback First Priority

## Status

Accepted

## Context

Harness engineering only works when agents can observe whether their changes
helped. The OpenAI harness engineering writeup describes the development loop
becoming stronger as testing, validation, review, feedback handling, and
recovery are encoded into the repository. It also frames agent struggle as a
signal that the repository is missing a tool, check, document, or feedback loop.

Simple Agent Lab is still in architecture exploration. The concrete core API is
not settled enough to justify a real test suite yet. Even so, the project should
decide now that testing and feedback are part of the architecture, not an
afterthought.

## Decision

Testing and feedback are first-priority design concerns.

During architecture exploration, this means every proposed architecture should
explain its feedback surface before it is selected:

- What behavior can be checked deterministically?
- What trace or state will an agent inspect after a run?
- What should be a unit test, smoke test, eval, or review checklist later?
- What failure would tell us the architecture is becoming too complex?

For future tasks, the expected order is:

```text
define feedback signal -> design the check shape -> change implementation
  -> run the available check -> update docs or decisions if the harness changed
```

## Consequences

New behavior should eventually arrive with a focused test, a deterministic
demo, an eval, or a run script. Before the core architecture is final, it is
enough to document the intended feedback signal.

Tests should stay readable enough for students to learn from them. Prefer
stdlib `unittest`, deterministic toy agents, and direct assertions over a large
testing framework until the project needs more.

Feedback signals are allowed to be lightweight. A smoke command is acceptable
when the goal is to prove that an example still runs. A unit test is preferred
when the goal is to protect a settled runtime invariant.

## Alternatives Considered

- Keep tests as a future concern. Rejected because this would make harness
  engineering aspirational rather than operational.
- Add a full test framework immediately. Rejected because the project should
  remain easy to understand without extra tooling.
