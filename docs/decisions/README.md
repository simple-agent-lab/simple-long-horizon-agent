# Decisions

This directory contains architecture decision records.

Create a decision record when the project commits to a meaningful technical or product direction, especially if future contributors may wonder why it was chosen.

Use [0000-template.md](0000-template.md) for new records. Number accepted decisions sequentially.

## Accepted

- [ADR 0001: Use a Tiny Message Runtime](0001-use-tiny-message-runtime.md)
- [ADR 0002: Adopt Harness Engineering Workflow](0002-adopt-harness-engineering-workflow.md)
- [ADR 0003: Make Testing And Feedback First Priority](0003-make-testing-and-feedback-first-priority.md)
- [ADR 0004: Treat Self-Evolution As Harness Capability](0004-treat-self-evolution-as-harness-capability.md)
- [ADR 0005: Make Balanced Runtime The Lead Core Candidate](0005-make-balanced-runtime-the-lead-core-candidate.md)
- [ADR 0006: Use A Role-Specific Message Protocol](0006-use-role-specific-message-protocol.md) — parts superseded by ADR 0012 and ADR 0014
- [ADR 0008: Separate Trajectory, Evaluation, and Training Data Records](0008-collect-training-trajectories-across-design-versions.md)
- [ADR 0009: Promote Balanced Runtime To Src Core](0009-promote-balanced-runtime-to-src-core.md)
- [ADR 0010: Make Context View An Explicit Projection](0010-make-context-view-an-explicit-projection.md)
- [ADR 0011: Keep Benchmark Suites As Eval Adapters](0011-keep-benchmark-suites-as-eval-adapters.md)
- [ADR 0012: Unify the Message Protocol on Content Blocks](0012-unify-message-protocol-on-content-blocks.md) — tool-result fragment superseded by ADR 0014
- [ADR 0013: Provider-Namespaced `extra` Channel and Two-Layer Trace](0013-extra-channel-and-two-layer-trace.md)
- [ADR 0014: Tool Results Are Content Blocks, Not a Separate Message Role](0014-tool-result-as-content-block.md)

## Withdrawn

- [ADR 0007: Make 03 Event Runtime the One-Stop Observable Baseline](0007-make-03-event-runtime-one-stop-observable.md) — superseded by ADR 0009, which retired the `03_event_runtime` design version this ADR was scoped to.
