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
- [ADR 0011: Keep Benchmark Suites As Eval Adapters](0011-keep-benchmark-suites-as-eval-adapters.md) — superseded for the containerized case by ADR 0017
- [ADR 0012: Unify the Message Protocol on Content Blocks](0012-unify-message-protocol-on-content-blocks.md) — tool-result fragment superseded by ADR 0014
- [ADR 0013: Provider-Namespaced `extra` Channel and Two-Layer Trace](0013-extra-channel-and-two-layer-trace.md)
- [ADR 0014: Tool Results Are Content Blocks, Not a Separate Message Role](0014-tool-result-as-content-block.md)
- [ADR 0015: Three-Layer Trace Architecture — Event, Span, Training](0015-three-layer-trace-event-span-training.md)
- [ADR 0016: Eval Output Directory Convention](0016-eval-output-directory-convention.md)
- [ADR 0017: Context-Size Accounting — Provider Usage, Confined Estimation, a Safety Buffer](0017-context-size-accounting.md)
- [ADR 0018: MCP Servers Are a Tool Source, Wrapped at the Tool Boundary](0018-mcp-as-tool-source.md)
- [ADR 0020: Collapse the Scorer Seam into the Run Primitive](0020-collapse-scorer-seam-into-run-primitive.md) — amends ADR 0019 and ADR 0017, builds on ADR 0018
- [ADR 0021: Add Agent Skills (read-based, on by default)](0021-add-agent-skills.md)
- [ADR 0022: Consolidate Agent Presets behind one AgentSession + Toolsets](0022-consolidate-agent-presets.md) — builds on ADR 0018 and ADR 0021, amended by ADR 0023
- [ADR 0023: A Pluggable State Initializer Makes Skills a Bare-Agent Capability](0023-pluggable-state-init-hook.md) — amends ADR 0021 and ADR 0022

## Proposed

- [ADR 0017: Generic Containerized Eval Framework](0017-generic-containerized-eval-framework.md) — supersedes ADR 0011 for containerized suites
- [ADR 0018: Oracle Run Mode for Suite Self-Check](0018-oracle-run-mode-for-suite-self-check.md) — builds on ADR 0017
- [ADR 0019: Scorer Seam and Per-Suite Scoring Topology](0019-scorer-seam-and-scoring-topology.md) — amends ADR 0017, builds on ADR 0018, amended by ADR 0020

## Withdrawn

- [ADR 0007: Make 03 Event Runtime the One-Stop Observable Baseline](0007-make-03-event-runtime-one-stop-observable.md) — superseded by ADR 0009, which retired the `03_event_runtime` design version this ADR was scoped to.
