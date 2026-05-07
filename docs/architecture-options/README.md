# Architecture Options

This directory holds the architecture documentation for the three
**focused runtime candidates** living under
[`examples/design_versions/`](../../examples/design_versions/README.md).
Each candidate has its own folder here with a `README.md` (architecture
overview), `components.md` (module-by-module breakdown), and
`example-experiment.md` (a concrete experiment the version is well
suited for).

The team will eventually prune this to one canonical implementation.
[ADR 0005](../decisions/0005-make-balanced-runtime-the-lead-core-candidate.md)
makes `02_balanced_runtime` the lead core candidate for self-evolution work,
and [ADR 0009](../decisions/0009-promote-balanced-runtime-to-src-core.md)
promotes the simplified version into `src/simple_agent_lab/core.py`. `01`
stays the teaching baseline and `03` stays the graph orchestration /
observability / provider-boundary reference.

The committed *direction* (a small message-first runtime, not a heavy
framework) is recorded in
[ADR 0001](../decisions/0001-use-tiny-message-runtime.md). What remains
open is the runtime *shape*.

## Options

| Option | Folder | Implementation | One-line shape |
| --- | --- | --- | --- |
| [01 Functional Loop](01-functional-loop/README.md) | `01-functional-loop/` | [`examples/design_versions/01_functional_loop/`](../../examples/design_versions/01_functional_loop) | Single function, single agent, message list is the trace |
| [02 Balanced Runtime](02-balanced-runtime/README.md) | `02-balanced-runtime/` | [`src/simple_agent_lab/core.py`](../../src/simple_agent_lab/core.py) plus [`examples/design_versions/02_balanced_runtime/`](../../examples/design_versions/02_balanced_runtime) | Canonical core: generator-based multi-agent loop, request/response trace events, tools, agent-as-tool delegation |
| [03 Event Runtime](03-event-runtime/README.md) | `03-event-runtime/` | [`examples/design_versions/03_event_runtime/`](../../examples/design_versions/03_event_runtime) | Graph nodes, edges, handoffs, event log, observers, replay, reports, ModelClient provider boundary |

## How to Compare

Each option's `README.md` follows the same structure:

- **What it optimizes for** — the one-line problem statement.
- **Runtime Shape** — pseudocode of the loop.
- **Core Ideas** — the design decisions that distinguish this version.
- **Data Model** — the typed surface a consumer sees.
- **Strengths / Weaknesses** — honest cost-benefit summary.
- **When to Pick This Version** — selection guidance.

Selection axes the team should weigh, in roughly this order:

1. Reading cost for new contributors.
2. Smallest core that supports the recipes the team actually uses
   (debate, pipeline, voting, manager-worker, tool-using single agent).
3. Cost of plugging in a real LLM provider.
4. Cost of adding tools (parallel and sequential, with cancellation).
5. Quality of the trace for debugging and evals.
6. Friction of extension: hooks, custom message kinds, scheduling rules.

## Decision Path

1. Read the three READMEs side by side.
2. Pick a small concrete experiment your team actually wants to run.
   Use the `example-experiment.md` in each folder as a model.
3. Run it on at least two of the three versions. Note where the
   runtime "fits" and where you fight it.
4. Use `02_balanced_runtime` first unless the experiment is purely
   educational (`01`) or explicitly graph/provider/eval-boundary focused
   (`03`).
5. Capture the result and recommendation as an ADR update when the team is
   ready to prune.
6. Prune the two folders not chosen, both here and under
   `examples/design_versions/`.

## Cross-References

- [`examples/design_versions/README.md`](../../examples/design_versions/README.md):
  runnable side-by-side commands for all three versions.
- [`docs/reference-architectures/pi-mono-agent-runtime.md`](../reference-architectures/pi-mono-agent-runtime.md):
  the inspiration for version 02's generator runtime shape and 03's small
  provider/tool boundary.
- [`docs/reference-architectures/`](../reference-architectures/README.md):
  notes on Claude Code, Hermes, opencode, and pi-mono runtimes for
  comparison.
- [ADR 0001](../decisions/0001-use-tiny-message-runtime.md): the
  decision to commit to a small message-first runtime in general.
- [ADR 0004](../decisions/0004-treat-self-evolution-as-harness-capability.md):
  the self-evolution harness loop this comparison now serves.
- [ADR 0005](../decisions/0005-make-balanced-runtime-the-lead-core-candidate.md):
  the decision to focus new self-evolution runtime work on version 02.
- [ADR 0009](../decisions/0009-promote-balanced-runtime-to-src-core.md):
  the decision to promote the simplified balanced runtime into `src`.
