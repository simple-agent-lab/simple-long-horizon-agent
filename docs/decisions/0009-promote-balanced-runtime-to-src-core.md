# ADR 0009: Promote Balanced Runtime To Src Core

## Status

Accepted

## Context

ADR 0005 selected `02_balanced_runtime` as the lead core candidate for
self-evolution work. The demo has since proved the shape the project wants:
one generator-based runtime that records request/response events, supports
tool dispatch, and can model agent-as-tool delegation without a heavier graph
runtime.

Keeping the implementation only under `examples/design_versions/` now creates
the wrong maintenance pressure. The examples should demonstrate the runtime;
they should not be the canonical source of the runtime.

The 02 sketch also carried pi-mono-inspired steering and follow-up queues.
Those are useful for interactive shells, but they add a second input path before
the project needs it. The simpler contract is: record explicit messages on
`State`, then call `resume()` if another run is needed.

## Decision

Promote the simplified balanced runtime into `src/simple_agent_lab/core.py`.

The canonical runtime includes:

- `Agent.generate(visible) -> Message` (name/role/tools are closed over by
  the factory that builds the function, e.g. `make_llm_agent`)
- `State.events` as simple `Event(index, kind, data)` records
- A single-agent `run(agent, state, max_turns=...)` loop that stops on
  the first `final` message from the agent and reports truncation via
  `agent_end(reason="max_turns")`
- Multi-agent flows expressed through `tools.task_tool` (the parent agent
  delegates by tool-call; each sub-agent runs its own `run()` inside the
  tool's `execute`)
- `context_view` with `ContextPolicy` for context-budgeting experiments
  (callers wanting to inject extra messages do so by recording them onto
  `state` before driving the event iterator; see `task_tool`)
- `model_request` and `model_response` trace events
- `AgentTool` dispatch and tool-result messages

Do not include steering or follow-up queues in the canonical `src` runtime.
The 02 example folder now re-exports the `src` runtime so the demo remains
runnable without maintaining a second copy.

## Consequences

Future implementation work should target `src/simple_agent_lab/core.py`, not a
local design-version runtime copy.

The beginner-facing recipe scripts now use the same `Agent.generate(visible) -> Message`
contract as the agent-as-tool demo. This raises the core slightly above the
original tiny schedule loop, but it removes a larger source of confusion:
parallel runtimes with overlapping concepts.

The loss of built-in steering means a future interactive CLI must choose its
own input policy. That is acceptable until an interactive product path proves
the exact queue semantics belong in the shared core.

## Alternatives Considered

- Keep 02 only as an example. Rejected because ADR 0005 made it the lead path
  and new work would keep duplicating changes.
- Promote 02 exactly as-is. Rejected because the steering and follow-up queues
  are not needed for the current self-evolution and agent-as-tool demos.
- Promote 03 instead. Rejected for the same reason as ADR 0005: it remains a
  useful observability and provider-boundary reference, but it is heavier than
  the current teaching and self-evolution path needs.
