---
title: "Make Context View An Explicit Projection"
status: Accepted
date: 2026-05-11
slug: make-context-view-an-explicit-projection
---

# Make Context View An Explicit Projection

## Status

Accepted

## Context

ADR use-tiny-message-runtime named `context_view()` as the context-management boundary, and ADR
0009 promoted the balanced runtime into `src`. The first promoted
implementation still kept `context_view()` as a small route filter with an
optional `last` slice.

That was enough for early demos, but context experiments now need a clearer
surface: visibility policy, rough budget pressure, clipping, and traceable
request metadata. Production references such as Claude Code-style query loops
and OpenAI Codex both keep durable history separate from the model-visible
projection. They also preserve tool-call/tool-result invariants while trimming
or compacting context.

## Decision

Keep `State.events` and `State.messages` as the full inspectable history.

Add `src/simple_agent_lab/context_view.py` as the projection layer:

- `ContextPolicy` describes visibility and budget knobs.
- `build_context_view(...)` projects a message sequence for one agent.
- `ContextView` returns selected messages plus simple stats and notes.
- Large text messages can be clipped without changing message identity fields.
- Tool-call and tool-result messages are grouped so budget trimming does not
  split valid pairs.

Keep the old core API:

```text
context_view(agent, state) -> list[Message]
```

and expose the detailed API:

```text
build_agent_context_view(agent, state, policy=...) -> ContextView
```

`run()` now records the context-view summary in each `model_request` event.

## Consequences

Context experiments can happen without editing the agent loop or provider
adapters. The request trace can explain what an agent saw and what was dropped.

The budget remains an approximate character heuristic, not provider-accurate
token accounting. That keeps the runtime stdlib-only and beginner-readable.

Real summarization, retrieval, cache editing, session memory, and remote
compaction remain future layers. They should be added around this projection
boundary only when a runnable experiment proves the need.

## Alternatives Considered

- Keep `context_view()` as only route filtering. Rejected because context
  experiments need stats and budget behavior.
- Add a production-style `ContextManager` class. Rejected because it would make
  the first `src` runtime too heavy.
- Put budget logic in `model_messages(...)`. Rejected because visibility and
  model-payload conversion are separate boundaries.
