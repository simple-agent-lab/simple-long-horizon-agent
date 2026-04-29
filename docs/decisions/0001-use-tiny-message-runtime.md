# ADR 0001: Use a Tiny Message Runtime

## Status

Accepted

## Context

Simple Agent Lab is aimed at students and small research groups exploring
multi-agent systems. Earlier architecture options introduced separate concepts
such as artifact stores, trace stores, lifecycle hooks, evaluators, and context
managers.

Those concepts are useful for studying mature agent systems, but they make the
first runnable core harder to read. The project needs a smaller default shape
that still supports communication patterns, pipelines, shared state, and context
experiments.

## Decision

Use this core model:

```text
Agent + Message + State + context_view() + run()
```

- `Agent` is a named role with one `act` function.
- `Message` is the communication unit and the provider-neutral model message.
  It contains model-facing `role` and `content`, plus lab-facing `sender`,
  `target`, `kind`, `channel`, and structured `data`.
- `State` is the shared world. It contains the task, all events, and small
  experiment data.
- `context_view()` is the context management boundary. It decides which
  messages an agent can see for one step.
- `run()` is the schedule loop. It calls agents in order.

`Event` is still useful, but only as trace structure: it wraps one `Message`
with a step number inside `State.events`.

The core exposes `model_messages(...)` as the bridge from lab messages to a
common chat-model payload. Provider adapters can transform that payload further
for OpenAI-compatible, Anthropic-style, or local model APIs.

The direct constructor keeps model fields first:

```python
Message("user", "hello")
```

Do not add separate `Artifact`, `RunTraceStore`, `EvaluationObserver`, or
`Runtime` abstractions to the core yet. Artifacts can live in messages or
`State.data`. The run trace is `State.events`. Evaluation code can read the
same state after a run.

## Consequences

The first implementation is small enough to teach by inspection. Debate,
pipeline, parallel, workspace, and society-style systems can be expressed as
recipes over the same message runtime.

Context management remains explicit without becoming a framework. To experiment
with memory, compaction, retrieval, or visibility, change `context_view()` or
pass a different view function into `run()`.

The tradeoff is that advanced concerns are not separately modeled yet. If the
project later needs durable storage, rich artifacts, or eval dashboards, those
can be added around `State` and `Message` without changing the beginner-facing
core.

## Alternatives Considered

- Message-centric runtime with mailbox, trace store, and evaluator.
- Pipeline-centric runtime with artifacts as first-class handoff objects.
- Workspace-centric runtime with board, artifact store, and manager-worker
  loops.
- Separate context manager class.

These remain useful reference designs, but they are intentionally not the
current core.
