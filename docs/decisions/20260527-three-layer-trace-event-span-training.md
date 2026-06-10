---
title: "Three-Layer Trace Architecture — Event, Span, Training"
status: Accepted
date: 2026-05-27
slug: three-layer-trace-event-span-training
---

# Three-Layer Trace Architecture — Event, Span, Training

## Status

Accepted

## Context

ADR collect-training-trajectories-across-design-versions established the separation of trajectory, evaluation, and training
records. The trajectory layer evolved into a three-level structure:
`ModelCall` (per-API-call), `ContextWindow` (per-compression-span), and
`RunTrace` (the full run). While functional, that design has become hard
to explain, extend, and maintain:

- `ModelCall` is extracted from paired events (`ModelRequestEvent` →
  `ModelResponseEvent` → assistant `MessageEvent`) via a 60-line state
  machine that reimplements what event pairing already implies.
- `ContextWindow` is an indirect grouping over `ModelCall` indices; it
  doesn't carry its own input/output and requires cross-referencing to
  understand.
- `ModelRequestEvent` carries the full LLM payload (`llm_payload`,
  `visible`, `context_view`, `tools`), making it both a lightweight
  timestamp marker and a heavy data container — two roles that conflict.
- Adding new observable operations (parallel tool calls, sub-agent
  invocations, memory retrieval) requires inventing new extraction logic
  each time rather than fitting into a uniform structure.

Industry practice in 2026 has converged on span-based observability for
agent systems: OpenTelemetry GenAI semantic conventions, Claude Code's
OTEL integration (`claude_code.interaction` → `claude_code.llm_request` →
`claude_code.tool`), OpenAI Codex's `rollout_trace` with propagated span
context across async boundaries, and Google ADK's event-sourced session
model. The common pattern: runtime emits lightweight events; observability
derives structured spans from those events; training derives
provider-formatted records from spans or messages.

## Decision

Adopt a three-layer trace architecture with clear names and clean
derivation:

```text
Layer 1: Event     — "what happened?"         (runtime emits, append-only)
Layer 2: Span      — "what operations ran?"    (derived from events, structured)
Layer 3: Training  — "what can be learned?"    (derived from messages/spans, provider-formatted)
```

Each layer is a pure function of the one below it. No layer stores
redundant copies of another's data.

### Layer 1: Event (unchanged)

Runtime events in `protocols.py` remain the append-only source of truth.
They are lightweight timestamped markers emitted by `core.py` via `yield`.
The existing event types (`AgentStartEvent`, `TurnStartEvent`,
`ModelRequestEvent`, `ToolExecutionStartEvent`, etc.) already form
natural start/end pairs. No changes to `protocols.py` or `core.py`.

### Layer 2: Span (new)

A `Span` is a single operation the agent performed, derived from event
pairs:

```python
@dataclass(frozen=True)
class Span:
    id: str
    parent_id: str | None
    kind: str               # "agent_run" | "turn" | "model_call" | "tool_call" | "compression"
    start: float            # elapsed seconds from run start
    end: float              # elapsed seconds from run start
    input: Any | None       # what the operation received
    output: Any | None      # what the operation produced
    attributes: dict[str, Any] | None  # lightweight metadata
```

Event pairs map to span kinds:

| Event pair                              | Span kind       |
|-----------------------------------------|-----------------|
| `AgentStartEvent` → `AgentEndEvent`     | `agent_run`     |
| `TurnStartEvent` → `TurnEndEvent`      | `turn`          |
| `ModelRequestEvent` → `ModelResponseEvent` | `model_call` |
| `ToolExecutionStartEvent` → `ToolExecutionEndEvent` | `tool_call` |
| `ContextCompressionEvent`               | `compression`   |

Parent-child relationships are inferred from event nesting order using a
stack, not stored on the events themselves.

`spans_from_events(trace_id, events) -> list[Span]` is the single
extraction function, replacing both `model_calls_from_events` and
`context_windows_from_events`.

### Layer 3: Training (unchanged in role)

`openai_training_record` in `trace.py` continues to produce
provider-formatted fine-tuning records from `State.messages`. It does not
depend on Layer 2 spans. Future training exporters may optionally use
spans for richer signal (e.g. filtering by span kind, attaching span-level
rewards).

### RunTrace

```python
@dataclass(frozen=True)
class RunTrace:
    trace_id: str
    producer: str
    task: str
    events: list[Any]
    messages: list[Any]
    meta: dict[str, Any] | None = None

    def spans(self) -> list[Span]:
        return spans_from_events(self.trace_id, self.events)
```

`model_calls` and `context_windows` fields are removed. Spans are
computed on demand from the event log. `ContextWindow` as a concept is
retired — it was a grouping of model calls between compressions, which
consumers can derive from the span tree by filtering for `compression`
spans and partitioning `model_call` spans between them.

## Consequences

The three layers have a clean 1:1 mapping to questions, consumers, and
files:

| Layer    | Question             | Consumer                | File            |
|----------|----------------------|-------------------------|-----------------|
| Event    | What happened?       | Runtime, replay         | `protocols.py`  |
| Span     | What operations ran? | Developers, eval, debug | `trajectory.py` |
| Training | What can be learned? | Training pipelines      | `trace.py`      |

> Update (2026-06): the derivation layers were consolidated into one package,
> `src/simple_agent_lab/trace/` — Span in `spans.py`, Training in
> `training.py` (provider-neutral) and `openai_export.py` (OpenAI Chat
> format), with `print_trace` in `render.py`. The former top-level
> `trace.py` module and `trajectory/` package names are gone; the three
> layers themselves are unchanged.

Adding a new observable operation (e.g. sub-agent delegation, memory
retrieval) requires: (a) adding a start/end event pair in `protocols.py`,
and (b) adding one branch in `spans_from_events`. No new extraction
function, no new top-level type.

Parallel tool calls work without changes: `ToolExecutionStartEvent`
already carries `tool_call_id` for correlation. Multiple concurrent
tool spans share the same parent turn span.

The `Span` type is structurally compatible with OpenTelemetry: `id` →
`span_id`, `parent_id` → `parent_span_id`, `input`/`output` →
`gen_ai.content.prompt`/`gen_ai.content.completion`, `attributes` →
`attributes`. A future OTel exporter is a straightforward field mapping.

`ModelCall` and `ContextWindow` are removed from the public API.
Backward compatibility aliases may be provided temporarily if external
consumers depend on them.

## Alternatives Considered

- **Keep ModelCall/ContextWindow alongside Span.** Rejected: three
  overlapping representations of the same data creates confusion about
  which one to use and which is the source of truth.
- **Store spans eagerly on RunTrace.** Rejected: spans are a pure
  derivation from events. Storing them duplicates data and creates a
  consistency risk. A `spans()` method computes them on demand.
- **Session Chain model** (sessions as first-class containers with
  handoff records between context windows). Explored but deferred:
  context compression is a spectrum (from trimming tool outputs to full
  summarization), and drawing a hard session boundary is an arbitrary
  policy decision. Compression is recorded as a `compression` span;
  consumers that need session-like grouping can split on compression
  spans above a chosen threshold.
- **Flat event list with no Span.** Works for sequential execution but
  breaks down with parallel tool calls and sub-agent nesting, which
  require correlation IDs and parent-child relationships — i.e. spans.
- **Full OTel SDK integration.** Rejected for now: adds a dependency and
  conceptual weight that conflicts with the project's teaching mission.
  The Span type is OTel-compatible by structure so a future bridge is
  cheap.
