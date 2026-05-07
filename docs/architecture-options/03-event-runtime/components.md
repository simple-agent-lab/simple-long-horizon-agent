# Components

Two files. `core.py` carries the runtime shape and helpers; `models.py`
carries the provider boundary.

## `core.py`

### Data types

| Type | Role |
| --- | --- |
| `Message` | Shared role-specific union from `simple_agent_lab.messages`. |
| `ModelMessage` | Shared provider-boundary union produced by `model_messages`. |
| `RuntimeEvent` (frozen) | `index + kind + data`. Trace element. |
| `RuntimeState` | `task + messages + events`. Each `record` appends both. |
| `EventObserver` | Synchronous callback that receives each future `RuntimeEvent`. |
| `Agent` (frozen) | `name + instruction`. No `act` function: behavior lives in `ModelClient`. |
| `RunConfig` (frozen) | `max_steps`, `last_messages`, `meta` mode. Validated in `__post_init__`. |
| `RunResult` (frozen) | `state + steps + stop_reason`. |
| `GraphNode` (frozen) | Named graph node: `Agent + ModelClient + tools + optional RunConfig`. |
| `GraphEdge` (frozen) | Ordered route: `source`, optional `target`, `label`, optional condition. |
| `AgentGraph` (frozen) | Entry node, node table, edge list, and `max_node_runs` guard. |
| `GraphRunResult` (frozen) | `state + node_runs + graph stop_reason + path`. |
| `RunReport` (frozen) | Event-derived metrics: model calls, tool calls, messages, output kinds, node stop reason, graph stop reason, graph path. |
| `ModelClient` | Abstract base. Subclasses override `generate`. |
| `AgentTool` / `ToolResult` | Shared tool values from `simple_agent_lab.tools`. |

### Functions and methods

#### `context_view(agent, state, config) -> list[Message]`

Filters `state.messages` for the agent (`target in {agent.name, "all"}` or
sender == agent.name), optional last-N truncation. Prepends a system
message carrying `agent.instruction` with the agent as `target`.

#### `model_messages(messages, meta="none") -> list[ModelMessage]`

Builds the shared provider-boundary payload. With `meta="header"`, prefixes
visible text content with `[sender -> target | kind/channel]`. Explicit
assistant `tool_calls` and tool-result identity fields are preserved through
the shared message projection. Validates the meta mode at runtime even though
the type is a `Literal`.

#### `as_agent_message(agent, output) -> Message`

Re-tags the model's reply: `sender = agent.name`, `target = "user"` if
`output.kind == "final"` else `agent.name`. Records the original sender
under `data["model_sender"]`.

#### `RuntimeState.emit(kind, **data) -> RuntimeEvent`

Appends a runtime event and synchronously notifies all current observers. Used
for `model_request`, `model_response`, tool events, and `stop`.

#### `RuntimeState.subscribe(observer) -> unsubscribe`

Registers a synchronous event observer. This is the integration point for UI,
logging, metrics, or persistence code that wants live events while the loop is
running. The returned function removes the observer.

#### `RuntimeState.record(message) -> Message`

Appends a message **and** emits a `kind="message"` event referencing it.
The two append in lockstep.

#### `RuntimeState.user(target, content, kind="task") -> Message`

Convenience for seeding a user task message.

#### `AgentLoop.run(agent, state, model, tools=None) -> RunResult`

The main loop. For each step up to `config.max_steps`:

1. Build `visible` via `context_view`.
2. Build `model_payload = model_messages(visible, meta=config.meta)`.
3. Emit `model_request { agent, visible_count, model_message_count, tool_count, model_payload }`.
4. Call `model.generate(visible, meta=config.meta, tools=...)`.
5. Emit `model_response { model_sender, output_kind, tool_call_count }`.
6. Convert the output to an agent-tagged message and `state.record(...)`.
7. Dispatch explicit assistant `tool_calls`, recording tool events and
   tool-result messages.
8. If `kind == "final"`, emit `stop { reason="final", steps=step }` and
   return `RunResult(state, step, "final")`.

If the loop runs out of steps, emit `stop { reason="max_steps", ... }`
and return `RunResult(state, max_steps, "max_steps")`.
If a tool returns `terminate=True`, emit
`stop { reason="tool_terminate", ... }`.

#### `AgentGraph.outgoing(source) -> tuple[GraphEdge, ...]`

Returns the ordered edges leaving a source node. `AgentGraph.__post_init__`
validates that the entry, node keys, edge sources, and edge targets are
coherent before a graph can run.

#### `GraphRunner.run(graph, state) -> GraphRunResult`

The graph scheduler. It emits `graph_start`, then repeatedly:

1. Emits `node_start`.
2. Delegates the node to `AgentLoop.run(...)`.
3. Emits `node_end`.
4. Chooses the first outgoing edge whose condition is absent or true.
5. Emits `edge_traversed` and records a graph handoff message targeted at the
   next node.

If no edge matches, it emits `graph_end` with `reason="done"`. If an edge has
`target=None`, it first emits `edge_traversed` for that terminal route, then
emits `graph_end`. If the graph reaches `max_node_runs`, it emits `graph_end`
with `reason="max_node_runs"`.

#### `messages_from_events(events) -> list[Message]`

Replays the transcript from `kind=="message"` events. This makes the
event-sourced claim concrete: consumers can derive the transcript from the
event log instead of trusting `RuntimeState.messages`.

#### `run_report(state) -> RunReport`

Computes common run metrics purely from events: node stop reason, step count,
model-call count, tool-call count, tool errors, message count, max visible
context size, output kinds, graph stop reason, graph node count, and graph
path.

#### `print_trace(state)`

Walks `state.events`. Renders model, tool, message, and stop events in a
compact trace. Useful for fixture-style inspection.

#### `print_report(report)`

Compact human-readable rendering for `RunReport`.

## `models.py`

### `LLMModelClient(ModelClient)`

Concrete model client backed by `simple_agent_lab.llm`.

- Projects shared runtime `Message` values through the shared LLM bridge.
- Passes shared tool definitions to the LLM layer.
- Converts `LLMResponse` back into an assistant `Message`, preserving
  thinking and tool calls as explicit fields.

### `fake_client(...) -> LLMModelClient`

Convenience factory for the deterministic fake provider. The main demo builds
the researcher client explicitly so readers can see the provider value, then
uses a tiny local advisor client to show that graph nodes can bind different
`ModelClient` implementations.

## What is intentionally absent

- No step hook stack. The version's extension surface is "subclass
  `ModelClient`" and "subclass `AgentLoop`", plus event observation through
  `RuntimeState.subscribe`.
- No injection queues. Mid-run intent injection is not modeled.
- No batched parallel execution. The shared tool values live in
  `simple_agent_lab.tools`; this runtime only owns sequential dispatch.
- No parallel graph fan-out or join. `GraphRunner` is a small ordered-edge
  scheduler, not a durable workflow engine.
- No `AbortSignal` or cancellation primitive.
- No streaming partial messages. `generate` returns one finalized
  message.

These are deliberate scope cuts; see the README's "Weaknesses" section
for the cost.
