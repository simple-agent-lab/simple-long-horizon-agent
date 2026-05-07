# 03 Event Runtime

> The richest candidate. Event-sourced graph runtime with explicit graph,
> node, edge, model, tool, message, and stop events. It keeps `AgentLoop` as
> the single-agent execution unit, then adds `AgentGraph` / `GraphRunner` for
> LangGraph- or CrewAI-style orchestration.
>
> Implementation:
> - [`examples/design_versions/03_event_runtime/core.py`](../../../examples/design_versions/03_event_runtime/core.py) - `RuntimeState`, `RuntimeEvent`, `AgentLoop`, `AgentGraph`, `GraphRunner`, `RunResult`, `GraphRunResult`, `RunReport`.
> - [`examples/design_versions/03_event_runtime/models.py`](../../../examples/design_versions/03_event_runtime/models.py) - `LLMModelClient` backed by `simple_agent_lab.llm`.
> - [`examples/design_versions/03_event_runtime/demo.py`](../../../examples/design_versions/03_event_runtime/demo.py) - two-node graph with a conditional edge, using a fake LLM client plus a local handoff client.

## What it optimizes for

Integrated graph orchestration, observability, replay, reporting, and provider
portability. Everything the graph and loop do becomes a typed event; observers
can subscribe to those events; reports and transcript replay are derived from
the event log. The model boundary is a class with a `generate` method that the
loop never bypasses. After ADR 0005, this version is a reference for graph,
event, and provider ideas rather than the main place to grow self-evolution
recipes.

## Runtime Shape

```text
state.user(entry_agent, task)
  -> state.subscribe(observer)                 # optional
  -> GraphRunner.run(graph, state):
       emit "graph_start"
       current = graph.entry
       for node_run in 1..max_node_runs:
         emit "node_start"
         AgentLoop.run(node.agent, state, node.model, node.tools):
           for step in 1..max_steps:
             visible = context_view(agent, state, config)
             emit "model_request" { agent, visible_count, model_payload, tool_count }
             output = model.generate(visible, meta=config.meta, tools=tools)
             emit "model_response" { model_sender, output_kind, tool_call_count }
             state.record(as_agent_message(agent, output))       # also emits "message"
             if message.tool_calls: dispatch tools and continue
             if message.kind == "final": emit "stop"; return RunResult(...)
         emit "node_end"
         choose first matching GraphEdge
         if no edge: emit "graph_end"; return GraphRunResult(...)
         emit "edge_traversed"
         record graph handoff message for the next node
  -> run_report(state)                         # derived metrics
  -> messages_from_events(state.events)        # replay transcript
```

The loop is still a method on `AgentLoop`, not a generator. `GraphRunner` is a
higher-level scheduler that repeatedly delegates to `AgentLoop`. Events live
on `state.events`; consumers read them after the run (or during, since they
are appended in real time). Each `state.record(message)` both appends the
message **and** emits a `kind="message"` event — message and event are coupled
here, unlike pi-mono's "messages = transcript, events = ephemeral" split.

## Core Ideas

- **Every loop transition is a typed `RuntimeEvent`.**
  - `graph_start` / `graph_end` - graph execution started or finished.
  - `node_start` / `node_end` - one graph node delegated to `AgentLoop`.
  - `edge_traversed` - routing moved from one node to another.
  - `message` — a message was added to the transcript.
  - `model_request` — the loop is about to call the model.
  - `model_response` — the model returned (with `output_kind`).
  - `tool_execution_start` / `tool_execution_end` - a tool call ran.
  - `stop` - the loop terminated, with `reason` in
    `{"final", "max_steps", "tool_terminate"}`.
  This is the version that takes "event sourcing" literally.
- **Graph orchestration is explicit.**
  `AgentGraph` names an entry node, a node table, ordered edges, and a
  `max_node_runs` guard. Each `GraphNode` binds an `Agent`, `ModelClient`,
  optional tools, and optional per-node `RunConfig`. Each `GraphEdge` can have
  a label and condition. `GraphRunner` records a handoff message when it moves
  to the next node, so crew-like delegation is visible in the transcript.
- **Provider abstraction is a class.**
  `ModelClient.generate(messages, meta, tools) -> Message`. The shipped
  implementation is `LLMModelClient`, which delegates to the shared
  `simple_agent_lab.llm` access layer.
- **Observers are integrated into `RuntimeState`.**
  `state.subscribe(observer)` receives every future `RuntimeEvent`
  synchronously. This is the event-runtime version of a logging, metrics, UI,
  or persistence hook.
- **Reports and replay are event-derived.**
  `run_report(state)` summarizes model/tool/message counts and output kinds
  from events. `messages_from_events(events)` reconstructs the transcript from
  `kind=="message"` events.
- **`RunResult` is a structured return value.** Includes the final
  `RuntimeState`, step count, and a typed `StopReason` (`"final" |
  "max_steps"`). The caller never has to inspect events to learn how a
  run ended.
- **`Message.channel` is shared protocol state**, and `meta="header"` mode
  lets `model_messages` prefix each user-visible
  message with `[sender -> target | kind/channel]` so the model can read
  routing in plain text. This makes graph handoffs over a real provider
  produce sensible transcripts without any pre-processing.
- **`MetaMode` is validated at runtime** even though the type is a
  `Literal`. The version errs on the side of clear errors over trust.
- **Single agent per `AgentLoop.run`, graph-level multi-agent above it.**
  The lower loop stays inspectable while the graph layer owns node routing.

## Data Model

```text
Message
  role     "system" | "user" | "assistant" | "tool_result"
  content  str | tuple[TextBlock | ImageBlock, ...]
  sender   str
  target   str
  kind     str
  channel  str
  data     Mapping[str, Any]

RuntimeEvent
  index    int
  kind     "graph_start" | "graph_end" | "node_start" | "node_end"
           | "edge_traversed" | "message" | "model_request"
           | "model_response" | "stop" | "tool_execution_start"
           | "tool_execution_end"
  data     dict

RuntimeState
  task      str
  messages  list[Message]
  events    list[RuntimeEvent]
  observers list[EventObserver]

Agent
  name        str
  instruction str

RunConfig
  max_steps      int = 4
  last_messages  Optional[int]
  meta           "none" | "header"

RunResult
  state        RuntimeState
  steps        int
  stop_reason  "final" | "max_steps" | "tool_terminate"

GraphNode
  name    str
  agent   Agent
  model   ModelClient
  tools   tuple[AgentTool, ...]
  config  RunConfig | None

GraphEdge
  source     str
  target     str | None
  label      str
  condition  Callable[[RuntimeState], bool] | None

AgentGraph
  entry          str
  nodes          dict[str, GraphNode]
  edges          tuple[GraphEdge, ...]
  max_node_runs  int

GraphRunResult
  state        RuntimeState
  node_runs    int
  stop_reason  "done" | "max_node_runs"
  path         tuple[str, ...]

RunReport
  stop_reason       StopReason | None
  steps             int
  model_calls       int
  tool_calls        int
  tool_errors       int
  messages          int
  max_visible_count int
  output_kinds      tuple[str, ...]
  graph_stop_reason "done" | "max_node_runs" | None
  node_runs         int
  graph_path        tuple[str, ...]

ModelClient (abstract)
  generate(messages, meta, tools) -> Message
```

## Tool Model

Built in, but intentionally narrow. The version supports sequential,
synchronous tool dispatch. Tool data comes from `simple_agent_lab.tools`:

- `Tool` is the provider-facing definition.
- `AgentTool` adds the local `execute` callable.
- Assistant messages carry explicit `tool_calls`.
- The loop records `tool_execution_start`, `tool_execution_end`, and
  `kind="tool_result"` messages.

This keeps the provider/event boundary honest without adopting 02's larger
parallel/sequential dispatch semantics.

## Provider Boundary

`ModelClient.generate(messages, meta, tools)` is the only entry the loop uses.
`LLMModelClient` is the shipped implementation. It projects shared
`Message` values through the shared LLM bridge, delegates to
`simple_agent_lab.llm.complete`, and converts the drained response back into an
assistant `Message`.

## State and Memory

State lives on `RuntimeState`:

- `messages` is the durable transcript.
- `events` is the durable trace.
- `observers` is the live event integration point.
- Each message append also appends a `kind="message"` event.

There is no `data` scratch dict (unlike 02). External memory belongs in
the consumer's data structures, not on `RuntimeState`. Replay is explicit:
`messages_from_events(state.events)` reconstructs the transcript by walking
events and projecting `kind=="message"` ones.

## Trace and Evaluation

`state.events` records:

- `graph_start`, `graph_end`, `node_start`, `node_end`, and
  `edge_traversed` for graph execution.
- `model_request` with `agent`, `visible_count`, `model_message_count`,
  `tool_count`, and the projected `model_payload`.
- `model_response` with `model_sender`, `output_kind`, and
  `tool_call_count`.
- `tool_execution_start` and `tool_execution_end` when a tool is called.
- `message` with the `Message` reference.
- `stop` with `reason` and `steps`.

Useful metrics (computable purely from events):

- Number of model calls per run.
- Number of tool calls and tool errors.
- Distribution of `output_kind` values.
- Correlation between `visible_count` and final answer kind.
- Pass-rate of `stop_reason == "final"` vs `"max_steps"` over a corpus.
- Graph paths taken by each task (`weather_researcher -> travel_advisor`,
  conditional branches, or max-node guard hits).

`run_report(state)` computes the common subset directly.

## Strengths

- The richest trace among the three versions; eval and observability
  code can be written without instrumenting the runtime.
- Live observers make it suitable for UI/logging/persistence integration
  without changing the loop body.
- `RunReport` and replay helpers make post-run analysis a first-class path,
  not demo-specific code.
- Provider abstraction is concrete: the loop depends on one `ModelClient`
  interface and the implementation delegates to the shared LLM access layer.
- Structured stop reasons remove guesswork from "did this run finish".
- Graph nodes, edges, labels, and conditional routing make this feel closer
  to LangGraph/CrewAI while keeping the lower loop readable.
- `Message.channel` and `meta="header"` work together to make runtime
  routing legible to the model in graph mode.

## Weaknesses

- **Graph scheduling is still simple.** One node runs at a time; there is no
  parallel fan-out, join node, retry policy, checkpoint resume, or durable
  scheduler.
- **No injection queues.** Pi-mono-style steering and follow-up are not
  built in; reproducing them would require subclassing `AgentLoop` or
  wrapping it.
- **Tools are intentionally simple.** No batched parallel execution, streaming
  updates, cooperative cancellation, or `before/after` hook.
- **`messages` and `events` duplicate the assistant transcript** (each
  message append emits an event). Consumers should pick one source of
  truth per query.
- **No live-provider CLI in this demo.** The example intentionally uses the
  deterministic fake provider plus a tiny local model client so the smoke path
  works without credentials.

## When to Pick This Version

Pick this version when the team's primary use case involves:

- Studying how a runtime should expose one pluggable provider boundary.
- Eval pipelines that read structured runtime events instead of regexing
  message text.
- Replay and debugging traces where the consumer needs to know not just
  the messages but the model-call boundaries around them.
- Integrations that need live event observers plus post-run graph reports.
- Provider-boundary experiments where sequential tool calling is enough.
- Multi-agent teaching demos where nodes, edges, and handoffs should be
  first-class concepts.

If the team's near-term goal is self-evolution harness work, version 02 is a
better starting point. If the goal is teaching the smallest possible loop,
version 01 wins.

## See Also

- [components.md](components.md): module-by-module breakdown.
- [example-experiment.md](example-experiment.md): a provider-boundary exercise
  and what to read out of `state.events`.
- [`docs/reference-architectures/pi-mono-agent-runtime.md`](../../reference-architectures/pi-mono-agent-runtime.md):
  the runtime shape this version would grow into if tool calling and
  cooperative cancellation become required.
