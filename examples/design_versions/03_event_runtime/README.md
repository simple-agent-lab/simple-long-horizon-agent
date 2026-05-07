# 03 Event Runtime

This version is shaped for integrated graph orchestration, observability, and
provider-boundary experiments. It now serves as the reference for LangGraph- or
CrewAI-style nodes, edges, handoffs, event observers, replay, reports, and
provider-boundary ideas; new self-evolution runtime work starts in
`02_balanced_runtime`.

Files:

- `core.py`: `RuntimeState`, `RuntimeEvent`, `AgentLoop`, `AgentGraph`,
  `GraphRunner`, event observers, replay helpers, and `RunReport`.
- `models.py`: `LLMModelClient` backed by `simple_agent_lab.llm`.
- `demo.py`: a two-node graph with a conditional edge, runnable with the
  deterministic fake provider and a tiny local handoff model by default.

The loop is:

```text
state.subscribe(observer) optional
GraphRunner.run(graph, state)
  -> graph_start event
  -> node_start event
  -> AgentLoop.run(node.agent, state, node.model, node.tools)
       context_view -> model_request event -> ModelClient.generate
         -> model_response event -> message event
         [-> tool_execution_start/_end events + tool_result message, then loop again]
         -> stop event
  -> node_end event
  -> edge_traversed event + graph handoff message
  -> next node ...
  -> graph_end event -> run_report(state) / messages_from_events(state.events)
```

## Tools

Borrowed (and trimmed) from pi-mono. The data shapes come from
`simple_agent_lab.tools`; the parallel / streaming / abort variants are dropped
to fit 03's command-style loop:

1. **Two-layer split.** `Tool` is wire-format only (name + description +
   JSON-Schema parameters); `AgentTool` adds the local `execute`
   callable and a UI `label`.
2. **`ToolResult` carries `content` + `details`.** `content` is what the
   model sees on the next call; `details: Any` is a sidecar payload for
   UI / inspection that never reaches the model.
3. **Errors are tool results.** `dispatch_tool_calls` wraps `execute` in
   try/except; a thrown exception becomes a
   `ToolResult(is_error=True)` whose `content` is the error text. The
   model gets it back and can self-correct. Only `terminate=True` ends
   the run.
4. **`kind="tool_result"` is a first-class message.** Provider adapters
   translate to wire format (Anthropic vs OpenAI) at the boundary; the
   transcript stays provider-agnostic. `model_messages` skips the
   routing-header prefix for tool-result messages.

Usage:

```python
echo = AgentTool(
    name="echo",
    description="Echo input back",
    parameters={"type": "object", "properties": {"msg": {"type": "string"}}},
    execute=lambda call_id, args: ToolResult(
        content=[ToolContent(text=args["msg"])],
        details={"length": len(args["msg"])},
    ),
)
loop = AgentLoop(RunConfig(max_steps=4))
loop.run(agent, state, model, tools=[echo])
```

A `ModelClient` that wants tool use returns an assistant `Message` with
explicit `tool_calls` when the model decides to call a tool. The loop
dispatches automatically and continues for another turn so the model can react
to the result. Tool calls do not count as "final"; only `kind="final"` from
the model or `terminate=True` from a tool ends the run.

The integrated event helpers are the main difference from 02:

- `AgentGraph` / `GraphRunner` for named nodes, ordered edges, and handoffs.
- `RuntimeState.subscribe(...)` for live observers.
- `messages_from_events(...)` for replay.
- `run_report(...)` / `RunReport` for post-run metrics.

The `fake_client(...)` helper in `models.py` gives the shortest path to a
ready-to-use client while still exposing the full `LLMModelClient` for custom
providers or request options.

Run with no network:

```bash
PYTHONPATH=src python3 examples/design_versions/03_event_runtime/demo.py
```

The core still does not store provider response objects. Provider details are
converted back into `Message.data`.
