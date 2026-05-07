# Example Experiment: Graph Trace Metrics

This experiment uses the deterministic fake provider and reads metrics directly
out of `state.events` instead of parsing message text.

## Goal

Validate that the event stream contains everything an evaluator needs:
graph path, node-run count, model-call count, projected payload size, output
kind distribution, tool-call count, and stop reason. If the answer is yes,
downstream eval code does not need to instrument the runtime further.

## Setup

```python
from core import Agent, AgentGraph, GraphEdge, GraphNode, GraphRunResult, GraphRunner, RunConfig, RunReport, RuntimeState, run_report
from models import fake_client
from demo import weather_tool

researcher = Agent("weather_researcher", "Use the weather tool before answering.")
advisor = Agent("travel_advisor", "Turn handoffs into concise travel advice.")
config = RunConfig(max_steps=4, meta="header")

def has_weather_result(state: RuntimeState) -> bool:
    return any(
        message.kind == "tool_result"
        and getattr(message, "tool_name", "") == "get_weather"
        and not getattr(message, "is_error", False)
        for message in state.messages
    )

def run_one() -> GraphRunResult:
    state = RuntimeState("Tell me what to wear in Tokyo today.")
    state.user(researcher.name, state.task)
    graph = AgentGraph(
        entry=researcher.name,
        nodes={
            researcher.name: GraphNode(researcher.name, researcher, fake_client(), (weather_tool,)),
            advisor.name: GraphNode(advisor.name, advisor, fake_client()),
        },
        edges=(
            GraphEdge(
                source=researcher.name,
                target=advisor.name,
                label="weather found",
                condition=has_weather_result,
            ),
            GraphEdge(source=researcher.name, target=None, label="no weather result"),
        ),
    )
    return GraphRunner(config).run(graph, state)
```

## Variants

Use different tasks or tool definitions while keeping the same `AgentLoop`.

## Metrics Pulled From `state.events`

The common metrics are already integrated:

```python
def metrics(state: RuntimeState) -> RunReport:
    return run_report(state)
```

The equivalent manual query is:

```python
def metrics(state: RuntimeState) -> dict:
    requests = [e for e in state.events if e.kind == "model_request"]
    responses = [e for e in state.events if e.kind == "model_response"]
    graph_end = next(e for e in state.events if e.kind == "graph_end")
    stop = next(e for e in state.events if e.kind == "stop")
    return {
        "graph_stop_reason": graph_end.data["reason"],
        "graph_path": graph_end.data["path"],
        "node_runs": graph_end.data["node_runs"],
        "model_calls": len(requests),
        "first_visible_count": requests[0].data["visible_count"] if requests else 0,
        "first_model_message_count": requests[0].data["model_message_count"] if requests else 0,
        "tool_calls": sum(e.data["tool_call_count"] for e in responses),
        "output_kinds": [e.data["output_kind"] for e in responses],
        "stop_reason": stop.data["reason"],
        "steps": stop.data["steps"],
    }
```

Comparison points across tasks:

- Did the graph path match the expected node route?
- Did the graph stop before the `max_node_runs` guard?
- Did `stop_reason` stay `"final"`?
- Did the tool-call count match the expected trajectory?
- Did `first_model_message_count` change when the context policy changed?
- Did `output_kinds` show thought -> final?

## Why This Version Suffices

- Graph and node events are explicit; you do not have to wrap the scheduler
  to know which route executed.
- `model_request` and `model_response` events are explicit; you do not have
  to wrap the provider call to count it.
- `run_report(state)` gives the common post-run and graph summary directly.
- `RunResult.stop_reason` is typed; no string parsing.
- The fake provider uses the same `ModelClient` and LLM bridge shape as a real
  provider adapter would use.

## When This Version Is Not Enough

If the experiment grows to need any of the following, version 03's
shape starts to bend:

- Batched parallel tool calls or streaming tool updates. Version 03 keeps tool
  dispatch sequential and synchronous.
- Parallel graph fan-out, join nodes, checkpoint resume, or retry policies.
  Version 03 is graph-shaped, but still deliberately tiny.
- Mid-run user steering. There is no queue; you would either subclass
  `AgentLoop` or fall back to version 02.
- Cooperative cancellation tied to a UI cancel button. There is no
  abort signal; you would have to thread one through.

At that point, the team should either commit to extending 03 toward
pi-mono shape (see the reference architecture note) or reconsider
whether 02 was a better starting point for the use case.
