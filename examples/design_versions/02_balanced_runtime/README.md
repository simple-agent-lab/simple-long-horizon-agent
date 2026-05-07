# 02 Balanced Runtime

Multi-agent runtime with the architecture ideas borrowed from
[pi-mono](https://github.com/badlogic/pi-mono)'s `pi-agent-core` package.
ADR 0005 makes this the lead core candidate for self-evolution work.

Files:

- `core.py`: compatibility re-export for
  `simple_agent_lab.core`, where the balanced runtime now lives.
- `agent.py`: compatibility re-export for
  `simple_agent_lab.core.AgentRuntime`.
- `demo.py`: agent-as-tool delegation. A coordinator calls `run_agent`, whose
  tool execution runs a focused child agent and returns its result.
- `evolution_probe.py`: a tiny baseline-vs-candidate comparison over a context
  transform, with `candidate_id` visible in request/response events.

## What is borrowed from pi-mono

| Idea | Where in this version |
|---|---|
| Agent step as one visible transition | `Agent.step(agent, visible, state) -> Message`; `make_llm_step` maps one step to one model call |
| Context experiment hook | optional `transform: TransformFn` before each agent step |
| Runtime is a generator that yields events | `run()` returns `Iterator[Event]` |
| Agent as tool | `AgentTool(name="run_agent", execute=...)` can run a child `Agent` inside tool execution |
| Tool definition / execution split | shared `simple_agent_lab.tools.Tool` for wire format, `AgentTool` for local execution |
| Two API layers (functional + stateful) | `core.run()` for functional, `AgentRuntime` for stateful |
| `prompt()` vs `resume()` | `AgentRuntime.prompt()` and `AgentRuntime.resume()` |
| Static schedule list replaced with a function | `next_agent: Callable[[State], str | None]` |

## Self-Evolution Fit

Each agent turn records `model_request` and `model_response` events. The
request event includes the visible-message outline, the LLM payload projection,
and optional `state.data["candidate_id"]`. That gives eval code a small,
inspectable surface for comparing prompt, context, tool-description, and recipe
candidates.

Events are one simple dataclass: `Event(index, kind, data)`. Consumer code
should switch on `event.kind`, read structured fields from `event.data`, and
use `event.message` for `kind == "message"`.

For code that owns the run, iterate the returned event stream:
`for event in runtime.prompt(...): ...`. Use `runtime.subscribe(...)` for
side observers such as logging, metrics, UI mirrors, or persistence that
should see events without controlling the run loop.

The promoted `src` version deliberately drops the earlier steering and
follow-up queues. Extra user input should be recorded explicitly on `State`
and then driven with `resume()` if another run is needed.

## What was deliberately left out

These belong in `03_event_runtime`:

- Streaming partial assistant messages
- Provider abstraction (`streamFn`, `getApiKey`, thinking budgets)
- A heavier event-sourced state model

## Run

```bash
PYTHONPATH=src python3 examples/design_versions/02_balanced_runtime/demo.py
PYTHONPATH=src python3 examples/design_versions/02_balanced_runtime/evolution_probe.py
```
