# 02 Balanced Runtime

Multi-agent runtime with the architecture ideas borrowed from
[pi-mono](https://github.com/badlogic/pi-mono)'s `pi-agent-core` package.

Files:

- `core.py`: pure-functional layer. `Message`, `Event`, `State`, `Agent`,
  `Queue`, hooks, and `run()` as a generator.
- `agent.py`: `AgentRuntime`, the stateful wrapper that owns the queues, the
  subscription list, and the abort flag.
- `demo.py`: a 3-agent debate that exercises `transform`, `convert_to_llm`,
  `subscribe`, and `steer`.

## What is borrowed from pi-mono

| Idea | Where in this version |
|---|---|
| Two-stage transform pipeline at the LLM boundary | `transform: TransformFn` + `convert_to_llm: ConvertToLlm` in `run()` |
| Runtime is a generator that yields events | `run()` returns `Iterator[Event]` |
| Steering / follow-up dual queues | `Queue` + `AgentRuntime.steer()` / `follow_up()` |
| `before_act` / `after_act` hooks with `block` and `terminate` | `BeforeActResult`, `AfterActResult` |
| Two API layers (functional + stateful) | `core.run()` for functional, `AgentRuntime` for stateful |
| `prompt()` vs `continue_()` | `AgentRuntime.prompt()` and `AgentRuntime.continue_()` |
| Static schedule list replaced with a function | `next_agent: Callable[[State], str | None]` |

## What was deliberately left out

These belong in `03_event_runtime`:

- First-class tool calling and parallel/sequential tool execution
- Streaming partial assistant messages
- AbortSignal threading through every callback
- Provider abstraction (`streamFn`, `getApiKey`, thinking budgets)

## Run

```bash
python3 examples/design_versions/02_balanced_runtime/demo.py
```
