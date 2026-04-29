# Design Versions

These folders compare three possible Python shapes for a tiny agent runtime.

They are runnable sketches, not competing production implementations.

The three folders move along a single axis — minimal, balanced, rich —
so the comparison stays about runtime architecture, not agent roles or
workflow recipes:

```text
01 minimal:   messages -> context_view -> model -> append message
02 balanced:  schedule list -> agents.act(visible, state) -> state.events
03 rich:      event log -> model request/response events -> projections
```

## 01 Functional Loop (minimal)

One file. No `State`, `Event`, or loop class. The core is just
`run_loop(messages, model)`. Best for teaching the loop in a few minutes.

```bash
python3 examples/design_versions/01_functional_loop/demo.py
```

## 02 Balanced Runtime

Single-file multi-agent runtime: `Agent + Message + State + context_view + run`,
plus helpers (`last_message`, `model_message`, `print_trace`). Each agent is
a function closure, so provider and stop logic stay out of the runtime. Best
when you want a small core you would actually reuse in a real project.

This is mirrored from `simple_agent_lab/core.py` so all three versions sit
side by side.

```bash
python3 examples/design_versions/02_balanced_runtime/demo.py
```

## 03 Event Runtime (rich)

Event-sourced runtime. It records `model_request`, `model_response`, `message`,
and `stop` events. Best when you want observability, replay, eval hooks, and
provider adapters.

```bash
python3 examples/design_versions/03_event_runtime/demo.py
```

For live OpenAI Responses usage later:

```bash
python3 examples/design_versions/03_event_runtime/demo.py --provider openai --model gpt-5-mini
```
