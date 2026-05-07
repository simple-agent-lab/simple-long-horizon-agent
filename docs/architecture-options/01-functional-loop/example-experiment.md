# Example Experiment: Single-Agent Q&A With Last-N Context

This experiment exercises the only knob the version exposes: how much of
the message history the model sees on each call.

## Goal

Measure how answer quality and token cost change when `context_view` is
parameterized with `last=N` for different `N`.

## Setup

```python
from demo import Message, Tool, context_view, run_loop

def model(messages: list[Message], tools: list[Tool]) -> Message:
    # Replace the fake-provider model with another provider call here.
    ...

def trimmed_view(last: int):
    return lambda messages, instruction: context_view(messages, instruction, last=last)
```

## Variants

| Run | `last` | Expectation |
| --- | --- | --- |
| A | `None` (full history) | Highest quality, highest cost |
| B | `4` | Slightly lower quality, much lower cost |
| C | `1` | Quality collapses on multi-turn questions |

To swap the view, edit the single `model(context_view(...))` call inside
`run_loop` (or copy the loop and parameterize the view function).

## Metrics

- Number of model calls per run.
- Total prompt tokens.
- Final answer score (manual rubric or another model as judge).
- Whether the loop terminated via `kind == "final"` or hit `max_steps`.

## Why This Version Suffices

This experiment is single-agent and does not use tools, so the runtime
shape of 01 is exactly what it needs. Adding event sourcing or a
multi-agent scheduler would not change the answer.

## When to Move On

If the next experiment needs:

- Two or more agents talking to each other,
- A tool call inside the loop,
- Mid-run human nudges,
- A trace richer than the message list (token usage per call, per-step
  latency, structured stop reasons),

stop here and switch to version 02 or 03. The functional loop is not the
right shape for those questions.
