# 02 Balanced Runtime

> The lead core candidate for self-evolution work. A small multi-agent
> core with a generator-based run loop, request/response trace events, tool
> dispatch, and agent-as-tool delegation. Borrowed in spirit from
> [pi-mono](https://github.com/badlogic/pi-mono).
>
> Implementation:
> - [`src/simple_agent_lab/core.py`](../../../src/simple_agent_lab/core.py) - canonical balanced runtime.
> - [`examples/design_versions/02_balanced_runtime/core.py`](../../../examples/design_versions/02_balanced_runtime/core.py) - compatibility re-export.
> - [`examples/design_versions/02_balanced_runtime/agent.py`](../../../examples/design_versions/02_balanced_runtime/agent.py) - `AgentRuntime` compatibility re-export.
> - [`examples/design_versions/02_balanced_runtime/demo.py`](../../../examples/design_versions/02_balanced_runtime/demo.py) - agent-as-tool delegation using `AgentTool` dispatch.
>
> See also [`docs/reference-architectures/pi-mono-agent-runtime.md`](../../reference-architectures/pi-mono-agent-runtime.md).

## What it optimizes for

A small core a real project would actually reuse. Multi-agent capable but
not multi-runtime; tool-capable but not stream-aware; provider-agnostic but
without a heavy provider abstraction. It is now the
first place to add self-evolution harness behavior from ADR 0004: trace,
eval, candidate comparison, and acceptance gates.

## Runtime Shape

```text
state.send("task", user, target_agent)
  -> agent_start
  -> while next_agent(state) yields a name:
       turn_start
       visible = context_view(agent, state, last)
       visible = transform(visible)               # optional context experiment
       llm_payload = to_model_messages(visible)
       model_request event                        # visible + payload
       output = agent.step(agent, visible, state)
       model_response event                       # kind + target + tools
       record(output)
       dispatch tool calls, if present
       turn_end
  -> agent_end
```

The whole thing is a generator that yields `Event` objects. The consumer
either iterates events live or calls `run_to_completion` to drain and read the
final `State`. In the stateful wrapper, `subscribe(...)` is a side-observer
hook; it mirrors the same events for logging, metrics, UI, or persistence, but
it is not the primary output path.

## Core Ideas

- **One visible context path.** `context_view(...)` chooses what an agent can
  see, `transform(...)` can run a small context experiment, and
  `to_model_messages(...)` records the model-facing payload for tracing. The
  core does not expose a second conversion hook.
- **Runtime is a generator.** `run(...) -> Iterator[Event]`. Events are the
  primary output stream; consumers pull them directly when they own the run.
- **Request and response events make evals comparable.**
  Each agent turn emits `model_request` and `model_response` around
  `agent.step`. The request event includes a compact visible-message outline,
  the LLM payload when available, and `state.data["candidate_id"]` when a
  caller is comparing baseline and candidate runs.
- **An agent step is one visible transition.** `Agent.step` reads the
  current visible context and emits the next `Message`. In the default
  LLM-backed helper (`make_llm_step`), one step maps to one model call.
- **No hidden input queues in the canonical core.**
  Extra user input is recorded explicitly on `State`, then driven with
  `resume()` if another run is needed.
- **Scheduling is a function, not a list.**
  `next_agent: Callable[[State], str | None]` lets `sequence(...)`,
  `round_robin`, mailbox, and reactive routing all share the same
  runtime. Returning `None` ends the inner loop.
- **State is the event log.**
  `State.events: list[Event]` is the durable trace; `State.messages` is a
  derived property over events of `kind == "message"`. There is no
  separate trace store.
- **Two API layers, same semantics.**
  - Functional: `core.run(agents, state, next_agent, ...) -> Iterator[Event]`.
  - Stateful: `AgentRuntime` class with `subscribe / abort / prompt /
    resume`. The class wraps the function and adds a
    listener list for side observers plus a cancel flag.
- **`prompt()` and `resume()` are first-class.** `prompt(task, target,
  next_agent)` starts a new run; `resume(next_agent)` resumes from
  existing state. Pi-mono shape.

## Data Model

```text
Message =
  UserMessage
  | SystemMessage
  | AssistantMessage
  | ToolResultMessage

ModelMessage =
  ModelUserMessage
  | ModelSystemMessage
  | ModelAssistantMessage
  | ModelToolResultMessage

AssistantMessage
  content     str | tuple[TextBlock | ImageBlock, ...]
  thinking    tuple[ThinkingBlock, ...]
  tool_calls  tuple[ToolCallBlock, ...]

Event
  index    int                    # event order, not an agent step
  kind     "agent_start" | "agent_end" | "turn_start" | "turn_end"
           | "message" | "model_request" | "model_response"
           | "tool_execution_start" | "tool_execution_update"
           | "tool_execution_end"
  data     dict                   # structured fields for that event kind

Common event data fields
  message event            data["message"]
  turn events              data["agent"], data["terminated"]
  model_request            data["agent"], data["visible"], data["llm_payload"]
  model_response           data["agent"], data["output_kind"], data["tool_call_count"]
  tool_execution_*         data["tool_call_id"], data["tool_name"], data["terminate"]

State
  task     str
  events   list[Event]
  data     dict                  # scratch + last_llm_payload + candidate_id

Agent
  name     str
  role     str
  step     StepFn = (Agent, list[Message], State) -> Message
```

Function types:

```text
TransformFn   = list[Message] -> list[Message]
NextFn        = State -> Optional[str]
```

## Built-In Coordination Patterns

The runtime ships only one helper, `sequence(*names)`. Other patterns are
expressed by writing a `NextFn`:

| Pattern | Sketch |
| --- | --- |
| Fixed order | `sequence("proposer", "critic", "judge")` |
| Round-robin until done | iterate `["a","b"]` while `state.data["done"]` is False |
| Mailbox | inspect last message's `target`; return that name |
| Reactive | inspect last message's `kind`; map kinds to agents |
| Until-N-turns | count `turn_end` events in `state.events` |

## Context Strategy

Each agent's visible context is built from:

```text
context_view(agent, state, last)
  -> messages whose target in {agent.name, "all"} or whose sender == agent.name
  -> optionally truncated to the last N
  -> optionally transformed by `transform`
  -> [boundary] handed to agent.step
```

`last_llm_payload` (the result of `to_model_messages(visible)`) is stashed on
`state.data` after each turn for inspection and logging. The same payload is
included on `model_request` so evals can compare what crossed the model
boundary without adding another hook.

## Trace and Evaluation

`state.events` is the trace. It contains:

- Lifecycle markers: `agent_start`, `turn_start`, `turn_end`, `agent_end`.
- Every recorded message as a `kind="message"` event.
- Every agent turn's model-facing boundary as `model_request` and
  `model_response`.
- Tool execution boundaries as `tool_execution_start`, `tool_execution_update`,
  and `tool_execution_end`.
- The terminating reason on `agent_end` (`"done" | "aborted" |
  "tool_terminate"`).

For self-evolution, eval code should set `state.data["candidate_id"]` before
running a candidate. The request/response events then give a cheap comparison
surface: which context was visible, which payload crossed the LLM boundary,
what kind of output came back, whether tools were requested, and how the run
ended. There is no token-usage or latency event yet; add those as plain events
only when an experiment needs them.

## Strengths

- Multi-agent capable with one canonical `src` core.
- Generator-based events make normal CLI and eval consumers simple: iterate
  the stream directly. `subscribe` stays useful for side observers.
- `context_view` plus a single optional `transform` keeps context experiments
  visible without making the model boundary configurable too early.
- Request/response events make prompt, context, tool-description, and recipe
  candidates easier to compare without instrumenting every example.
- Tool calls are first-class enough for small experiments: provider-facing
  tool definitions stay separate from local execution, and results are
  recorded as messages.
- Two API layers let students use `core.run(...)` directly while UI code
  uses `AgentRuntime`.

## Weaknesses

- More to learn than 01: the runtime adds scheduling, request/response events,
  and tool dispatch.
- No streaming partial messages. The runtime treats each `step` call as a
  single returned `Message`.
- No full provider client abstraction. The helper `make_llm_step` uses the
  shared `simple_agent_lab.llm` boundary, but richer provider lifecycle belongs
  in a higher layer or version 03 reference code.
- Tool execution has cooperative cancellation and timeout handling, but not a
  full stream-aware UI contract.

## When to Pick This Version

Pick this version when the team's primary use case involves:

- Two or more agents collaborating, including agent-as-tool delegation, AND
- Self-evolution experiments that compare prompt, context, tool description,
  or recipe candidates, AND
- A small runtime that needs tools and comparable request/response traces.

If the goal is to teach the smallest possible loop, see version 01. If the
goal is to study a provider abstraction or richer event sourcing in isolation,
see version 03.

## See Also

- [components.md](components.md): module-by-module breakdown.
- [example-experiment.md](example-experiment.md): a debate experiment
  that exercises transform and live event consumption.
- [`evolution_probe.py`](../../../examples/design_versions/02_balanced_runtime/evolution_probe.py):
  the smallest baseline-vs-candidate comparison using `candidate_id`.
