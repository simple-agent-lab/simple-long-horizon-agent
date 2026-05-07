# 01 Functional Loop

> The minimal candidate. One file, one function, no class hierarchy.
> Implementation: [`examples/design_versions/01_functional_loop/demo.py`](../../../examples/design_versions/01_functional_loop/demo.py).

## What it optimizes for

Reading and understanding an agent loop in one sitting. Everything that is
not strictly required to express *messages → model → append → stop* has
been removed.

## Runtime Shape

```text
user task
  -> user_message(task)               # seed
  -> for step in 1..max_steps:
       context_view(messages, instr)  # snapshot of what the model sees
       model(visible, tools)          # one assistant message
       as_agent_message(name, msg)    # tag with sender/target/kind
       messages.append(reply)
       dispatch reply.tool_calls, if any
       if reply.kind == "final": break
  -> messages                         # the trace is the message list
```

## Core Ideas

- **The message list is the trace.** There is no separate `State`, no
  `Event`, no `RunResult`. To replay a run, replay the messages.
- **`context_view` is the only context boundary.** It builds the model
  input from `messages + instruction`, optionally truncated by `last`. Any
  smarter context policy (windowing, summary, multi-channel filtering) is
  expressed by replacing this single function.
- **Model is a function `list[Message], list[Tool] -> Message`.** Provider
  details still stay outside the loop; this demo uses the shared deterministic
  fake provider through the shared LLM bridge.
- **Tool data is shared, dispatch is local.** The example imports
  `AgentTool`, `ToolResult`, and helpers from `simple_agent_lab.tools`, then
  uses a tiny local dispatcher. This keeps 01 aligned with 02 / 03 without
  giving it their scheduling semantics.
- **Stop signal is a `kind == "final"` convention.** Not an enum, not a
  status code. Any model implementation can opt in by tagging its last
  reply.
- **Single agent.** No `next_agent`, no scheduler, no routing. Multi-agent
  is deliberately deferred to versions 02 and 03.
- **Shared `Message` protocol.** The demo imports the shared role-specific
  message dataclasses instead of defining a local protocol.

## Data Model

```text
Message
  role     "system" | "user" | "assistant" | "tool_result"
  content  str | tuple[TextBlock | ImageBlock, ...]
  sender   str
  target   str
  kind     str                       # "task", "thought", "final", ...
  data     Mapping[str, Any]
```

No `Event`, `Agent`, `State`, or `Config` types. The only function-shaped
type is:

```text
ModelFn = Callable[[list[Message], list[Tool]], Message]
```

## Control Flow

The complete loop, copied verbatim:

```python
def run_loop(task, instruction, model, tools=(), max_steps=3) -> list[Message]:
    messages = [user_message(task, sender="user", target="assistant", kind="task")]
    for _ in range(max_steps):
        output = model(context_view(messages, instruction), list(tools))
        message = as_agent_message("assistant", output)
        messages.append(message)
        if message_tool_calls(message):
            messages.extend(dispatch_tool_calls(message, tool_registry))
            continue
        if message.kind == "final":
            break
    return messages
```

That is the entire runtime.

## What This Version Avoids

- No `State`, `Event`, or trace store.
- No multi-agent coordination, scheduling, or routing.
- No parallel dispatch, provider class, or event stream.
- No streaming, no cancellation, no retry.
- No mid-run injection, no follow-up queue, no hooks.
- No structured stop reasons. `kind == "final"` is the contract.

## Strengths

- Smallest possible reading cost. A new contributor sees the whole loop in
  ~30 lines.
- Trivially testable: `run_loop` is a pure function from a `ModelFn` to a
  message list.
- Hardest to misuse. There is almost nothing to configure incorrectly.

## Weaknesses

- Stops being useful the moment two agents need to talk to each other.
- Tool dispatch is intentionally basic and local; anything beyond a simple
  sequential call should move to 02 or 03.
- No structured observability beyond the message list.
- Every feature requires editing the loop directly; there is no extension
  point.

## When to Pick This Version

Pick this version when the team's primary use case is **teaching** or
**reading**, not **using**. If the only thing you ever build on top of
this is a single-agent question-answering bot with no runtime features beyond
optional simple tools, this is the
right choice. If multi-agent recipes (debate, pipeline, voting) or tool
calling are real near-term goals, see versions 02 and 03.

## See Also

- [components.md](components.md): the four functions that make up the
  whole version.
- [example-experiment.md](example-experiment.md): how to run a
  question-answering experiment with this version and what to measure.
