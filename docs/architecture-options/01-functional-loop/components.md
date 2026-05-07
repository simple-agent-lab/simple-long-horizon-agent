# Components

The whole version is a few functions in one file
(`examples/design_versions/01_functional_loop/demo.py`).

## `Message`

Uses the shared role-specific `Message` union from
`simple_agent_lab.messages`. The demo constructs messages through
`user_message`, `system_message`, `assistant_message`, and
`tool_result_message` rather than defining a local dataclass.

`role` and `content` are model-adjacent. `sender`, `target`, `kind`,
`channel`, and rare `data` are project metadata for tagging and filtering.
The loop itself only looks at `kind == "final"` and explicit assistant
`tool_calls`.

## `context_view(messages, instruction, last=None) -> list[Message]`

Builds the model input. Always prepends a system message carrying the
instruction; optionally truncates to the last `last` messages.

This is the **only** context boundary. To experiment with different
visibility policies (channels, summaries, role-based filtering), replace
or wrap this function — do not change `run_loop`.

## `make_model(provider, system_prompt, request_extra=None) -> ModelFn`

Builds a `ModelFn` against the shared `simple_agent_lab.llm` layer. The demo
uses the deterministic fake provider, but the loop only knows it received one
assistant `Message`.

Contract for any `ModelFn` substitute:

- Takes a `list[Message]` (the output of `context_view`).
- Takes a `list[Tool]` for provider-visible tool definitions.
- Returns a single `Message` with `role="assistant"`.
- Tags the final reply with `kind="final"` to stop the loop.

## `as_agent_message(agent_name, output) -> Message`

Re-tags the model's reply with sender/target metadata. Sets `target="user"`
when the reply is final, otherwise routes it back to the agent's own name.
Preserves the model's original sender under `data["model_sender"]`.

In a single-agent setup this is mostly cosmetic; it exists so the trace
reads cleanly when the version is compared against 02 and 03.

## `run_loop(task, instruction, model, tools=(), max_steps=3) -> list[Message]`

The loop itself. Seeds the message list with the user task, then iterates
*context_view -> model -> append -> dispatch local tools if any -> check final*
up to `max_steps` times.

Returns the full message list, which **is** the trace.

## Shared tool values

Tool data comes from `simple_agent_lab.tools`: `AgentTool` is imported under
the local name `Tool`, and `ToolResult` / `ToolContent` are the same values used
by 02 and 03. The dispatcher reads explicit assistant `tool_calls`, runs each
call sequentially, turns exceptions into `is_error=True`, appends one
`kind="tool_result"` message, and continues.

## `print_trace(messages)`

A 4-line helper that prints `index, kind, sender, content` for each
message. Not part of the runtime; lives in the same file because
everything else does too.
