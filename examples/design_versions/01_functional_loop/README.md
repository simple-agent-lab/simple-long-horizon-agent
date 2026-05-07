# 01 Functional Loop

This is the smallest version and remains the teaching baseline. It is not the
lead self-evolution implementation; use it when the goal is to see the whole
agent loop by inspection.

Everything lives in `demo.py`:

- shared `Message` / `ToolCallBlock` imported from `simple_agent_lab.messages`
- shared `AgentTool` / `ToolResult` imported from `simple_agent_lab.tools`
- local `dispatch_tool_calls`
- `context_view`
- LLM-backed `ModelFn`
- `run_loop`

There is no `State`, `Agent`, `Event`, or runtime class. The message list is
the trace.

The loop is:

```text
context_view(messages) -> model -> append message
  [-> dispatch tool_calls -> append tool_result messages, then loop again]
  -> stop or continue
```

## Tools (optional)

Borrowed from pi-mono and aggressively trimmed. The data shapes now come from
`simple_agent_lab.tools`; 01 only owns the simplest dispatch rule. It keeps the
four load-bearing ideas:

1. **Use shared `AgentTool` under the local name `Tool`.** The example still
   reads like a single-file sketch, while the schema and result contract match
   02 / 03.
2. **`ToolResult.content` + `ToolResult.details`.** `content` is a list of
   model-visible `ToolContent` blocks; `details` is a sidecar for inspection
   and never reaches the model.
3. **Exceptions become `ToolResult(is_error=True)`** — the model gets the
   error text and can self-correct. Only `terminate=True` ends the run.
4. **`kind="tool_result"` is a first-class message.** Provider adapters
   would translate to wire format at the boundary; the transcript stays
   provider-agnostic.

Skipped vs 02: parallel execution, sequential-vote-poisons-batch,
streaming `on_update`, abort signal. None of these fit a 100-line single
file.

Usage:

```python
from simple_agent_lab.tools import text_result

echo = Tool(
    name="echo", description="Echo input back",
    parameters={"type": "object", "properties": {"msg": {"type": "string"}}},
    execute=lambda args: text_result(
        f"echo: {args['msg']}",
        details={"length": len(args["msg"])},
    ),
)
run_loop(task, instruction, my_model, tools=[echo])
```

A model function that wants tool use returns an assistant `Message` with
explicit `tool_calls`. The loop dispatches automatically and continues for
another turn so the model can react.

Run:

```bash
PYTHONPATH=src python3 examples/design_versions/01_functional_loop/demo.py
```
