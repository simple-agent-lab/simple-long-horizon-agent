# `simple_agent_lab.llm` — LLM Access Layer

Provider-agnostic, sync-first wire layer. Used by `simple_agent_lab.core`.

Lives at `src/simple_agent_lab/llm/` — inside the installable package
(PyPA src-layout). Importable as `from simple_agent_lab.llm import ...`
once the package is installed (`uv sync` / `pip install -e .`).

This package knows **nothing** about agent loops, scheduling, tool dispatch,
or routing. It only knows how to talk to a model and stream events back.

## Public surface (intentionally small)

```python
from llm import (
    # data types
    Provider, LLMMessage, LLMTool, LLMRequest, LLMResponse,
    ContentBlock, ToolCall, Usage, StreamEvent,

    # message bridge
    messages_to_llm_messages, tool_to_llm_tool, llm_response_to_assistant_message,

    # entry points
    complete,            # blocking: req -> LLMResponse
    iter_stream,         # streaming: req -> Iterator[StreamEvent]
    register_adapter,    # extensibility: api -> stream fn
)
```

## The three-layer split

```
┌────────────────────────────────────────────────────┐
│  agent loop (01 / 02 / 03)                         │
│  - shared Message / State / Event with routing     │
│  - simple_agent_lab.tools owns tool values         │
│  - llm.bridge projects to LLMMessage at the        │
│    boundary; never lets routing fields leak        │
└────────────────────────────────────────────────────┘
                ↓ LLMRequest
┌────────────────────────────────────────────────────┐
│  llm/ (this package)                               │
│  - Pure provider-agnostic types                    │
│  - StreamEvent protocol                            │
│  - Adapter registry                                │
└────────────────────────────────────────────────────┘
                ↓ adapter dispatch (by Provider.api)
┌────────────────────────────────────────────────────┐
│  llm/adapters/                                     │
│  - fake.py        (built-in, deterministic)        │
│  - anthropic_messages.py    (TODO Step 2)          │
│  - openai_chat.py           (TODO Step 6)          │
│  - openai_responses.py      (TODO Step 3)          │
└────────────────────────────────────────────────────┘
```

## Quick examples

### Blocking call

```python
from llm import Provider, LLMMessage, LLMRequest, complete

provider = Provider(id="test", api="fake", model="fake-1")
resp = complete(LLMRequest(
    provider=provider,
    messages=[LLMMessage(role="user", content="Hi")],
))
print(resp.text, resp.stop_reason, resp.usage)
```

The built-in `fake` adapter is deterministic and reads the request messages,
system prompt, and tool definitions. It can emit deterministic tool calls when
the tool list makes the next step obvious, so examples exercise the same
boundary a real adapter would use. Examples should not pass response text
through `req.extra`.

### Streaming consumption

```python
from llm import iter_stream

for event in iter_stream(req):
    if event.kind == "text_delta":
        print(event.payload["delta"], end="", flush=True)
    elif event.kind == "tool_call_complete":
        tc = event.payload["tool_call"]
        # dispatch via your agent loop's tool layer
    elif event.kind == "done":
        final = event.payload["response"]
```

### Custom provider

```python
from llm import register_adapter, LLMRequest, StreamEvent, LLMResponse, Usage

def my_adapter(req: LLMRequest):
    # ...talk to your endpoint...
    yield StreamEvent(kind="text_delta", payload={"delta": "..."})
    yield StreamEvent(kind="done", payload={
        "response": LLMResponse(text="...", stop_reason="end_turn", usage=Usage()),
    })

register_adapter("my-api", my_adapter)
```

Then use `Provider(api="my-api", ...)` like any built-in.

## Design notes

- **Sync iterator** because 01/02/03 are sync. The protocol is shaped so
  an async variant can be added later without breaking sync callers.
- **`StreamEvent` is a single dataclass** with a `kind` discriminant +
  `payload: dict`, not one class per event type. Trades a bit of static
  typing for terseness; payload contracts are documented in
  `StreamEvent`'s docstring.
- **`complete()` requires a `done` event.** Adapters that forget to emit
  one will raise — fail loud, not silently truncate.
- **`Provider` carries no callable.** A custom Ollama setup is a literal:
  `Provider(api="openai-chat", base_url="http://localhost:11434/v1", ...)`.
- **`role="tool_result"` is internal, not wire-format.** Each adapter
  translates at its own boundary (OpenAI → `role="tool"` + tool_call_id;
  Anthropic → `role="user"` + a `tool_result` content block). Keeping
  the internal name distinct from any provider's role lets the same
  transcript reach either provider with no caller-side shimming.
- **`LLMMessage.cache_breakpoint`** is the unified caching marker. The
  layer doesn't auto-place breakpoints; the caller decides where to
  anchor cache reads. Adapters translate at the wire boundary; adapters
  for providers without caching ignore the field.
- **`req.extra: dict[str, Any]`** carries request options that only a
  specific provider adapter understands (e.g., Anthropic `extra_headers`,
  OpenAI `seed`). Adapters are free to read keys they recognize. The fake
  adapter only uses it for streaming mechanics such as chunk size and delay.

## What this layer does NOT do

- **Tool dispatch.** Shared tool values live in `simple_agent_lab.tools`, but
  execution stays in each agent loop's own dispatcher. This layer surfaces
  `tool_call_complete` events; the agent loop runs them.
- **Retry / backoff policy.** Adapters may do basic timeouts, but
  retries belong above (where the loop knows whether retry is safe).
- **Token counting / pricing math.** Adapters report `Usage`. Cost
  calculation is a one-liner in user code if needed.
- **Caching decisions.** Caller marks breakpoints; layer carries them.

## Implementation status

| Step | Adapter | Status |
| ---- | ------- | ------ |
| 1    | `fake`  | ✅ done (this PR) |
| 2    | `openai-chat`        | TODO - first live adapter target |
| 3    | `openai-responses`   | TODO |
| 4    | `anthropic-messages` | TODO |

Owner confirmation on 2026-05-11 chose `openai-chat` as the first live
provider adapter target. Its smoke should be opt-in: skip cleanly when no API
key or compatible local `base_url` is configured, and do not add it to required
CI until the owner explicitly accepts that dependency.

Once Step 2 lands, demos can swap their `Provider(api=...)` value while keeping
the same LLM request path, and streaming can be added by consuming `text_delta`
events rather than mutating recorded transcript messages.
