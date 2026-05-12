# ADR 0014: Tool Results Are Content Blocks, Not a Separate Message Role

## Status

Accepted (supersedes the tool-result fragments of ADRs 0006 and 0012)

## Context

ADR 0012 unified message content on a single `ContentBlock` union, but
tool results stayed as their own `Message` subtype:

```text
ToolResultMessage(content, tool_call_id, tool_name, is_error, sender, ...)
```

with role `"tool_result"` — a 4th value distinct from `"system" /
"user" / "assistant"`. The asymmetry was visible at every layer:

```text
tool_call:     AssistantMessage.content = [..., ToolCallBlock(id, name, args)]    # block
tool_result:   ToolResultMessage(role="tool_result", tool_call_id=..., ...)        # message
```

Tool calls and tool results are the two halves of one operation, but the
codebase modeled the request as a block and the response as a message.

Two practical pressures forced the issue:

- **Parallel tool calling is the norm now.** Current frontier models
  routinely emit multiple `tool_call` blocks in a single assistant
  turn — for example, a coding agent dispatching three parallel
  `bash` commands. The runtime already supports this
  (`dispatch_tool_calls` uses a `ThreadPoolExecutor`), but it emitted
  N separate `ToolResultMessage`s afterward.
- **Wire-shape mismatch with Anthropic.** Anthropic's documented form
  is **one user message containing N `tool_result` content blocks**.
  Our N-separate-messages shape was technically valid Anthropic wire,
  but not idiomatic — and it forced the adapter to emit N adjacent
  user wire messages each carrying a single `tool_result` block.

OpenAI Chat's wire format goes the other way: each tool result is its
own `role="tool"` wire entry, looked up by `tool_call_id`. So whatever
shape we picked, one adapter would do the splitting/bundling work.

## Decision

`ToolResultBlock` joins the `ContentBlock` union:

```python
@dataclass(frozen=True)
class ToolResultBlock:
    tool_call_id: str
    tool_name: str
    content: tuple[TextBlock | ImageBlock, ...] = ()
    is_error: bool = False
```

`content` is a tuple of visible blocks so multimodal tool results
(screenshots, structured renders) ride through unchanged — multimodal
tool results are now mainstream, not future-tense.

`ToolResultMessage` is deleted. Tool results live inside `UserMessage`:

```python
UserMessage(
    content=(ToolResultBlock(...), ToolResultBlock(...), ...),
    sender="tool",
    target=<agent>,
    kind="tool_result",
)
```

One `UserMessage` bundles N parallel tool results from one assistant
turn. The runtime emits the bundle from `dispatch_tool_calls` as one
state record, not N.

`Role` shrinks from 4 values to 3:

```text
Role = Literal["system", "user", "assistant"]
```

`"tool_result"` is no longer a role — it was always internal (no
provider's wire used it). What remains is `message.kind == "tool_result"`
as a label for context_view filtering and trace discrimination; the
runtime invariant "this user message is a tool-result bundle" is
checked by `is_tool_result_message(message)`.

`is_error` moves from message-level to per-block. Parallel calls can
have mixed outcomes (some succeed, some fail) — per-block is the
correct location.

The previously-dead `LLMMessage.tool_call_id` field is removed; its
content moved onto `ToolResultBlock.tool_call_id`.

Adapter wire translation per provider:

- **Anthropic**: one bundled `UserMessage` becomes one `{"role":
  "user", "content": [{"type": "tool_result", "tool_use_id": ...,
  "content": ...}, ...]}` wire entry. Idiomatic.
- **OpenAI Chat**: one bundled `UserMessage` becomes N `{"role":
  "tool", "tool_call_id": ..., "content": ...}` wire entries.
- **OpenAI Responses**: one bundled `UserMessage` becomes N
  `{"type": "function_call_output", "call_id": ..., "output": ...}`
  items.

`context_view` pairs assistant tool_calls with subsequent
tool_result bundles by scanning `tool_results_of(candidate.content)`
for any matching `tool_call_id` from the assistant's wanted set.

`message_text` falls through to the first `ToolResultBlock`'s visible
text when a message's top-level `content` has no `TextBlock` — so
tool-result bundles still produce a useful preview in the trace.

## Consequences

Parallel tool-call sessions become structurally honest: one assistant
turn → one bundled tool-result user message → next assistant turn.
The runtime's existing concurrent execution is no longer hidden behind
N adjacent single-block messages.

Anthropic wire is now idiomatic at the protocol layer, not just
technically valid. OpenAI wire keeps the "one role=tool per result"
shape via per-block splitting inside the adapter.

`tool_call` and `tool_result` finally sit at the same conceptual
layer (both blocks), making the protocol easier to teach.

`Role` is 3-valued and matches every real provider's wire role.

The tradeoffs:

- `UserMessage.content` can now contain `ToolResultBlock`. The "this
  is a user-input message vs a tool-result bundle" distinction is no
  longer encoded in the type — callers check `message.kind` or use
  `is_tool_result_message(...)`. validate_message enforces the
  required `tool_call_id` / `tool_name` invariants per block.
- `message.sender` for a bundle is `"tool"` (a fixed label) rather
  than a specific tool name. Per-tool identity lives on each block's
  `tool_name`, which is correct for parallel-mixed bundles. Trace
  output shows per-block `tool_name` directly.
- Code that branched on `message.role == "tool_result"` now branches
  on `message.kind == "tool_result"` (or `is_tool_result_message`).

## Alternatives Considered

- **Keep `ToolResultMessage` as a separate subtype, but allow N
  results per message.** Would let parallel bundles share a message
  but keeps the role asymmetry with `ToolCallBlock`. Half a step;
  not worth the partial migration.
- **Keep one `ToolResultMessage` per result and let the adapter
  group adjacent ones.** Pushes the bundling logic into every adapter
  that prefers the bundled wire shape, and forces context_view to
  invent group-spanning logic. The conceptual mismatch is upstream;
  fix it upstream.
- **Make `ToolResultBlock.content` a plain `str` rather than a tuple
  of visible blocks.** Simpler, but loses the multimodal channel that
  current models already exercise (screenshots returned by browser
  tools, image diffs returned by visual diff tools).
- **Keep `Role = "tool_result"` even though it never reaches the
  wire.** Adds a fourth value that exists only to discriminate
  internal messages — `message.kind == "tool_result"` already does
  this job, so the role value is redundant.
