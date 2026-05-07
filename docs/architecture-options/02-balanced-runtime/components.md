# Components

The canonical implementation now lives in `src/simple_agent_lab/core.py`.
The 02 example folder keeps tiny compatibility re-exports so older demo
imports continue to run.

## `core.py`

### Data types

Tool-related data types come from `simple_agent_lab.tools`; the balanced
runtime only owns the dispatcher and event ordering.

| Type | Role |
| --- | --- |
| `MessageRole` | Stable model-facing role vocabulary: `"system"`, `"user"`, `"assistant"`, `"tool_result"`. |
| `MessageKind` / `MessageChannel` | Open string vocabularies for lab-level routing and recipes. Do not enum these until repeated mistakes prove the vocabulary is stable. |
| `MessageContentBlock` | Minimal structured content block for multimodal experiments; text remains the common path. |
| `ToolCallBlock` | The single project-owned tool-call shape: non-empty `id`, non-empty `name`, and `Mapping[str, Any]` arguments. Reused by assistant `Message.tool_calls` and assistant `ModelMessage` content. |
| `ModelUserMessage` / `ModelSystemMessage` / `ModelAssistantMessage` / `ModelToolResultMessage` | Role-specific provider-boundary message variants without runtime routing fields. |
| `ModelMessage` | The provider-boundary union of role-specific model-message variants. |
| `UserMessage` / `SystemMessage` / `AssistantMessage` / `ToolResultMessage` | Role-specific frozen message variants. Each variant owns only the fields that make sense for that role. |
| `Message` | The runtime transcript union of role-specific message variants. |
| `Event` (frozen) | `index + kind + data`. The trace element; `index` is event order, not an agent step. |
| `State` | Holds `task`, `events`, free-form `data` dict. |
| `Agent` (frozen) | `name + role + step` function. |
| `Tool` | Wire-format tool definition (`name + description + parameters`). JSON-serializable; safe to send to a provider. |
| `AgentTool` | `Tool` + local execution contract (`execute` + `label` + `execution_mode` + optional `timeout_seconds`). |
| `ToolContent` | One block of model-visible tool output (`kind="text"\|"image"`). |
| `ToolResult` | What `execute` returns: `content` (model-visible) + `details` (typed sidecar) + `is_error` + `terminate`. |

**Convention:** `transform` implementations and event listeners treat input
messages as immutable values. Future streaming experiments should use update
events or replace unrecorded temporary values rather than mutating recorded
transcript history.

The type boundary is intentionally small. Borrow the OpenAI SDK pattern of
project-owned typed values and explicit conversion helpers, but do not add a
`RunItem` inheritance layer, Pydantic validation, or provider SDK raw objects
to this runtime.

Shared message types live in `src/simple_agent_lab/messages.py`. The
design-version folders import those types instead of defining their own message
protocols. Match pi-mono's boundary semantics, not its naming: keep this
project's `Message` / `ModelMessage` terms rather than renaming the transcript
layer to `AgentMessage`.

Follow pi-mono's shape here: `Message` should be a role-specific union, not one
wide dataclass with many optional fields. Start with `UserMessage`,
`SystemMessage`, `AssistantMessage`, and `ToolResultMessage` so instruction,
summary, and runtime guidance remain inspectable in transcript state.
Request-level system prompt remains available for static agent or run
instructions. `SystemMessage` is for dynamic transcript-visible instruction,
summary, lesson, or runtime guidance.

Use `Message.kind` and `Message.channel` for custom runtime semantics that
pi-mono would express through `CustomAgentMessages`. `Message.data` is only a
rare sidecar for metadata that is not stable enough to become a named field or
project-owned type. Do not let common protocol concepts live indefinitely in
`data`.

Tool calls and tool-result identity are explicit shared `Message` fields. The
shared message module includes a field catalog and small helpers for
construction, projection, and validation so callers know which type owns which
field. `ToolCallBlock` replaces temporary dict-shaped tool-call records in the
shared protocol. Its `arguments` field is a mapping: do not mutate it in place;
adapters can shallow-copy it when building wire payloads.

Keep the API value-oriented: dataclasses carry fields, while module-level
functions handle construction, projection, validation, and common reads.
Constructor helpers should stay small: `user_message(...)`,
`system_message(...)`, `assistant_message(...)`, `tool_result_message(...)`,
`model_user_message(...)`, `model_assistant_message(...)`, and
`model_tool_result_message(...)`.
Use `tool_result` as the internal provider-neutral role. Provider adapters may
translate it to provider wire roles such as OpenAI's `tool`.
Keep `role` and `kind` separate: `role` is model-facing, while `kind` is
runtime-facing. A tool-result transcript entry normally has both
`role="tool_result"` and `kind="tool_result"`, but model projection should not
derive provider roles from `kind`.

Because multimodal providers are an expected future target, `ModelMessage`
should be a role-specific provider-neutral union rather than a flat
string-only chat dict. Provider adapters translate those variants and content
blocks into their wire-specific OpenAI, Anthropic, or other payloads. The first
block set should be text, image, thinking, and tool call; file/audio/video
should wait for concrete examples. Define the blocks and model-message variants
as frozen dataclasses in the shared message module, then let provider adapters
convert them to dict or SDK-specific wire payloads.

`Message.content` should support plain text or text/image blocks so user and
tool-result multimodal inputs can remain in the transcript. Thinking blocks and
tool-call blocks belong to `ModelMessage` or explicit `Message.tool_calls`,
not ordinary runtime content.

`ThinkingBlock` should still be preserved as structured model output for trace,
replay, and provider continuity. Preserving thinking does not imply it is
always rendered to users; display is a separate policy. Assistant `Message`
values should carry thinking in an explicit `thinking` field, not in `content`
or `data`. `ThinkingBlock` fields are `text`, `signature`, and `redacted`;
provider identity belongs outside the block.

`ImageBlock` should follow pi-mono's normalized shape: image `data` as a plain
base64 string plus `mime_type`. URL, file path, or provider file-id inputs
should be resolved before they become message content.

The shared `Message` should also be a frozen dataclass. The current
design-version implementation may remain mutable while it is a sketch, but the
project-owned type should make transcript history immutable by default.

### Functions and protocols

#### `context_view(agent, state, last=None) -> list[Message]`

Filters `state.messages` for the given agent: keep messages whose
`target in {agent.name, "all"}` or whose `sender == agent.name`. Optional
last-N truncation. The default visibility policy.

#### `model_message(message, with_header=True) -> ModelMessage`

Projects a `Message` into a model-facing role-specific dataclass. When
`with_header` is true and the message has lab metadata
(sender/target/kind/channel), prefixes the visible text content with
`[sender -> target | kind/channel]` so the model can see routing.
Assistant tool calls become `ToolCallBlock` entries in
`ModelAssistantMessage.content`; tool results carry `tool_call_id`,
`tool_name`, and `is_error` as explicit `ModelToolResultMessage` fields.

#### `message_tool_calls(message) -> tuple[ToolCallBlock, ...]`

Reads explicit `AssistantMessage.tool_calls`. Non-assistant messages return an
empty tuple. Malformed tool calls fail close to construction or projection
instead of leaking loose dict assumptions through the loop.

#### `sequence(*names) -> NextFn`

Built-in scheduler: yield each name once, then `None`. The user can
write any other `NextFn` (round-robin, mailbox, reactive).

#### `run(agents, state, next_agent, *, transform, last, tools, abort) -> Iterator[Event]`

The runtime. Drives the outer/inner loop; emits `agent_start`,
`turn_start`, `model_request`, `model_response`, `message`,
`tool_execution_start` / `_update` / `_end` (when an assistant message contains
explicit `tool_calls`), `turn_end`, `agent_end`. Each yielded event is also
recorded in `state.events`. The `abort` arg is a `Callable[[], bool]` polled in
the dispatcher; tool functions also receive it as their `signal` arg.

Events use one small dataclass: `Event(index, kind, data)`. Consumers switch
on `event.kind`, read structured fields from `event.data`, and use
`event.message` for recorded transcript messages.

`model_request` carries the scheduled agent name, visible-message count, a
compact visible-message outline, the `to_model_messages(visible)` payload, and
`state.data["candidate_id"]` when present. `model_response` carries the agent
name, output kind, target, tool-call count, and the same candidate id. Those two
events are the default comparison surface for self-evolution evals.

#### `run_to_completion(agents, state, next_agent, **kwargs) -> State`

One-line wrapper: drains the generator, returns the final state. Use
this when you do not need to subscribe to events.

#### Helpers: `default_role`, `message_text`, `last_message`, `print_trace`

Read-only utilities for building step functions and inspecting traces.

### Tools

Borrowed from pi-mono and trimmed to fit the rest of `core.py`. Six load-bearing
decisions:

1. **Shared two-layer split.** `Tool` is wire-format only (name + description
   + JSON-Schema parameters); `AgentTool` adds the local `execute` callable,
   a UI `label`, and `execution_mode`. Both live in `simple_agent_lab.tools`.
   The first is what you serialize to a provider; the second is what you run
   locally. They can diverge (e.g. a remote-execution proxy implements the
   second by RPC).
2. **`ToolResult` has `content` + `details`.** `content` is what the model
   sees on the next turn; `details: Any` is a typed sidecar for UI /
   listeners that never reaches the model. Avoids stuffing JSON into
   `content` and re-parsing in the UI.
3. **Errors are tool results, not exceptions.** `dispatch_tool_calls`
   wraps `execute` in try/except; a thrown exception becomes a
   `ToolResult(is_error=True)` whose `content` is the error text. The
   model gets it back and can self-correct. The only way a tool ends the
   run is `terminate=True` on its return.
4. **`kind="tool_result"` is a first-class message.** Not "a `user` role
   message containing a `tool_result` block" (Anthropic's wire shape).
   This keeps the transcript provider-agnostic; provider adapters
   translate at the boundary. `tool_result_message` keeps `tool_call_id`,
   `tool_name`, and `is_error` as explicit fields, and model projection skips
   the routing-header prefix for tool-result messages.
5. **Per-tool `execution_mode`, with sequential as a poison vote.** When
   the model emits multiple tool_calls, **any** tool with
   `execution_mode="sequential"` forces the entire batch sequential. This
   avoids "4 in parallel + 1 stuck waiting" footguns. Default is
   `"parallel"` (via `ThreadPoolExecutor`). Concurrency is capped at
   `max_concurrency` (default 8) to bound thread pressure when a model
   emits many tool calls at once.
6. **`on_update(complete_snapshot)`, not deltas.** A tool's `execute`
   may receive an `on_update: Callable[[ToolResult], None]` it can call
   with intermediate full-shape results (not deltas). Each call fires a
   `tool_execution_update` event carrying the snapshot. Consumers diff /
   replace; the runtime does not accumulate. Updates are buffered per
   tool and flushed alongside that tool's `tool_execution_end` event,
   so a fast tool's updates appear early even if a slow tool in the
   same batch is still running.

### Event ordering for slow tools

`tool_execution_start` events fire upfront in original tool-call order
(so listeners can immediately show "running..." for every tool). After
that, **end events fire in completion order**: a fast tool's
`tool_execution_end` flushes as soon as it finishes, even if other
tools in the batch are still running. The `tool_result` messages
themselves are appended to the transcript in **original tool-call
order** at the end, so the conversation log stays deterministic
regardless of completion timing.

### Tool timeouts

`AgentTool.timeout_seconds` (optional, default `None`) bounds the
dispatcher's wait. On timeout the tool's `tool_result` becomes
`is_error=True` with a "timed out" message and the run continues; the
model gets a chance to react. The tool's underlying thread keeps
running in the background; Python cannot safely kill a thread.
Cooperative cancellation via `abort()` polling is the tool author's
responsibility. For uncooperative work (third-party SDKs, opaque
subprocesses), wrap in a subprocess that can be `terminate()`d.

#### `dispatch_tool_calls(assistant_msg, tools, state, *, abort) -> Iterator[Event]`

Called automatically inside `run()` when an assistant output has
explicit `tool_calls`. Yields the `tool_execution_*` events and records a
`kind="tool_result"` message per call. If any tool returns
`terminate=True`, the run ends after that turn with
`reason="tool_terminate"`.

#### `make_tool_result_message(call_id, tool_name, result, *, target) -> Message`

The bridge from `ToolResult` (in-process value) to `Message`
(transcript). Joins text content blocks for the model-visible field;
keeps `tool_call_id`, `tool_name`, and `is_error` as explicit
`ToolResultMessage` fields; stashes `details` and the structured content
blocks under `message.data`.

### `AgentRuntime`

Owns the long-lived state (`State`), the listener list, and a cancel flag.
Construction takes the same small runtime knobs that `core.run` accepts;
`prompt` and `resume` only require a `next_agent` (and a target for `prompt`).

#### `prompt(task, *, target, next_agent) -> Iterator[Event]`

Resets state, seeds a `("task", "user", target)` message, and drives the
loop. The returned iterator pushes each event to subscribers and yields
to the caller. Code that owns the run should usually consume this iterator
directly; this is the authoritative execution stream and it provides natural
backpressure.

#### `resume(next_agent) -> Iterator[Event]`

Drives the loop without seeding a new message. The current state must
already have at least one message. Use this to retry after an error or
to resume after `abort`.

#### `subscribe(listener) -> unsubscribe`

Registers a `Callable[[Event], None]`. Listeners run synchronously on
each yielded event in registration order. The returned function removes
the listener. Use this for side observers (logging, metrics, UI mirrors,
persistence), not as the main way to drive a CLI or collect the final output.

#### `abort()`

Sets a cancel flag. Checked after each yielded event; on next check the
generator emits `agent_end{reason="aborted"}` and stops.

### Surface that is **not** present

Pi-mono has these; this version intentionally does not:

- A pluggable `streamFn` and `getApiKey`.
- `pendingToolCalls` state and listener-await settlement on `agent_end`.
- Mid-run input queues. Record explicit messages on `State` and use
  `resume()` when another run is needed.
- Streaming partial assistant messages.
- A full provider client lifecycle; the current helper routes through
  `simple_agent_lab.llm`.

Those belong around the runtime or in version 03 reference code until a
concrete self-evolution experiment proves they are needed.
