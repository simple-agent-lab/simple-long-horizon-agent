# ADR 0006: Use A Role-Specific Message Protocol

## Status

Accepted

## Context

ADR 0001 chose a small message-first runtime, and ADR 0005 selected
`02_balanced_runtime` as the lead core candidate. The next shared boundary is
the message protocol itself: examples currently define local `Message` shapes,
tool calls are still partly carried through sidecar dictionaries, and future
multimodal providers need a clearer model-message boundary.

The project should borrow pi-mono's abstraction boundary without copying its
names. In pi-mono, the agent runtime works with `AgentMessage`, then converts
to a provider-neutral LLM `Message` only at the model-call boundary. In Simple
Agent Lab, the equivalent names are `Message` for runtime transcript values and
`ModelMessage` for provider-neutral model-call values.

## Decision

Create the shared message protocol under `src/simple_agent_lab/messages.py`.

Use role-specific frozen dataclasses rather than one wide dataclass with many
optional fields:

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
```

`Message` variants keep runtime transcript fields such as `sender`, `target`,
`kind`, and `channel`. `ModelMessage` variants remove those runtime routing
fields and keep only provider-neutral model-call fields.

Use content blocks for model-visible multimodal content:

- `TextBlock`
- `ImageBlock(data: str, mime_type: str)`, where `data` is plain base64 with
  no data-URL prefix
- `ThinkingBlock(text: str, signature: str | None, redacted: bool)`
- `ToolCallBlock(id: str, name: str, arguments: Mapping[str, Any])`

Runtime `Message.content` may contain plain text or text/image blocks.
Assistant messages preserve thinking in an explicit `thinking` field and tool
calls in an explicit `tool_calls` field. Tool result messages preserve
`tool_call_id`, `tool_name`, and `is_error` as explicit fields. Common protocol
concepts must not hide in `data`; `data` remains a rare sidecar for unstable or
uncommon metadata.

Use `tool_result` as the internal provider-neutral role. Provider adapters may
translate it to provider-specific wire roles such as OpenAI's `tool`.

Keep construction and projection simple. `Message` and `ModelMessage` are type
aliases, so construction should use module-level helpers:

```text
user_message(...)
system_message(...)
assistant_message(...)
tool_result_message(...)

model_user_message(...)
model_system_message(...)
model_assistant_message(...)
model_tool_result_message(...)
to_model_messages(...)
```

Request-level `system_prompt` remains available for static agent or run
instructions. `SystemMessage` is for dynamic, transcript-visible instruction,
summary, lesson, or runtime guidance.

## Consequences

The message protocol becomes easier to inspect because each role owns only the
fields that make sense for that role. This avoids a wide `Message` value full
of unused optional fields.

The provider boundary becomes clearer. Runtimes and examples use project-owned
`Message` and `ModelMessage` values; provider adapters translate those values
to OpenAI, Anthropic, or other wire payloads.

Multimodal support has a stable starting point. Text and images can live in the
runtime transcript, while thinking and tool calls remain structured assistant
output.

The tradeoff is a slightly larger type surface. Beginners must learn the four
message variants instead of one dataclass, but the field catalog is clearer and
closer to the actual protocol.

## Alternatives Considered

- Keep a single wide `Message` dataclass. Rejected because tool calls,
  thinking, tool-result identity, and sidecar metadata would produce too many
  optional fields.
- Rename runtime messages to `AgentMessage`. Rejected because the project
  already uses `Message` as the teaching term. We borrow pi-mono's boundary,
  not its names.
- Keep tool calls and tool-result identity in `data`. Rejected because these
  are core protocol concepts and should be explicit.
- Use provider-specific message shapes in shared code. Rejected because
  provider wire payloads belong inside adapters.
- Keep system prompts only as request fields. Rejected because summaries,
  lessons, and runtime guidance should be inspectable in transcript state.
