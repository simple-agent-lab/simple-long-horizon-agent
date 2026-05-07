# Reference Architecture: OpenAI Agents SDK Run Items

## Source

- Link: pasted SDK excerpt in the local design discussion
- Date reviewed: 2026-05-06
- Reviewer: Codex

## Summary

The OpenAI Agents SDK uses typed run items to wrap raw model outputs, tool
calls, tool outputs, approvals, handoffs, and reasoning records. Each item keeps
the raw provider object but exposes a project-owned type and a small conversion
surface such as `to_input_item()`.

The useful lesson for Simple Agent Lab is not the full class hierarchy. The
lesson is to put type constraints at the boundaries where values cross between
agent state, model payloads, and tools.

## Core Ideas

- Keep project-owned types around provider data instead of letting raw provider
  objects spread through the runtime.
- Give replay/projection behavior a named helper rather than repeating ad hoc
  dict shaping.
- Accept that some fields are output-only and should be stripped or normalized
  before sending data back to a model.
- Keep tool call identity accessible through small properties or helpers.

## Agent Loop

The SDK treats a run as a sequence of typed items. Simple Agent Lab should keep
the smaller message-first loop:

```text
Agent + Message + State + context_view() + run()
```

`State.events` remains the trace. `Message` remains the transcript unit.
The conversion helpers decide what becomes model input or tool execution input.

## Tool Model

The SDK has many tool item variants because it supports hosted tools, MCP,
local shell, approvals, handoffs, and provider-specific raw objects.

Simple Agent Lab only needs the current split:

- `Tool`: provider-facing definition
- `AgentTool`: local execution metadata
- `ToolCallBlock`: frozen project-owned shape carried by assistant messages and
  assistant model messages, with non-empty identity fields
- `ToolResult`: model-visible content plus local `details`

## State and Memory

The SDK keeps raw response items and converts them back into input items for
continuations. Simple Agent Lab should avoid provider raw objects in core state.
State should remain inspectable through `State.events`, messages, and plain
payloads.

## What We Might Borrow

- Literal role types for stable model-facing values.
- Frozen dataclass shapes for provider-boundary messages and tool-call blocks.
- One explicit helper for reading assistant `Message.tool_calls`.
- Boundary validation with actionable errors when a message carries malformed
  tool-call data.

## What We Should Avoid

- A `RunItemBase` inheritance tree before the runtime has enough item kinds to
  justify it.
- Pydantic as a core dependency while the project is still stdlib-only.
- Provider SDK raw response objects inside `State.events`.
- Encoding every recipe-level `Message.kind` as an enum. Recipe vocabularies
  are still experimental and should stay open strings.

## Notes

The balanced runtime now uses this trimmed-down borrowing: constrained
`MessageRole`, open `MessageKind`, typed `ToolCallBlock`, typed
`ModelMessage`, and `message_tool_calls(...)` at the projection/execution
boundary. Shared message definitions live in `src/simple_agent_lab/messages.py`
and are reused by examples.

This should stay semantically compatible with the pi-mono boundary: Simple
Agent Lab `Message` is the runtime transcript layer, while `ModelMessage` is
the provider-neutral model-call layer.

For custom runtime semantics, prefer open `Message.kind`, `Message.channel`,
and structured `Message.data` over introducing a parallel custom-message
union.

Do not let common protocol concepts hide in sidecar dictionaries. Tool calls
and tool-result identity should become explicit project-owned fields or
content blocks with helper methods for projection and validation.
Use one `ToolCallBlock` shape for assistant message tool calls and assistant
model-message content; avoid parallel dict-shaped tool-call records.
Use `tool_result` as the internal role name; provider adapters can translate it
to OpenAI's wire role `tool` or another provider-specific representation.

For the provider boundary, prefer pi-mono-style content blocks so future
multimodal models share one `ModelMessage` protocol. The first useful block
set is text, image, thinking, and tool call. Use frozen dataclasses for these
project-owned values, then convert to provider wire dicts at the adapter edge.
