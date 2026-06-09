---
title: "Unify the Message Protocol on Content Blocks"
status: Accepted
date: 2026-05-12
slug: unify-message-protocol-on-content-blocks
note: "tool-result fragment superseded by `tool-result-as-content-block`"
---

# Unify the Message Protocol on Content Blocks

## Status

Accepted (supersedes parts of ADR use-role-specific-message-protocol). The tool-result fragments
of this ADR are further superseded by ADR tool-result-as-content-block, which collapses the
`tool_result` role into a `ToolResultBlock` content block and removes
the `ToolResultMessage` subtype entirely.

## Context

ADR use-role-specific-message-protocol set up the message protocol: role-specific frozen dataclasses
(`UserMessage`, `SystemMessage`, `AssistantMessage`, `ToolResultMessage`), a
parallel routing-stripped `ModelMessage` union, content blocks for multimodal
text/image, and sibling `thinking` / `tool_calls` fields on `AssistantMessage`.

Building real provider adapters (Anthropic Messages, OpenAI Chat, OpenAI
Responses, mimo-style OpenAI-compatible endpoints) and threading reasoning
through multi-turn tool use surfaced three frictions:

- The LLM access layer kept its own `ContentBlock` / `ToolCall` dataclasses
  parallel to the runtime ones in `messages.py`, with a translation step in
  `bridge.py`. Two sources of truth for one concept.
- Sibling `thinking` / `tool_calls` fields on `AssistantMessage` meant adapters
  had to re-assemble the wire order from three separate sources
  (`content` text, `thinking` tuple, `tool_calls` tuple). Anthropic Extended
  Thinking rejects out-of-order or unsigned thinking blocks on tool-use replay,
  so the adapter needed verbatim ordering and signatures preserved across
  turns — straightforward only if blocks live in one ordered list.
- The `Message → ModelMessage → LLMMessage` projection chain was effectively
  one transformation in two passes once content shapes matched. ModelMessage
  added a concept without earning its keep.

DeepSeek- / mimo-style endpoints also expose reasoning as a separate
`reasoning_content` field that the original protocol had no first-class home
for; it collapsed into a string on `LLMResponse.thinking` and was dropped on
replay.

## Decision

Make `ContentBlock` a single union defined once in `messages.py`:

```text
ContentBlock = TextBlock | ImageBlock | ThinkingBlock | ToolCallBlock
```

Every message at every layer stores `content: tuple[ContentBlock, ...]` in the
order the model produced the blocks. The wire ordering is the storage
ordering. `messages.normalize_content` accepts a `str` shorthand for terse
factory calls but the stored shape is always the tuple of blocks.

`AssistantMessage.thinking` and `.tool_calls` become derived `@property` views
over `content` — there is no separate storage. The same `@property` shape is
available on `LLMMessage` so adapters can ask `msg.tool_calls` /
`msg.thinking_blocks` without re-implementing the filter.

Reasoning is a first-class block kind. Provider adapters that surface
reasoning as a separate wire field (`reasoning_content` for DeepSeek / mimo,
`thinking` blocks with signatures for Anthropic) capture it into
`ThinkingBlock` entries inside `content`, in the order the model produced
them, with signature and `redacted` fields preserved. On the next turn the
adapter replays those blocks in the same shape it captured them — gated by
`Provider.replay_reasoning: bool = True`, a hidden opt-out for endpoints that
reject the replay shape (e.g., strict DeepSeek-tier servers).

Drop the `ModelMessage` layer. The bridge does the runtime → wire projection
in one step: `simple_agent_lab.llm.bridge.message_to_llm_message`. The
routing-header injection that used to live on `to_model_message` moves into
the bridge. The four `Model*Message` dataclasses, their factories,
`to_model_message(s)`, `model_message_text`, and `validate_model_message` are
removed.

The LLM access layer reuses the same block types from `messages.py`. The
duplicate `llm/types.py:ContentBlock` and `llm/types.py:ToolCall` are removed
(`ToolCall` survives as an alias of `ToolCallBlock` for one cycle of source
compatibility). `LLMResponse.usage` uses `TokenUsage` directly — the parallel
`Usage` dataclass is removed.

What ADR use-role-specific-message-protocol still holds:

- Role-specific frozen dataclass subtypes for `Message` (still
  `UserMessage`, `SystemMessage`, `AssistantMessage`, `ToolResultMessage`).
- `tool_result` as the internal provider-neutral role; adapters translate at
  the wire boundary.
- Module-level construction helpers (`user_message`, `assistant_message`, …).
- Content blocks as the multimodal representation — strengthened, since every
  layer now uses them.

What this ADR supersedes from 0006:

- The `Message` / `ModelMessage` two-layer split.
- `thinking` / `tool_calls` as sibling storage on `AssistantMessage`.
- `model_user_message`, `model_assistant_message`, `to_model_messages`, and
  related helpers.

## Consequences

Adding a new block kind is one dataclass in `messages.py` plus an isinstance
case in the adapters that need to handle it; no parallel definition in
`llm/types.py`.

Adapters dispatch on block kind with `isinstance(block, ThinkingBlock)`
rather than stringly-typed `block.kind == "thinking"` comparisons. The
type-checker narrows correctly without a helper.

Multi-turn tool use is continuous by construction: the assistant's prior
thinking + text + tool_call live in the same ordered tuple, the bridge
copies that tuple, and the next outbound call's adapter sees the same order
it captured. Anthropic's signature-verified replay and mimo's
`reasoning_content` replay are both expressible without sibling-field
choreography.

The bridge shrinks to ~one function. `LLMResponse.usage` uses `TokenUsage`
directly, eliminating a no-op translation.

The tradeoff is broader `content` type at the type level: `UserMessage` and
`SystemMessage` can technically hold a `ToolCallBlock` even though it makes
no sense semantically. We accept this for the uniform-storage win;
`validate_message` enforces stronger invariants where they matter (assistant
tool_calls have valid `id` and `name`, tool_result has `tool_call_id` and
`tool_name`).

`AssistantMessage.data: Sidecar` becomes a real channel for debug payloads
(see ADR extra-channel-and-two-layer-trace for `data["extra"]` and `data["wire"]`), so `message_text` no
longer falls back to printing the data dict — that fallback printed wire
dumps once the new channels landed.

## Alternatives Considered

- Keep `ModelMessage` as a routing-stripped projection layer. Rejected: with
  unified content, the only remaining difference was the absence of
  `sender / target / kind / channel` fields, and no adapter ever reads them.
  Bridge can simply not forward routing — no second type needed.
- Keep a flat tagged-union `ContentBlock` with a `kind` string field and
  every per-kind field optional. Rejected: each block ends up with 5–7
  fields where 1–2 are meaningful, and call sites compare strings instead of
  using `isinstance`. The per-kind dataclasses already existed in
  `messages.py`; reusing them was the obvious move.
- Keep sibling `thinking` / `tool_calls` fields and add a parallel `content`
  list. Rejected: two sources of truth for the same blocks invites drift,
  and the adapter still has to read both to know wire order.
- Keep reasoning as a single string on `LLMResponse.thinking`. Rejected:
  Anthropic Extended Thinking requires signatures and per-block structure for
  replay; a string loses that and prevents future per-block
  attributes (e.g., redacted_thinking).
- Replay reasoning off by default and require opt-in. Rejected: every
  reasoning-capable provider we currently target either accepts or requires
  replay. The hidden opt-out (`Provider.replay_reasoning=False`) is enough
  escape hatch.
