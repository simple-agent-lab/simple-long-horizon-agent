# Simple Agent Lab Context

Simple Agent Lab is a teaching and experimentation context for small agent
runtimes. Its language should keep runtime concepts separate from provider API
payloads so students and agents can reason about boundaries clearly.

## Language

**Message**:
A runtime transcript message that carries agent-facing communication plus lab routing fields.
_Avoid_: Provider message, model payload

**LLMMessage**:
A provider-agnostic model-call payload projected from a runtime **Message**.
_Avoid_: Message, transcript message, provider wire payload

**Content Block**:
A typed unit of model-visible content such as text, image, thinking, tool call, or tool result.
_Avoid_: Ad hoc content dict, provider-specific content part

**Provider Adapter**:
A boundary that translates project-owned model request and response values for a specific model provider.
_Avoid_: Runtime core, agent loop

**Message Type**:
A project-owned message data shape reused by runtimes or examples.
_Avoid_: Example-local message protocol

**UserMessage**:
A **Message** variant representing user-provided input, task context, or tool-result bundles.
_Avoid_: Generic message with unused assistant fields

**RuntimeMessage**:
A **Message** variant representing system, instruction, summary, or runtime guidance that should remain visible in transcript state.
_Avoid_: Hidden prompt-only instruction

**AssistantMessage**:
A **Message** variant representing model or agent output, including optional thinking and tool calls.
_Avoid_: Generic message with sidecar tool-call data

**ToolResultBlock**:
A **Content Block** representing one tool result inside a tool-result **UserMessage**.
_Avoid_: ToolResultMessage, generic message with hidden tool result metadata

**Message Sidecar**:
Rare escape-hatch metadata attached to a **Message** when the structure is not stable enough to become a named field or type.
_Avoid_: Primary message field, common protocol field

## Relationships

- A **Message** is projected into an **LLMMessage** before a model call.
- A **Provider Adapter** may translate an **LLMMessage** differently for each model provider.
- A **Message** remains in runtime state; an **LLMMessage** belongs at the provider-agnostic LLM boundary.
- **Message** should be a role-specific union of message variants rather than one wide dataclass with many optional fields.
- **RuntimeMessage** belongs in the same **Message** union so instruction, summary, and runtime guidance can remain inspectable in state and trace.
- **LLMMessage** carries model-facing role and content, but not runtime routing fields such as `sender`, `target`, `kind`, or `channel`.
- Request-level system prompt is for static agent or run instructions. **RuntimeMessage** is for dynamic transcript-visible instruction, summary, lesson, or runtime guidance.
- An **LLMMessage** uses **Content Blocks** so multimodal providers can share one project-owned boundary.
- The current **Content Block** set is text, image, thinking, tool call, and tool result. File, audio, and video blocks wait for concrete use cases.
- **Content Blocks**, **Message**, and **LLMMessage** should be frozen dataclasses so provider adapters convert immutable project-owned values into provider wire payloads explicitly.
- Recorded transcript history should not be mutated in place.
- Every **Message** and **LLMMessage** stores `content: tuple[ContentBlock, ...]`; validators enforce which blocks are legal on which message roles.
- **ImageBlock** uses normalized base64 image data plus MIME type (`data`, `mime_type`), following pi-mono's boundary. `data` is a plain base64 string without a data-URL prefix. URL, file path, or provider file-id inputs must be resolved before becoming an **ImageBlock**.
- **ThinkingBlock** is important model output and should be preserved as structured data for trace, replay, and provider continuity. Whether it is shown to users is a separate visibility decision.
- Assistant **Message** values preserve thinking as ordered **ThinkingBlock** entries in `content`; `AssistantMessage.thinking` is a derived view.
- **ThinkingBlock** fields are `text`, `signature`, and `redacted`. The signature is provider-continuity metadata; provider identity belongs outside the block.
- A **Message Type** should have one project-owned source of truth; example folders may demonstrate runtime behavior but should not become the long-term source of message protocol definitions.
- Simple Agent Lab should align with pi-mono's message boundary semantics without copying its names: this project's **Message** corresponds to pi-mono's `AgentMessage`, and this project's **LLMMessage** corresponds to the provider-agnostic LLM message boundary.
- Custom runtime semantics should be represented on **Message** through `kind`, `channel`, and structured `data`, then filtered or projected at the **LLMMessage** boundary.
- Frequently used message semantics should become explicit fields or named project-owned types. **Message Sidecar** is only for uncommon or still-experimental metadata.
- Tool calls and tool-result identity are common protocol concepts, so they belong in explicit **Content Blocks** rather than in **Message Sidecar**.
- **ToolCallBlock** is the single project-owned tool-call shape. Assistant **Message** values and assistant **LLMMessage** values reuse it instead of carrying separate dict-shaped tool-call records.
- **ToolCallBlock.arguments** is typed as `Mapping[str, Any]`: callers should not mutate it in place, while adapters may shallow-copy it to provider wire dictionaries.
- **ToolResultBlock** is the single project-owned tool-result shape. Tool results live inside `UserMessage.content`; one tool-result **UserMessage** may bundle multiple result blocks from a parallel tool-call turn.
- A tool-result **UserMessage** normally has `role="user"` and `kind="tool_result"`. Provider adapters may translate each **ToolResultBlock** to provider wire shapes such as OpenAI's `role="tool"` entries or Anthropic's `tool_result` content blocks.
- Each **Message Type** should have a documented field catalog plus named construction, projection, or validation helpers where plain construction would be ambiguous.
- **Message Type** behavior should stay in module-level construction, projection, validation, and helper functions.
- Shared **Message** constructor helpers are `user_message(...)`, `runtime_message(...)`, `assistant_message(...)`, `tool_result_message(...)`, and `tool_results_message(...)`.
- `llm.llm_message(...)` constructs an **LLMMessage**. Runtime projection uses `llm.bridge.message_to_llm_message(...)` and `messages_to_llm_messages(...)`.
- `role` is model-facing and `kind` is runtime-facing. Projection code should not use `kind` as a provider role.
