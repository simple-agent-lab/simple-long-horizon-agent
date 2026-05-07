# Simple Agent Lab Context

Simple Agent Lab is a teaching and experimentation context for small agent
runtimes. Its language should keep runtime concepts separate from provider API
payloads so students and agents can reason about boundaries clearly.

## Language

**Message**:
A runtime transcript message that carries agent-facing communication plus lab routing fields.
_Avoid_: Provider message, model payload

**ModelMessage**:
A provider-facing payload derived from one or more **Messages** for a model call.
_Avoid_: Message, transcript message

**Content Block**:
A typed unit of model-visible content such as text, image, thinking, or tool call.
_Avoid_: Ad hoc content dict, provider-specific content part

**Provider Adapter**:
A boundary that translates project-owned model request and response values for a specific model provider.
_Avoid_: Runtime core, agent loop

**Message Type**:
A project-owned message data shape reused by runtimes or examples.
_Avoid_: Example-local message protocol

**UserMessage**:
A **Message** variant representing user-provided input or task context.
_Avoid_: Generic message with unused assistant/tool fields

**SystemMessage**:
A **Message** variant representing system, instruction, summary, or runtime guidance that should remain visible in transcript state.
_Avoid_: Hidden prompt-only instruction

**AssistantMessage**:
A **Message** variant representing model or agent output, including optional thinking and tool calls.
_Avoid_: Generic message with sidecar tool-call data

**ToolResultMessage**:
A **Message** variant representing the result of a tool call.
_Avoid_: Generic message with hidden tool result metadata

**Message Sidecar**:
Rare escape-hatch metadata attached to a **Message** when the structure is not stable enough to become a named field or type.
_Avoid_: Primary message field, common protocol field

## Relationships

- A **Message** may be projected into a **ModelMessage** before a model call.
- A **Provider Adapter** may translate a **ModelMessage** differently for each model provider.
- A **Message** remains in runtime state; a **ModelMessage** belongs at the provider boundary.
- **Message** should be a role-specific union of message variants rather than one wide dataclass with many optional fields.
- **SystemMessage** belongs in the same **Message** union so instruction, summary, and runtime guidance can remain inspectable in state and trace.
- **ModelMessage** should also be a role-specific union mirroring **Message** roles, but without runtime routing fields.
- Request-level system prompt is for static agent or run instructions. **SystemMessage** is for dynamic transcript-visible instruction, summary, lesson, or runtime guidance.
- A **ModelMessage** should use **Content Blocks** so multimodal providers can share one project-owned boundary.
- The first **Content Block** set is text, image, thinking, and tool call. File, audio, and video blocks wait for concrete use cases.
- **Content Blocks** and **ModelMessage** should be frozen dataclasses so provider adapters convert immutable project-owned values into provider wire payloads explicitly.
- Shared **Message** should also be a frozen dataclass; recorded transcript history should not be mutated in place.
- **Message** content may be text or text/image content blocks. Thinking and tool-call blocks belong to the model boundary or explicit tool-call fields, not ordinary runtime content.
- **ImageBlock** uses normalized base64 image data plus MIME type (`data`, `mime_type`), following pi-mono's boundary. `data` is a plain base64 string without a data-URL prefix. URL, file path, or provider file-id inputs must be resolved before becoming an **ImageBlock**.
- **ThinkingBlock** is important model output and should be preserved as structured data for trace, replay, and provider continuity. Whether it is shown to users is a separate visibility decision.
- Assistant **Message** values preserve thinking in an explicit `thinking` field, not in `content` or **Message Sidecar**.
- **ThinkingBlock** fields are `text`, `signature`, and `redacted`. The signature is provider-continuity metadata; provider identity belongs outside the block.
- A **Message Type** should have one project-owned source of truth; example folders may demonstrate runtime behavior but should not become the long-term source of message protocol definitions.
- Simple Agent Lab should align with pi-mono's message boundary semantics without copying its names: this project's **Message** corresponds to pi-mono's `AgentMessage`, and this project's **ModelMessage** corresponds to pi-mono / pi-ai's LLM `Message`.
- Custom runtime semantics should be represented on **Message** through open `kind`, `channel`, and structured `data`, then filtered or projected at the **ModelMessage** boundary.
- Frequently used message semantics should become explicit fields or named project-owned types. **Message Sidecar** is only for uncommon or still-experimental metadata.
- Tool calls and tool-result identity are common protocol concepts, so they belong on explicit **Message** fields rather than in **Message Sidecar**.
- **ToolCallBlock** is the single project-owned tool-call shape. Assistant **Message** values and assistant **ModelMessage** values reuse it instead of carrying separate dict-shaped tool-call records.
- **ToolCallBlock.arguments** is typed as `Mapping[str, Any]`: callers should not mutate it in place, while adapters may shallow-copy it to provider wire dictionaries.
- Each **Message Type** should have a documented field catalog plus named construction, projection, or validation helpers where plain construction would be ambiguous.
- **Message Type** behavior should stay in module-level construction, projection, validation, and helper functions.
- First shared **Message** constructor helpers should be `user_message(...)`, `system_message(...)`, `assistant_message(...)`, and `tool_result_message(...)`.
- First shared **ModelMessage** constructor helpers should be `model_user_message(...)`, `model_assistant_message(...)`, and `model_tool_result_message(...)`.
- The internal provider-neutral tool-result role is `tool_result`. Provider adapters may translate it to provider wire roles such as OpenAI's `tool`.
- `role` is model-facing and `kind` is runtime-facing. A tool-result message normally has both `role="tool_result"` and `kind="tool_result"`, but projection code should not use `kind` as a provider role.

## Example Dialogue

> **Dev:** "Can we add OpenAI-specific image fields to the message?"
> **Domain expert:** "Add provider-specific fields at the **Provider Adapter** boundary. Keep the **Message** provider-neutral and project it into **ModelMessage** values with **Content Blocks** that each provider adapter can translate."

## Flagged Ambiguities

- "Message" was used to mean both runtime transcript data and provider-facing model payload. Resolved: use **Message** for the runtime transcript value and **ModelMessage** for the provider-facing payload.
- Message type ownership was ambiguous because design-version examples define their own message shapes. Resolved direction: future project-owned message types should live under `src/simple_agent_lab`.
- "AgentMessage" was considered to mirror pi-mono naming. Resolved: use pi-mono's abstraction boundary but keep this project's simpler names **Message** and **ModelMessage**.
- Python does not need pi-mono's TypeScript declaration-merging pattern. Resolved: custom transcript semantics use **Message** fields (`kind`, `channel`, `data`) rather than a `CustomMessage` union.
- Mutable recorded messages were considered for future streaming. Resolved: shared **Message** stays frozen; streaming should use update events or replace an unrecorded temporary value rather than mutating transcript history.
- Dropping model thinking was considered to keep transcript messages simple. Resolved: preserve **ThinkingBlock** as structured model output; decide display separately.
- `data` was considered as the general home for custom message semantics. Resolved: use it only as **Message Sidecar**; common protocol concepts need explicit fields or named types.
- Method-heavy dataclasses were considered for message behavior. Resolved: keep dataclasses value-like and use module-level helpers.
- A single wide **Message** dataclass was considered for explicit tool/thinking fields. Resolved: use role-specific variants such as **UserMessage**, **AssistantMessage**, and **ToolResultMessage** so each type has a clear field catalog.
- A `Message` namespace class with `Message.user(...)` factories was considered. Resolved: keep `Message` as a type alias union and use module-level constructor helpers.
- Keeping system prompts outside transcript messages was considered. Resolved: include **SystemMessage** in the shared **Message** union so system, instruction, and summary entries can be inspected like other runtime context.
- A wide **ModelMessage** dataclass was considered. Resolved: mirror **Message** with role-specific model-message variants at the provider boundary.
