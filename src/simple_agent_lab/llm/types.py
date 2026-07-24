"""Wire-format types for the unified LLM access layer.

These are *provider-agnostic*. Each agent loop has its own `Message`
with routing fields (sender, target, kind); none of those reach
a provider. The boundary is `message_to_llm_message(...)` (see bridge.py),
producing the types in this file.

All types are frozen dataclasses — they represent immutable values that
flow through the protocol. The mutable part (streaming accumulation,
state) lives in the agent loop.

Content model and token-usage shape are shared with the runtime layer:
`LLMMessage` and `LLMResponse` carry `tuple[ContentBlock, ...]` (union
of `TextBlock` / `ImageBlock` / `ThinkingBlock` / `ToolCallBlock`) and
`TokenUsage` exactly as the runtime sees them. There are no separate
LLM-layer block or usage types.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from ..messages import (
    ContentBlock,
    ContentInput,
    MessageContent,
    Role,
    TextBlock,
    ThinkingBlock,
    TokenUsage,
    ToolCallBlock,
    ToolResultBlock,
    normalize_content,
    text_of,
    thinking_blocks_of,
    tool_calls_of,
    tool_results_of,
)

if TYPE_CHECKING:
    from .provider import Provider, ReasoningEffort


StopReason = Literal["end_turn", "tool_use", "max_tokens", "error"]

# When a model emits tool-call arguments that aren't valid JSON, adapters can't
# build a real argument dict. Rather than raise (which would abort the run on a
# self-correctable model slip), they stash the raw string under this single key.
# It is the agent-visible signal that "the model produced a malformed tool call",
# which `simple_agent_lab.llm.retry` looks for to re-ask the model.
RAW_ARGUMENTS_KEY = "_raw_arguments"


@dataclass(frozen=True)
class LLMMessage:
    """Provider-agnostic chat message. No agent-routing fields.

    `role` is `"system" | "user" | "assistant"`. Tool results ride
    inside a user message as ``ToolResultBlock`` content blocks (matching
    Anthropic's wire shape); adapters that need the OpenAI "one
    role=tool entry per result" shape split the bundle on the way out.

    `content` carries every block the message emits — text, image,
    thinking, tool_call, tool_result — in the order produced.

    `extra` is the per-message twin of `LLMRequest.extra`: a namespaced
    bag of provider-specific instructions on this particular message
    (e.g. `extra["anthropic.cache_breakpoint"] = True` to mark this
    message as a prompt-caching anchor on Anthropic). Adapters read
    only their own namespace; unknown keys are silently ignored so a
    transcript stays portable across providers. Trace renders `extra`
    so the standardized view shows these instructions; `--raw` on the
    trace shows the translated shape an adapter produced.
    """

    role: Role
    content: MessageContent = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Accept str / list at construction for ergonomics; canonicalize to
        # a tuple of blocks so downstream readers can assume the typed shape.
        if isinstance(self.content, (str, list)):
            object.__setattr__(self, "content", normalize_content(self.content))

    @property
    def thinking_blocks(self) -> tuple[ThinkingBlock, ...]:
        return thinking_blocks_of(self.content)

    @property
    def tool_calls(self) -> tuple[ToolCallBlock, ...]:
        return tool_calls_of(self.content)

    @property
    def tool_results(self) -> tuple[ToolResultBlock, ...]:
        return tool_results_of(self.content)


def llm_message(
    role: Role,
    content: ContentInput = "",
    *,
    extra: Mapping[str, Any] | None = None,
) -> LLMMessage:
    """Factory that accepts a str shorthand or a block sequence."""
    return LLMMessage(
        role=role,
        content=normalize_content(content),
        extra=dict(extra or {}),
    )


@dataclass(frozen=True)
class LLMTool:
    """Wire-format tool definition. No `execute` callable.

    `simple_agent_lab.tools.Tool` / `AgentTool` projects to this via a
    `to_llm_tool()` method. The execute side stays local to the agent loop.
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


@dataclass(frozen=True)
class LLMRequest:
    """One call to a provider. Pure data; safe to log or replay."""

    provider: "Provider"
    messages: list[LLMMessage]
    tools: list[LLMTool] = field(default_factory=list)
    system_prompt: str | None = None
    temperature: float | None = None  # None → use provider.default_temperature
    max_tokens: int | None = None
    # Provider-agnostic reasoning depth; adapters translate it to each
    # endpoint's wire shape. None → fall back to provider.default_reasoning.
    reasoning: ReasoningEffort | None = None
    # Generous ceiling: one high-reasoning call can take minutes, and a
    # too-tight timeout aborts a long run mid-flight.
    timeout_seconds: float | None = 600.0
    extra: dict[str, Any] = field(
        default_factory=dict
    )  # provider-specific request options


@dataclass(frozen=True)
class LLMResponse:
    """Drained result of a stream.

    `content` is the source of truth: blocks in the order the model
    produced them, so callers can replay the exact shape on the next turn.
    `text` / `thinking_blocks` / `tool_calls` are derived views over
    `content`.

    `raw` is the verbatim provider-call snapshot: ``{"request": ...,
    "response": ...}``. Empty when the adapter doesn't make a real
    call (e.g. fake). Serves two needs:

      * Debugging — `print_trace(state, raw=True)` shows what actually
        crossed the wire (which fields landed in the request body,
        which fields the endpoint returned that we abstracted away).
      * Programmatic access — applications can pull provider-specific
        response fields the standardized layer doesn't surface
        (`refusal`, `prompt_tokens_details.cached_tokens`,
        `safety_ratings`, etc.) directly off `raw["response"]`.

    The `request` half is the full outbound request body — model,
    tools, temperature, system, the `messages` / `input` history, and
    our outbound `extra` translations — so "did our cache_control /
    reasoning_content land?" and "what exactly did the model see?" are
    both answerable from `raw["request"]` alone. It is the single ground
    truth of what crossed the model boundary. The trace writer keeps it
    out of the main trajectory file by externalizing it (deduped) to the
    sibling `*.raw.jsonl` pool rather than embedding the growing history
    inline, which would pin O(N²) size across a long session. See ADR
    trajectory-schema-v5.
    """

    content: MessageContent = ()
    stop_reason: StopReason = "end_turn"
    usage: TokenUsage = field(default_factory=TokenUsage)
    # The model that served this response: adapters set it from the
    # provider's echoed `response.model` (which resolves aliases to dated
    # snapshots). `complete()` back-fills the requested `provider.model`
    # only if an adapter left it blank, so every drained response carries a
    # model id for downstream cost/routing analysis.
    model: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return text_of(self.content)

    @property
    def thinking_blocks(self) -> tuple[ThinkingBlock, ...]:
        return thinking_blocks_of(self.content)

    @property
    def tool_calls(self) -> tuple[ToolCallBlock, ...]:
        return tool_calls_of(self.content)


@dataclass(frozen=True)
class StreamEvent:
    """One event in the LLM's output stream. Discriminated by `kind`.

    Payload contracts (by `kind`):
      - "text_delta":         {"delta": str}
      - "thinking_delta":     {"delta": str}
      - "tool_call_start":    {"tool_call": ToolCallBlock}    # args may be empty
      - "tool_call_delta":    {"tool_call_id": str, "arguments_json_delta": str}
      - "tool_call_complete": {"tool_call": ToolCallBlock}    # args fully parsed
      - "usage_update":       {"usage": TokenUsage}
      - "done":               {"response": LLMResponse}

    The `done` event always fires last and carries the final drained
    response. `complete()` waits for it; streaming consumers can stop
    on it.
    """

    kind: Literal[
        "text_delta",
        "thinking_delta",
        "tool_call_start",
        "tool_call_delta",
        "tool_call_complete",
        "usage_update",
        "done",
    ]
    payload: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "RAW_ARGUMENTS_KEY",
    "ContentBlock",
    "LLMMessage",
    "LLMRequest",
    "LLMResponse",
    "LLMTool",
    "MessageContent",
    "Role",
    "StopReason",
    "StreamEvent",
    "TextBlock",
    "ThinkingBlock",
    "TokenUsage",
    "ToolCallBlock",
    "ToolResultBlock",
    "llm_message",
]
