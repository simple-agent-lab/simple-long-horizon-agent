"""Wire-format types for the unified LLM access layer.

These are *provider-agnostic*. Each agent loop has its own `Message`
with routing fields (sender, target, kind, channel); none of those reach
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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional

from ..messages import (
    ContentBlock,
    ContentInput,
    MessageContent,
    Role,
    TextBlock,
    ThinkingBlock,
    TokenUsage,
    ToolCallBlock,
    normalize_content,
    text_of,
    thinking_blocks_of,
    tool_calls_of,
)

if TYPE_CHECKING:
    from .provider import Provider


StopReason = Literal["end_turn", "tool_use", "max_tokens", "error"]


@dataclass(frozen=True)
class LLMMessage:
    """Provider-agnostic chat message. No agent-routing fields.

    Roles are intentionally **internal**, not wire-format. In particular
    `role="tool_result"` is our neutral name for a tool-call result; each
    provider adapter translates at its own boundary:
      - OpenAI Chat / Responses → `role="tool"` + `tool_call_id`
      - Anthropic Messages      → `role="user"` + content block of
                                  type `"tool_result"`
    Keeping the internal name distinct from any provider's wire role
    means the same transcript can be sent to either provider without
    a translation step at the call site.

    `cache_breakpoint` marks this message as a prompt-caching anchor.
    Adapters that support caching (Anthropic) translate it to the wire
    format; adapters that don't (OpenAI Chat) ignore it. The decision
    of *where* to place breakpoints is left to the caller — the layer
    does not auto-place them.

    `content` carries every block the message emits — text, image,
    thinking, and tool_call — in the order the model produced them.
    Adapters split blocks into the right wire slot per provider.
    """
    role: Role
    content: MessageContent = ()
    tool_call_id: Optional[str] = None         # only set when role="tool_result"
    name: Optional[str] = None                 # speaker label (some APIs use it)
    cache_breakpoint: bool = False

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


def llm_message(
    role: Role,
    content: ContentInput = "",
    *,
    tool_call_id: Optional[str] = None,
    name: Optional[str] = None,
    cache_breakpoint: bool = False,
) -> LLMMessage:
    """Factory that accepts a str shorthand or a block sequence."""
    return LLMMessage(
        role=role,
        content=normalize_content(content),
        tool_call_id=tool_call_id,
        name=name,
        cache_breakpoint=cache_breakpoint,
    )


@dataclass(frozen=True)
class LLMTool:
    """Wire-format tool definition. No `execute` callable.

    `simple_agent_lab.tools.Tool` / `AgentTool` projects to this via a
    `to_llm_tool()` method. The execute side stays local to the agent loop.
    """
    name: str
    description: str
    parameters: dict[str, Any]   # JSON Schema


@dataclass(frozen=True)
class LLMRequest:
    """One call to a provider. Pure data; safe to log or replay."""
    provider: "Provider"
    messages: list[LLMMessage]
    tools: list[LLMTool] = field(default_factory=list)
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None        # None → use provider.default_temperature
    max_tokens: Optional[int] = None
    timeout_seconds: Optional[float] = 60.0
    extra: dict[str, Any] = field(default_factory=dict)  # provider-specific request options


@dataclass(frozen=True)
class LLMResponse:
    """Drained result of a stream.

    `content` is the source of truth: blocks in the order the model
    produced them, so callers can replay the exact shape on the next turn.
    `text` / `thinking` / `thinking_blocks` / `tool_calls` are derived
    views over `content`.

    `wire` is the verbatim HTTP-level snapshot the adapter captured:
    ``{"request": <kwargs sent to the SDK>, "response": <dumped SDK
    response>}``. Empty when the adapter doesn't make a real call
    (e.g. fake). It rides along to the runtime AssistantMessage via the
    bridge so `print_trace(state, wire=True)` can show the wire layer
    alongside the standardized content blocks.
    """
    content: MessageContent = ()
    stop_reason: StopReason = "end_turn"
    usage: TokenUsage = field(default_factory=TokenUsage)
    raw: dict[str, Any] = field(default_factory=dict)   # adapter-extracted metadata
    wire: dict[str, Any] = field(default_factory=dict)  # raw {request, response} dump

    @property
    def text(self) -> str:
        return text_of(self.content)

    @property
    def thinking(self) -> str:
        return "\n\n".join(
            block.text for block in thinking_blocks_of(self.content) if block.text
        )

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


# Public alias for the legacy `ToolCall` name (now == ToolCallBlock).
ToolCall = ToolCallBlock


__all__ = [
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
    "ToolCall",
    "ToolCallBlock",
    "llm_message",
]
