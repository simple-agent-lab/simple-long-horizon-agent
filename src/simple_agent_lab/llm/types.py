"""Wire-format types for the unified LLM access layer.

These are *provider-agnostic*. Each agent loop (01 / 02 / 03) has its own
`Message` with routing fields (sender, target, kind, channel); none of
those reach a provider. The boundary is `to_llm_message(...)` in each
agent loop, producing the types in this file.

All types are frozen dataclasses — they represent immutable values that
flow through the protocol. The mutable part (streaming accumulation,
state) lives in the agent loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

if TYPE_CHECKING:
    from .provider import Provider


Role = Literal["system", "user", "assistant", "tool_result"]
StopReason = Literal["end_turn", "tool_use", "max_tokens", "error"]


@dataclass(frozen=True)
class ContentBlock:
    """One block inside a message's content. Text, image, or thinking.

    Multimodal content is a list of these. Plain-text content can use a
    bare `str` on `LLMMessage.content` to skip the block ceremony.
    """
    kind: Literal["text", "image", "thinking"] = "text"
    text: str = ""
    data: str = ""
    mime_type: str = ""
    thinking: str = ""
    signature: Optional[str] = None
    redacted: bool = False


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation emitted by the model."""
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


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
    """
    role: Role
    content: Union[str, list[ContentBlock]] = ""
    tool_call_id: Optional[str] = None         # only set when role="tool_result"
    tool_calls: Optional[list[ToolCall]] = None  # only set when role="assistant"
    name: Optional[str] = None                 # speaker label (some APIs use it)
    cache_breakpoint: bool = False


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
class Usage:
    """Token-usage breakdown for one call. All fields default to 0."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


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
    """Drained result of a stream — what `complete()` returns."""
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str = ""
    stop_reason: StopReason = "end_turn"
    usage: Usage = field(default_factory=Usage)
    raw: dict[str, Any] = field(default_factory=dict)   # original provider response


@dataclass(frozen=True)
class StreamEvent:
    """One event in the LLM's output stream. Discriminated by `kind`.

    Payload contracts (by `kind`):
      - "text_delta":         {"delta": str}
      - "thinking_delta":     {"delta": str}
      - "tool_call_start":    {"tool_call": ToolCall}    # args may be empty
      - "tool_call_delta":    {"tool_call_id": str, "arguments_json_delta": str}
      - "tool_call_complete": {"tool_call": ToolCall}    # args fully parsed
      - "usage_update":       {"usage": Usage}
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


# `LLMRequest.provider: "Provider"` is a forward reference. With
# `from __future__ import annotations` (top of file), all annotations
# are evaluated lazily, so we don't import Provider at runtime here.
