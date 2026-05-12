"""Shared message protocol for Simple Agent Lab.

`Message` is the runtime transcript union — every assistant turn, user
input, system prompt, or tool result lives as one of these four
dataclasses. Routing fields (sender, target, kind, channel) stay on the
runtime side; adapters never see them.

The content model is unified across every layer:

    ContentBlock = TextBlock | ImageBlock | ThinkingBlock | ToolCallBlock

so `message.content` is always `tuple[ContentBlock, ...]`. There are no
sibling thinking / tool_calls fields on the message; `AssistantMessage`
exposes those as derived `@property` views over `content`.

Projecting a runtime `Message` into the provider-facing `LLMMessage`
happens in one step inside `simple_agent_lab.llm.bridge`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias


Role: TypeAlias = Literal["system", "user", "assistant", "tool_result"]
MessageRole: TypeAlias = Role
MessageKind: TypeAlias = str
MessageChannel: TypeAlias = str
AgentName: TypeAlias = str


@dataclass(frozen=True)
class TextBlock:
    text: str

    kind: Literal["text"] = field(default="text", init=False)


@dataclass(frozen=True)
class ImageBlock:
    data: str
    mime_type: str

    kind: Literal["image"] = field(default="image", init=False)


@dataclass(frozen=True)
class ThinkingBlock:
    text: str = ""
    signature: str | None = None
    redacted: bool = False

    kind: Literal["thinking"] = field(default="thinking", init=False)


@dataclass(frozen=True)
class ToolCallBlock:
    id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    kind: Literal["tool_call"] = field(default="tool_call", init=False)


@dataclass(frozen=True)
class TokenUsage:
    """Provider-reported token counts for one model call.

    Attached to the AssistantMessage produced from that call. Providers report
    per-call totals: `input_tokens` covers ALL inputs the call saw (system +
    history + tool results), while `output_tokens` is just this assistant
    message. We therefore only persist usage on AssistantMessage — splitting
    the input total across earlier messages would require re-tokenizing.

    `output_tokens` doubles as a precise estimate of how many tokens this
    message will cost when re-sent in a future call's context, so context_view
    prefers it over char-based estimation when present.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


ContentBlock: TypeAlias = TextBlock | ImageBlock | ThinkingBlock | ToolCallBlock
MessageContent: TypeAlias = tuple[ContentBlock, ...]
ContentInput: TypeAlias = str | Sequence[ContentBlock]
Sidecar: TypeAlias = Mapping[str, Any]


@dataclass(frozen=True)
class UserMessage:
    role: Literal["user"] = field(default="user", init=False)
    content: MessageContent = ()
    sender: AgentName = "user"
    target: AgentName = "all"
    kind: MessageKind = "message"
    channel: MessageChannel = "main"
    data: Sidecar = field(default_factory=dict)


@dataclass(frozen=True)
class SystemMessage:
    role: Literal["system"] = field(default="system", init=False)
    content: MessageContent = ()
    sender: AgentName = "system"
    target: AgentName = "all"
    kind: MessageKind = "system"
    channel: MessageChannel = "main"
    data: Sidecar = field(default_factory=dict)


@dataclass(frozen=True)
class AssistantMessage:
    role: Literal["assistant"] = field(default="assistant", init=False)
    content: MessageContent = ()
    sender: AgentName = "assistant"
    target: AgentName = "all"
    kind: MessageKind = "message"
    channel: MessageChannel = "main"
    usage: TokenUsage | None = None
    data: Sidecar = field(default_factory=dict)

    @property
    def thinking(self) -> tuple[ThinkingBlock, ...]:
        return tuple(block for block in self.content if isinstance(block, ThinkingBlock))

    @property
    def tool_calls(self) -> tuple[ToolCallBlock, ...]:
        return tuple(block for block in self.content if isinstance(block, ToolCallBlock))


@dataclass(frozen=True)
class ToolResultMessage:
    role: Literal["tool_result"] = field(default="tool_result", init=False)
    content: MessageContent = ()
    tool_call_id: str = ""
    tool_name: str = ""
    sender: AgentName = ""
    target: AgentName = ""
    kind: MessageKind = "tool_result"
    channel: MessageChannel = "main"
    is_error: bool = False
    data: Sidecar = field(default_factory=dict)


Message: TypeAlias = UserMessage | SystemMessage | AssistantMessage | ToolResultMessage


def user_message(
    content: ContentInput = "",
    *,
    sender: AgentName = "user",
    target: AgentName = "all",
    kind: MessageKind = "message",
    channel: MessageChannel = "main",
    data: Sidecar | None = None,
) -> UserMessage:
    return UserMessage(
        content=normalize_content(content),
        sender=sender,
        target=target,
        kind=kind,
        channel=channel,
        data=dict(data or {}),
    )


def system_message(
    content: ContentInput = "",
    *,
    sender: AgentName = "system",
    target: AgentName = "all",
    kind: MessageKind = "system",
    channel: MessageChannel = "main",
    data: Sidecar | None = None,
) -> SystemMessage:
    return SystemMessage(
        content=normalize_content(content),
        sender=sender,
        target=target,
        kind=kind,
        channel=channel,
        data=dict(data or {}),
    )


def assistant_message(
    content: ContentInput = "",
    *,
    sender: AgentName = "assistant",
    target: AgentName = "all",
    kind: MessageKind = "message",
    channel: MessageChannel = "main",
    usage: TokenUsage | None = None,
    data: Sidecar | None = None,
) -> AssistantMessage:
    message = AssistantMessage(
        content=normalize_content(content),
        sender=sender,
        target=target,
        kind=kind,
        channel=channel,
        usage=usage,
        data=dict(data or {}),
    )
    validate_message(message)
    return message


def tool_result_message(
    content: ContentInput = "",
    *,
    tool_call_id: str,
    tool_name: str,
    sender: AgentName | None = None,
    target: AgentName,
    kind: MessageKind = "tool_result",
    channel: MessageChannel = "main",
    is_error: bool = False,
    data: Sidecar | None = None,
) -> ToolResultMessage:
    message = ToolResultMessage(
        content=normalize_content(content),
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        sender=sender if sender is not None else tool_name,
        target=target,
        kind=kind,
        channel=channel,
        is_error=is_error,
        data=dict(data or {}),
    )
    validate_message(message)
    return message


def message_tool_calls(message: Message) -> tuple[ToolCallBlock, ...]:
    if isinstance(message, AssistantMessage):
        return message.tool_calls
    return ()


def thinking_blocks_of(content: Iterable[ContentBlock]) -> tuple[ThinkingBlock, ...]:
    return tuple(block for block in content if isinstance(block, ThinkingBlock))


def tool_calls_of(content: Iterable[ContentBlock]) -> tuple[ToolCallBlock, ...]:
    return tuple(block for block in content if isinstance(block, ToolCallBlock))


def text_of(content: Iterable[ContentBlock]) -> str:
    return "".join(block.text for block in content if isinstance(block, TextBlock))


def message_text(message: Message) -> str:
    text = text_of(message.content).replace("\n", " ").strip()
    if text:
        return text[:120]
    if message.data:
        return str(dict(message.data))[:120]
    return ""


def validate_message(message: Message) -> None:
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, ToolCallBlock):
                validate_tool_call(block)
    if isinstance(message, ToolResultMessage):
        if not message.tool_call_id:
            raise ValueError("ToolResultMessage.tool_call_id must be non-empty")
        if not message.tool_name:
            raise ValueError("ToolResultMessage.tool_name must be non-empty")


def validate_tool_call(tool_call: ToolCallBlock) -> None:
    if not tool_call.id:
        raise ValueError("ToolCallBlock.id must be non-empty")
    if not tool_call.name:
        raise ValueError("ToolCallBlock.name must be non-empty")


def normalize_content(content: ContentInput) -> MessageContent:
    """Coerce a str-or-sequence input to the canonical tuple-of-blocks form."""
    if isinstance(content, str):
        return (TextBlock(content),) if content else ()
    blocks: list[ContentBlock] = []
    for block in content:
        if not isinstance(block, (TextBlock, ImageBlock, ThinkingBlock, ToolCallBlock)):
            raise TypeError(f"Unexpected content block: {type(block)!r}")
        blocks.append(block)
    return tuple(blocks)
