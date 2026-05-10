"""Shared message protocol for Simple Agent Lab.

`Message` is the runtime transcript union. It keeps routing fields such as
sender, target, kind, and channel.

`ModelMessage` is the provider-neutral model-call union. It strips runtime
routing fields and is the input shape provider adapters translate to wire
payloads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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


MessageContentBlock: TypeAlias = TextBlock | ImageBlock
MessageContent: TypeAlias = str | tuple[MessageContentBlock, ...]
ModelContentBlock: TypeAlias = TextBlock | ImageBlock | ThinkingBlock | ToolCallBlock
ModelContent: TypeAlias = tuple[ModelContentBlock, ...]
Sidecar: TypeAlias = Mapping[str, Any]


@dataclass(frozen=True)
class UserMessage:
    role: Literal["user"] = field(default="user", init=False)
    content: MessageContent = ""
    sender: AgentName = "user"
    target: AgentName = "all"
    kind: MessageKind = "message"
    channel: MessageChannel = "main"
    data: Sidecar = field(default_factory=dict)


@dataclass(frozen=True)
class SystemMessage:
    role: Literal["system"] = field(default="system", init=False)
    content: MessageContent = ""
    sender: AgentName = "system"
    target: AgentName = "all"
    kind: MessageKind = "system"
    channel: MessageChannel = "main"
    data: Sidecar = field(default_factory=dict)


@dataclass(frozen=True)
class AssistantMessage:
    role: Literal["assistant"] = field(default="assistant", init=False)
    content: MessageContent = ""
    sender: AgentName = "assistant"
    target: AgentName = "all"
    kind: MessageKind = "message"
    channel: MessageChannel = "main"
    thinking: tuple[ThinkingBlock, ...] = ()
    tool_calls: tuple[ToolCallBlock, ...] = ()
    usage: TokenUsage | None = None
    data: Sidecar = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResultMessage:
    role: Literal["tool_result"] = field(default="tool_result", init=False)
    content: MessageContent = ""
    tool_call_id: str = ""
    tool_name: str = ""
    sender: AgentName = ""
    target: AgentName = ""
    kind: MessageKind = "tool_result"
    channel: MessageChannel = "main"
    is_error: bool = False
    data: Sidecar = field(default_factory=dict)


Message: TypeAlias = UserMessage | SystemMessage | AssistantMessage | ToolResultMessage


@dataclass(frozen=True)
class ModelUserMessage:
    role: Literal["user"] = field(default="user", init=False)
    content: tuple[TextBlock | ImageBlock, ...] = ()


@dataclass(frozen=True)
class ModelSystemMessage:
    role: Literal["system"] = field(default="system", init=False)
    content: tuple[TextBlock | ImageBlock, ...] = ()


@dataclass(frozen=True)
class ModelAssistantMessage:
    role: Literal["assistant"] = field(default="assistant", init=False)
    content: ModelContent = ()


@dataclass(frozen=True)
class ModelToolResultMessage:
    role: Literal["tool_result"] = field(default="tool_result", init=False)
    content: tuple[TextBlock | ImageBlock, ...] = ()
    tool_call_id: str = ""
    tool_name: str = ""
    is_error: bool = False


ModelMessage: TypeAlias = (
    ModelUserMessage | ModelSystemMessage | ModelAssistantMessage | ModelToolResultMessage
)


def user_message(
    content: MessageContent = "",
    *,
    sender: AgentName = "user",
    target: AgentName = "all",
    kind: MessageKind = "message",
    channel: MessageChannel = "main",
    data: Sidecar | None = None,
) -> UserMessage:
    return UserMessage(
        content=_normalize_message_content(content),
        sender=sender,
        target=target,
        kind=kind,
        channel=channel,
        data=dict(data or {}),
    )


def system_message(
    content: MessageContent = "",
    *,
    sender: AgentName = "system",
    target: AgentName = "all",
    kind: MessageKind = "system",
    channel: MessageChannel = "main",
    data: Sidecar | None = None,
) -> SystemMessage:
    return SystemMessage(
        content=_normalize_message_content(content),
        sender=sender,
        target=target,
        kind=kind,
        channel=channel,
        data=dict(data or {}),
    )


def assistant_message(
    content: MessageContent = "",
    *,
    sender: AgentName = "assistant",
    target: AgentName = "all",
    kind: MessageKind = "message",
    channel: MessageChannel = "main",
    thinking: Sequence[ThinkingBlock] = (),
    tool_calls: Sequence[ToolCallBlock] = (),
    usage: TokenUsage | None = None,
    data: Sidecar | None = None,
) -> AssistantMessage:
    message = AssistantMessage(
        content=_normalize_message_content(content),
        sender=sender,
        target=target,
        kind=kind,
        channel=channel,
        thinking=tuple(thinking),
        tool_calls=tuple(tool_calls),
        usage=usage,
        data=dict(data or {}),
    )
    validate_message(message)
    return message


def tool_result_message(
    content: MessageContent = "",
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
        content=_normalize_message_content(content),
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


def model_user_message(content: MessageContent = "") -> ModelUserMessage:
    return ModelUserMessage(content=_normalize_visible_blocks(content))


def model_system_message(content: MessageContent = "") -> ModelSystemMessage:
    return ModelSystemMessage(content=_normalize_visible_blocks(content))


def model_assistant_message(
    content: MessageContent | ModelContent = "",
    *,
    thinking: Sequence[ThinkingBlock] = (),
    tool_calls: Sequence[ToolCallBlock] = (),
) -> ModelAssistantMessage:
    message = ModelAssistantMessage(
        content=(
            *_normalize_model_blocks(content),
            *tuple(thinking),
            *tuple(tool_calls),
        )
    )
    validate_model_message(message)
    return message


def model_tool_result_message(
    content: MessageContent = "",
    *,
    tool_call_id: str,
    tool_name: str,
    is_error: bool = False,
) -> ModelToolResultMessage:
    message = ModelToolResultMessage(
        content=_normalize_visible_blocks(content),
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        is_error=is_error,
    )
    validate_model_message(message)
    return message


def to_model_message(message: Message, *, with_header: bool = True) -> ModelMessage:
    """Project one runtime Message to one provider-neutral ModelMessage."""
    header = _routing_header(message) if with_header else ""
    if isinstance(message, UserMessage):
        return model_user_message(_content_with_header(message.content, header))
    if isinstance(message, SystemMessage):
        return model_system_message(_content_with_header(message.content, header))
    if isinstance(message, AssistantMessage):
        return model_assistant_message(
            _content_with_header(message.content, header),
            thinking=message.thinking,
            tool_calls=message.tool_calls,
        )
    if isinstance(message, ToolResultMessage):
        return model_tool_result_message(
            message.content,
            tool_call_id=message.tool_call_id,
            tool_name=message.tool_name,
            is_error=message.is_error,
        )
    raise TypeError(f"Unexpected message type: {type(message)!r}")


def to_model_messages(
    messages: Sequence[Message],
    *,
    with_header: bool = True,
    skip_kinds: set[str] | None = None,
) -> list[ModelMessage]:
    skipped = skip_kinds if skip_kinds is not None else {"notification", "trace"}
    return [
        to_model_message(message, with_header=with_header)
        for message in messages
        if message.kind not in skipped
    ]


def message_tool_calls(message: Message) -> tuple[ToolCallBlock, ...]:
    if isinstance(message, AssistantMessage):
        return message.tool_calls
    return ()


def message_text(message: Message) -> str:
    content = message.content
    if isinstance(content, str) and content:
        return content.replace("\n", " ")[:120]
    if isinstance(content, tuple) and content:
        texts = [block.text for block in content if isinstance(block, TextBlock)]
        if texts:
            return " ".join(texts).replace("\n", " ")[:120]
        return str(content)[:120]
    if message.data:
        return str(dict(message.data))[:120]
    return ""


def model_message_text(message: ModelMessage) -> str:
    texts = [block.text for block in message.content if isinstance(block, TextBlock)]
    return " ".join(texts)


def validate_message(message: Message) -> None:
    if isinstance(message, AssistantMessage):
        for tool_call in message.tool_calls:
            validate_tool_call(tool_call)
    if isinstance(message, ToolResultMessage):
        if not message.tool_call_id:
            raise ValueError("ToolResultMessage.tool_call_id must be non-empty")
        if not message.tool_name:
            raise ValueError("ToolResultMessage.tool_name must be non-empty")


def validate_model_message(message: ModelMessage) -> None:
    if isinstance(message, ModelAssistantMessage):
        for block in message.content:
            if isinstance(block, ToolCallBlock):
                validate_tool_call(block)
    if isinstance(message, ModelToolResultMessage):
        if not message.tool_call_id:
            raise ValueError("ModelToolResultMessage.tool_call_id must be non-empty")
        if not message.tool_name:
            raise ValueError("ModelToolResultMessage.tool_name must be non-empty")


def validate_tool_call(tool_call: ToolCallBlock) -> None:
    if not tool_call.id:
        raise ValueError("ToolCallBlock.id must be non-empty")
    if not tool_call.name:
        raise ValueError("ToolCallBlock.name must be non-empty")


def _normalize_message_content(content: MessageContent) -> MessageContent:
    if isinstance(content, str):
        return content
    return _normalize_visible_blocks(content)


def _normalize_visible_blocks(content: MessageContent) -> tuple[TextBlock | ImageBlock, ...]:
    if isinstance(content, str):
        return (TextBlock(content),) if content else ()
    blocks: list[TextBlock | ImageBlock] = []
    for block in content:
        if not isinstance(block, TextBlock | ImageBlock):
            raise TypeError(f"Expected TextBlock or ImageBlock, got {type(block)!r}")
        blocks.append(block)
    return tuple(blocks)


def _normalize_model_blocks(content: MessageContent | ModelContent) -> ModelContent:
    if isinstance(content, str):
        return (TextBlock(content),) if content else ()
    blocks: list[ModelContentBlock] = []
    for block in content:
        if not isinstance(block, TextBlock | ImageBlock | ThinkingBlock | ToolCallBlock):
            raise TypeError(f"Unexpected model content block: {type(block)!r}")
        blocks.append(block)
    return tuple(blocks)


def _routing_header(message: Message) -> str:
    has_meta = bool(message.sender or message.target) or message.kind != "message"
    if not has_meta or isinstance(message, ToolResultMessage):
        return ""
    return f"[{message.sender} -> {message.target} | {message.kind}/{message.channel}]"


def _content_with_header(content: MessageContent, header: str) -> MessageContent:
    if not header:
        return content
    if isinstance(content, str):
        return f"{header}\n{content}" if content else header
    return (TextBlock(header), *content)
