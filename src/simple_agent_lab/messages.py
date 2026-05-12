"""Shared message protocol for Simple Agent Lab.

The protocol is built on **one** unified content model:

    TextBlock | ImageBlock | ThinkingBlock | ToolCallBlock | ToolResultBlock
    (== ContentBlock)

Every message carries the same shape ``content: tuple[ContentBlock, ...]``.

Tool results live as ``ToolResultBlock`` entries inside ``UserMessage.content``
(matching Anthropic's wire shape); a parallel-tool-call assistant turn returns
one user message bundling N ``ToolResultBlock`` blocks rather than N separate
messages. Per-block ``is_error`` lets a single bundle express partial failure
across parallel calls.

`Message` is the runtime transcript union. Routing fields (sender, target,
kind, channel) stay on the runtime side; adapters never see them.

Projecting a runtime `Message` into the provider-facing `LLMMessage` happens
in one step inside `simple_agent_lab.llm.bridge`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias


Role: TypeAlias = Literal["system", "user", "assistant"]
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
class ToolResultBlock:
    """One tool's return value, carried inside a user message.

    `content` is what the model sees next turn. It is a tuple of visible
    blocks (text and image) so multimodal tool results (screenshots,
    structured renders) ride through unchanged.

    `tool_call_id` links back to the assistant's `ToolCallBlock.id` that
    requested this work. `tool_name` is convenience metadata (some wire
    formats use it). `is_error` is per-block so a parallel-tool bundle
    can express partial failure.
    """
    tool_call_id: str
    tool_name: str
    content: tuple[TextBlock | ImageBlock, ...] = ()
    is_error: bool = False

    kind: Literal["tool_result"] = field(default="tool_result", init=False)


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


ContentBlock: TypeAlias = (
    TextBlock | ImageBlock | ThinkingBlock | ToolCallBlock | ToolResultBlock
)
VisibleBlock: TypeAlias = TextBlock | ImageBlock  # blocks legal inside a tool_result
MessageContent: TypeAlias = tuple[ContentBlock, ...]
ContentInput: TypeAlias = str | Sequence[ContentBlock]
ToolResultContentInput: TypeAlias = str | Sequence[VisibleBlock]
Sidecar: TypeAlias = Mapping[str, Any]


TOOL_RESULT_KIND = "tool_result"
TOOL_RESULT_SENDER = "tool"


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


Message: TypeAlias = UserMessage | SystemMessage | AssistantMessage


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
    content: ToolResultContentInput = "",
    *,
    tool_call_id: str,
    tool_name: str,
    target: AgentName,
    sender: AgentName | None = None,
    is_error: bool = False,
    kind: MessageKind = TOOL_RESULT_KIND,
    channel: MessageChannel = "main",
    data: Sidecar | None = None,
) -> UserMessage:
    """Build a UserMessage wrapping a single ToolResultBlock.

    Multi-result bundles (parallel tool calls) should use
    `tool_results_message(...)` so all results land in one message.
    """
    block = ToolResultBlock(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        content=_normalize_visible(content),
        is_error=is_error,
    )
    message = UserMessage(
        content=(block,),
        sender=sender if sender is not None else TOOL_RESULT_SENDER,
        target=target,
        kind=kind,
        channel=channel,
        data=dict(data or {}),
    )
    validate_message(message)
    return message


def tool_results_message(
    results: Sequence[ToolResultBlock],
    *,
    target: AgentName,
    sender: AgentName = TOOL_RESULT_SENDER,
    kind: MessageKind = TOOL_RESULT_KIND,
    channel: MessageChannel = "main",
    data: Sidecar | None = None,
) -> UserMessage:
    """Bundle N tool results from one assistant turn into a single message.

    This is the natural shape on Anthropic wire (one user message with N
    tool_result content blocks). Adapters that need the OpenAI shape (one
    `role="tool"` wire entry per result) split the bundle on the way out.
    """
    if not results:
        raise ValueError("tool_results_message requires at least one ToolResultBlock")
    message = UserMessage(
        content=tuple(results),
        sender=sender,
        target=target,
        kind=kind,
        channel=channel,
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


def tool_results_of(content: Iterable[ContentBlock]) -> tuple[ToolResultBlock, ...]:
    return tuple(block for block in content if isinstance(block, ToolResultBlock))


def text_of(content: Iterable[ContentBlock]) -> str:
    """Concatenate top-level TextBlock text.

    Tool result blocks are skipped here — their inner text is reachable
    via `tool_result_text(block)` so callers don't accidentally pull
    tool-result payloads into a user-facing message preview.
    """
    return "".join(block.text for block in content if isinstance(block, TextBlock))


def tool_result_text(block: ToolResultBlock) -> str:
    """Flatten the visible text inside a ToolResultBlock."""
    return "".join(b.text for b in block.content if isinstance(b, TextBlock))


def encode_image_data_url(mime: str, data: str) -> str:
    """Build a `data:<mime>;base64,<data>` URL from an ImageBlock's parts."""
    return f"data:{mime or 'image/png'};base64,{data}"


def decode_image_data_url(url: str) -> tuple[str, str] | None:
    """Split `data:<mime>;base64,<data>` into `(mime, base64)`; None when malformed."""
    if not url.startswith("data:"):
        return None
    head, _, payload = url.partition(",")
    if not payload or ";base64" not in head:
        return None
    mime = head[len("data:") :].split(";", 1)[0] or "image/png"
    return mime, payload


def message_text(message: Message) -> str:
    """One-line preview of the visible text in `message.content`.

    Tool-result bundles have no top-level TextBlock — their visible text
    lives inside each ToolResultBlock — so this helper falls through to
    the first inner result's text. Both runtime callers (trace, demo
    live event printer) and agent code that wants "what did the model
    see one line" then work uniformly across message types.
    """
    text = text_of(message.content).replace("\n", " ").strip()
    if text:
        return text[:120]
    for block in message.content:
        if isinstance(block, ToolResultBlock):
            inner = tool_result_text(block).replace("\n", " ").strip()
            if inner:
                return inner[:120]
    return ""


def is_tool_result_message(message: Message) -> bool:
    """True when the message is a user-message envelope of tool results."""
    return (
        isinstance(message, UserMessage)
        and message.kind == TOOL_RESULT_KIND
        and any(isinstance(block, ToolResultBlock) for block in message.content)
    )


def validate_message(message: Message) -> None:
    """Reject structurally illegal block placements and missing block fields.

    `ToolCallBlock` and `ThinkingBlock` only belong on `AssistantMessage`;
    `ToolResultBlock` only on `UserMessage`. The looseness of the shared
    `ContentBlock` union is OK at the type level, but the validator closes
    it at the runtime layer.
    """
    is_assistant = isinstance(message, AssistantMessage)
    is_user = isinstance(message, UserMessage)
    for block in message.content:
        if isinstance(block, ToolCallBlock):
            if not is_assistant:
                raise ValueError(
                    f"ToolCallBlock only belongs on AssistantMessage, "
                    f"found on {type(message).__name__}"
                )
            validate_tool_call(block)
        elif isinstance(block, ThinkingBlock):
            if not is_assistant:
                raise ValueError(
                    f"ThinkingBlock only belongs on AssistantMessage, "
                    f"found on {type(message).__name__}"
                )
        elif isinstance(block, ToolResultBlock):
            if not is_user:
                raise ValueError(
                    f"ToolResultBlock only belongs on UserMessage, "
                    f"found on {type(message).__name__}"
                )
            validate_tool_result(block)


def validate_tool_call(tool_call: ToolCallBlock) -> None:
    if not tool_call.id:
        raise ValueError("ToolCallBlock.id must be non-empty")
    if not tool_call.name:
        raise ValueError("ToolCallBlock.name must be non-empty")


def validate_tool_result(block: ToolResultBlock) -> None:
    if not block.tool_call_id:
        raise ValueError("ToolResultBlock.tool_call_id must be non-empty")
    if not block.tool_name:
        raise ValueError("ToolResultBlock.tool_name must be non-empty")


def normalize_content(content: ContentInput) -> MessageContent:
    """Coerce a str-or-sequence input to the canonical tuple-of-blocks form."""
    if isinstance(content, str):
        return (TextBlock(content),) if content else ()
    blocks: list[ContentBlock] = []
    for block in content:
        if not isinstance(block, (TextBlock, ImageBlock, ThinkingBlock, ToolCallBlock, ToolResultBlock)):
            raise TypeError(f"Unexpected content block: {type(block)!r}")
        blocks.append(block)
    return tuple(blocks)


def _normalize_visible(content: ToolResultContentInput) -> tuple[VisibleBlock, ...]:
    """Coerce tool-result inner content to text/image blocks only."""
    if isinstance(content, str):
        return (TextBlock(content),) if content else ()
    blocks: list[VisibleBlock] = []
    for block in content:
        if not isinstance(block, (TextBlock, ImageBlock)):
            raise TypeError(
                f"ToolResultBlock content only accepts TextBlock or ImageBlock, "
                f"got {type(block)!r}"
            )
        blocks.append(block)
    return tuple(blocks)
