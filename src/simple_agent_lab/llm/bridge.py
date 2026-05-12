"""Bridge between runtime Messages and the LLM access layer.

One projection: `Message → LLMMessage`. The content block tuple is
unified between runtime and wire layers, so this only:

  * picks the right wire `role`,
  * surfaces ToolResultMessage's `tool_call_id` / `tool_name` sidecar,
  * optionally prepends a routing header (`[sender -> target | kind/channel]`)
    so multi-agent transcripts stay legible when sent to a single model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from simple_agent_lab.messages import (
    Message,
    TextBlock,
    TokenUsage,
    ToolResultMessage,
    assistant_message,
)
from simple_agent_lab.tools import AgentTool, Tool

from .types import LLMMessage, LLMResponse, LLMTool


def message_to_llm_message(message: Message, *, with_header: bool = False) -> LLMMessage:
    """Project a runtime Message into the LLM layer's provider-neutral shape.

    Per-message provider hints stashed under ``message.data["extra"]``
    are lifted to ``LLMMessage.extra`` so adapters that opt in can
    apply them on the wire.
    """
    extra = dict(message.data.get("extra") or {}) if message.data else {}
    if isinstance(message, ToolResultMessage):
        return LLMMessage(
            role="tool_result",
            content=message.content,
            tool_call_id=message.tool_call_id,
            extra=extra,
        )
    header = _routing_header(message) if with_header else ""
    content = (TextBlock(header), *message.content) if header else message.content
    return LLMMessage(role=message.role, content=content, extra=extra)


def messages_to_llm_messages(
    messages: Sequence[Message],
    *,
    with_header: bool = False,
    skip_kinds: set[str] | None = None,
) -> list[LLMMessage]:
    skipped = skip_kinds if skip_kinds is not None else {"notification", "trace"}
    return [
        message_to_llm_message(message, with_header=with_header)
        for message in messages
        if message.kind not in skipped
    ]


def tool_to_llm_tool(tool: Tool | AgentTool) -> LLMTool:
    """Project a shared Tool value to the LLM layer's wire tool definition."""
    return LLMTool(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
    )


def llm_response_to_assistant_message(
    response: LLMResponse,
    *,
    sender: str,
    target: str,
    kind: str,
    data: dict[str, Any] | None = None,
) -> Message:
    """Wrap a drained LLM response in a runtime AssistantMessage.

    `response.content` is already the canonical block tuple, so we just
    pass it through. The adapter's wire snapshot (verbatim request kwargs
    + response dump) rides along on `AssistantMessage.data["wire"]` so
    the runtime trace can show the HTTP-level layer alongside the
    standardized content blocks.
    """
    merged_data = dict(data or {})
    if response.wire:
        merged_data["wire"] = response.wire
    return assistant_message(
        response.content,
        sender=sender,
        target=target,
        kind=kind,
        usage=_usage_or_none(response.usage),
        data=merged_data,
    )


def _usage_or_none(usage: TokenUsage) -> TokenUsage | None:
    """Treat an all-zeros usage as 'unknown' rather than as authoritative."""
    if usage.input_tokens or usage.output_tokens or usage.cache_read_tokens or usage.cache_write_tokens:
        return usage
    return None


def _routing_header(message: Message) -> str:
    if isinstance(message, ToolResultMessage):
        return ""
    has_meta = bool(message.sender or message.target) or message.kind != "message"
    if not has_meta:
        return ""
    return f"[{message.sender} -> {message.target} | {message.kind}/{message.channel}]"
