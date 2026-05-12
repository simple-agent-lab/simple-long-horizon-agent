"""Bridge between lab messages and the LLM access layer.

With the unified content model, runtime `Message` / `ModelMessage` and
wire-layer `LLMMessage` all carry the same `tuple[ContentBlock, ...]`
shape, so the bridge is now mostly a re-roling pass plus runtime usage
translation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from simple_agent_lab.messages import (
    Message,
    ModelAssistantMessage,
    ModelMessage,
    ModelToolResultMessage,
    TokenUsage,
    assistant_message,
    to_model_message,
)
from simple_agent_lab.tools import AgentTool, Tool

from .types import LLMMessage, LLMResponse, LLMTool, Usage


def message_to_llm_message(message: Message, *, with_header: bool = False) -> LLMMessage:
    """Project a runtime Message into the LLM layer's provider-neutral shape."""
    return model_message_to_llm_message(to_model_message(message, with_header=with_header))


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


def model_message_to_llm_message(message: ModelMessage) -> LLMMessage:
    """Project a ModelMessage into the LLM layer.

    Both layers share the same content shape, so this only re-roles and
    surfaces the tool-result sidecar fields.
    """
    if isinstance(message, ModelToolResultMessage):
        return LLMMessage(
            role="tool_result",
            content=message.content,
            tool_call_id=message.tool_call_id,
            name=message.tool_name,
        )
    if isinstance(message, ModelAssistantMessage):
        return LLMMessage(role="assistant", content=message.content)
    return LLMMessage(role=message.role, content=message.content)


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
    pass it through.
    """
    return assistant_message(
        response.content,
        sender=sender,
        target=target,
        kind=kind,
        usage=_translate_usage(response.usage),
        data=data,
    )


def _translate_usage(usage: Usage) -> TokenUsage | None:
    """Project the LLM-layer Usage onto the runtime TokenUsage.

    Returns None when every field is zero so the message-side default ("we have
    no usage info") is preserved instead of fabricating a zero-token record
    that downstream consumers would treat as authoritative.
    """
    fields = asdict(usage)
    if not any(fields.values()):
        return None
    return TokenUsage(**fields)
