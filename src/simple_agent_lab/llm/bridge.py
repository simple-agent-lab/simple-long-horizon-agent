"""Bridge between lab messages and the LLM access layer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from simple_agent_lab.messages import (
    ImageBlock,
    Message,
    ModelAssistantMessage,
    ModelMessage,
    ModelToolResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    assistant_message,
    to_model_message,
)
from simple_agent_lab.tools import AgentTool, Tool

from .types import ContentBlock, LLMMessage, LLMTool, LLMResponse, ToolCall


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
    """Project a ModelMessage into the current LLM access layer type."""
    if isinstance(message, ModelAssistantMessage):
        tool_calls = [
            ToolCall(block.id, block.name, dict(block.arguments))
            for block in message.content
            if isinstance(block, ToolCallBlock)
        ]
        content_blocks = [
            block
            for block in message.content
            if not isinstance(block, ToolCallBlock)
        ]
        return LLMMessage(
            role="assistant",
            content=_llm_content(content_blocks),
            tool_calls=tool_calls or None,
        )

    if isinstance(message, ModelToolResultMessage):
        return LLMMessage(
            role="tool_result",
            content=_llm_content(message.content),
            tool_call_id=message.tool_call_id,
            name=message.tool_name,
        )

    return LLMMessage(
        role=message.role,
        content=_llm_content(message.content),
    )


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
    """Convert a drained LLM response into a runtime assistant message."""
    thinking = (ThinkingBlock(text=response.thinking),) if response.thinking else ()
    tool_calls = [
        ToolCallBlock(tool_call.id, tool_call.name, dict(tool_call.arguments))
        for tool_call in response.tool_calls
    ]
    return assistant_message(
        response.text,
        sender=sender,
        target=target,
        kind=kind,
        thinking=thinking,
        tool_calls=tool_calls,
        data=data,
    )


def _llm_content(
    blocks: Sequence[TextBlock | ImageBlock | ThinkingBlock],
) -> str | list[ContentBlock]:
    if not blocks:
        return ""
    if len(blocks) == 1 and isinstance(blocks[0], TextBlock):
        return blocks[0].text
    return [_llm_content_block(block) for block in blocks]


def _llm_content_block(block: TextBlock | ImageBlock | ThinkingBlock) -> ContentBlock:
    if isinstance(block, TextBlock):
        return ContentBlock(kind="text", text=block.text)
    if isinstance(block, ImageBlock):
        return ContentBlock(
            kind="image",
            data=block.data,
            mime_type=block.mime_type,
        )
    return ContentBlock(
        kind="thinking",
        thinking=block.text,
        signature=block.signature,
        redacted=block.redacted,
    )
