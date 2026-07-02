"""Bridge between runtime Messages and the LLM access layer.

One projection: `Message → LLMMessage`. Runtime and wire layers share
the same `tuple[ContentBlock, ...]` shape, so the bridge only:

  * picks the wire `role` (system / user / assistant),
  * optionally prepends a routing header
    (`[sender -> target | kind]`) so multi-agent transcripts
    stay legible when sent to a single model — skipped for tool-result
    user messages since the per-block `tool_call_id` links them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from simple_agent_lab.messages import (
    AgentName,
    Message,
    MessageKind,
    MessageSidecar,
    TextBlock,
    TokenUsage,
    assistant_message,
    is_tool_result_message,
)
from simple_agent_lab.tools import Tool

from .types import LLMMessage, LLMResponse, LLMTool


_OPENAI_RESPONSES_REASONING_ITEMS_EXTRA = "openai_responses.reasoning_items"


def message_to_llm_message(
    message: Message, *, with_header: bool = False
) -> LLMMessage:
    """Project a runtime Message into the LLM layer's provider-neutral shape.

    Per-message provider hints stashed under ``message.sidecar["extra"]``
    are lifted to ``LLMMessage.extra`` so adapters that opt in can
    apply them on the wire.
    """
    extra = dict(message.sidecar.get("extra") or {}) if message.sidecar else {}
    header = _routing_header(message) if with_header else ""
    content = (TextBlock(header), *message.content) if header else message.content
    return LLMMessage(role=message.role, content=content, extra=extra)


def messages_to_llm_messages(
    messages: Sequence[Message],
    *,
    with_header: bool = False,
    model_invisible_kinds: set[str] | None = None,
) -> list[LLMMessage]:
    invisible = model_invisible_kinds if model_invisible_kinds is not None else set()
    return [
        message_to_llm_message(message, with_header=with_header)
        for message in messages
        if message.kind not in invisible
    ]


def tool_to_llm_tool(tool: Tool) -> LLMTool:
    """Project a shared Tool (or AgentTool subclass) to the wire tool def."""
    return LLMTool(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
    )


def llm_response_to_assistant_message(
    response: LLMResponse,
    *,
    sender: AgentName,
    target: AgentName,
    kind: MessageKind,
    sidecar: MessageSidecar | None = None,
) -> Message:
    """Wrap a drained LLM response in a runtime AssistantMessage.

    `response.content` is already the canonical block tuple, so we just
    pass it through. The adapter's `raw` snapshot (the request/response
    pair, with the messages history pruned) rides along on
    `AssistantMessage.sidecar["raw"]` so the runtime trace can show the
    provider-level view alongside the standardized content blocks,
    and so applications can pull provider-specific response fields
    that the standardized layer doesn't surface.
    """
    merged: MessageSidecar = {}
    if sidecar:
        merged.update(sidecar)
    reasoning_extra = _openai_responses_reasoning_extra(response)
    if reasoning_extra:
        extra = dict(merged.get("extra") or {})
        extra.update(reasoning_extra)
        merged["extra"] = extra
    if response.raw:
        merged["raw"] = response.raw
    return assistant_message(
        response.content,
        sender=sender,
        target=target,
        kind=kind,
        usage=_usage_or_none(response.usage),
        model=response.model,
        sidecar=merged,
    )


def _usage_or_none(usage: TokenUsage) -> TokenUsage | None:
    """Treat an all-zeros usage as 'unknown' rather than as authoritative."""
    return usage if usage.context_tokens else None


def _openai_responses_reasoning_extra(response: LLMResponse) -> dict[str, object]:
    """Extract Responses-only reasoning state for next-turn replay."""

    raw_response = response.raw.get("response") if response.raw else None
    items: list[dict[str, object]] = []
    raw_output = _field(raw_response, "output")
    if not isinstance(raw_output, (list, tuple)):
        return {}
    for item in raw_output:
        if _field(item, "type") != "reasoning":
            continue
        extra_item = _openai_responses_reasoning_item(item)
        if extra_item is not None:
            items.append(extra_item)
    if not items:
        return {}
    return {_OPENAI_RESPONSES_REASONING_ITEMS_EXTRA: items}


def _openai_responses_reasoning_item(item: object) -> dict[str, object] | None:
    extra_item: dict[str, object] = {"type": "reasoning"}
    item_id = _field(item, "id")
    if item_id:
        extra_item["id"] = str(item_id)
    summary = _openai_responses_reasoning_summary(item)
    if summary is not None:
        extra_item["summary"] = summary
    encrypted_content = _field(item, "encrypted_content")
    if encrypted_content:
        extra_item["encrypted_content"] = str(encrypted_content)
    if len(extra_item) == 1:
        return None
    return extra_item


def _openai_responses_reasoning_summary(
    item: object,
) -> list[dict[str, str]] | None:
    raw_summary = _field(item, "summary")
    if not isinstance(raw_summary, (list, tuple)):
        return None
    summary: list[dict[str, str]] = []
    for part in raw_summary:
        if _field(part, "type") != "summary_text":
            continue
        text = _field(part, "text")
        if text is None:
            continue
        summary.append({"type": "summary_text", "text": str(text)})
    return summary


def _field(value: object, name: str) -> object:
    if isinstance(value, dict):
        return cast("dict[str, object]", value).get(name)
    return getattr(value, name, None)


def _routing_header(message: Message) -> str:
    if is_tool_result_message(message):
        return ""
    has_meta = bool(message.sender or message.target) or message.kind != "message"
    if not has_meta:
        return ""
    return f"[{message.sender} -> {message.target} | {message.kind}]"
