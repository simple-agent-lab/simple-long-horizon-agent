"""OpenAI Responses API adapter.

Uses the official `openai` SDK's `client.responses.create(...)`. Blocking-only.
Same SDK as `openai-chat`, but the wire shape is different: `instructions`
replaces the leading system message, prior turns and tool calls/results are
flat `input` items, and function tools are unwrapped (no `{"type":"function",
"function": {...}}` nesting).

The SDK import is deferred to `stream()`.

Reasoning is a first-class block here too. Inbound, the adapter flattens an
output `reasoning` item's `summary_text` parts into a `ThinkingBlock` (the
item id is kept on `ThinkingBlock.signature`). Outbound, prior reasoning is
replayed as a `reasoning` input item ahead of the assistant message/tool
call it preceded -- reasoning models served this way (e.g.
deepseek-via-zenmux) reject the next turn otherwise. Gated by
`Provider.replay_reasoning`.

Provider config:

    Provider(
        id="gpt-responses",
        api="openai-responses",
        model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        base_url=None,
    )

Pass-through request options via `LLMRequest.extra`:

    extra["reasoning"]    : dict          (raw override of the normalized
                                           ``LLMRequest.reasoning`` knob,
                                           e.g. {"effort": "low"})
    extra["extra_headers"]: dict          (request headers)
    extra["metadata"]     : dict
    extra["store"]        : bool
    extra["user"]         : str
    extra["include"]      : list[str]     (merged with
                                           ``reasoning.encrypted_content`` when
                                           Provider.replay_reasoning is enabled)
    extra["previous_response_id"] : str
    extra["top_p"]        : float
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Iterator

from ...messages import (
    ContentBlock,
    ImageBlock,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    encode_image_data_url,
    text_of,
)
from ._spine import (
    TOOL_RESULT_VISUAL_CAPTION,
    emit_response,
    openai_usage,
    parse_tool_arguments,
    resolve_effort,
    resolve_temperature,
)
from ..env import resolve_api_key
from ..stream import register_adapter
from ..types import (
    LLMMessage,
    LLMRequest,
    LLMTool,
    StopReason,
    StreamEvent,
)


REASONING_ENCRYPTED_CONTENT = "reasoning.encrypted_content"
REASONING_ITEMS_EXTRA = "openai_responses.reasoning_items"


def stream(req: LLMRequest) -> Iterator[StreamEvent]:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - import error path
        raise RuntimeError(
            "openai-responses adapter requires the 'openai' package. "
            "Install the package dependencies with: uv sync "
            "or: pip install openai"
        ) from exc

    client = OpenAI(
        api_key=_api_key(req),
        base_url=req.provider.base_url,
    )

    items = _to_responses_input(req, include_reasoning=req.provider.replay_reasoning)
    tools = _to_responses_tools(req.tools)

    kwargs: dict[str, Any] = {
        "model": req.provider.model,
        "input": items,
    }
    include = _include_items(
        req.extra.get("include"),
        include_encrypted_reasoning=req.provider.replay_reasoning,
    )
    if include is not None:
        kwargs["include"] = include
    if req.system_prompt:
        kwargs["instructions"] = req.system_prompt
    if tools:
        kwargs["tools"] = tools
    temperature = resolve_temperature(req)
    if temperature is not None:
        kwargs["temperature"] = temperature
    max_tokens = req.max_tokens or req.provider.default_max_tokens
    if max_tokens is not None:
        kwargs["max_output_tokens"] = max_tokens
    if req.timeout_seconds:
        kwargs["timeout"] = req.timeout_seconds
    # The Responses API nests effort under ``reasoning={"effort": ...}``. A raw
    # extra["reasoning"] (below) overrides the normalized knob.
    effort = resolve_effort(req)
    if effort:
        kwargs["reasoning"] = {"effort": effort}
    for key in (
        "reasoning",
        "extra_headers",
        "metadata",
        "store",
        "user",
        "previous_response_id",
        "top_p",
    ):
        if key in req.extra:
            kwargs[key] = req.extra[key]

    sdk_response = client.responses.create(**kwargs)

    blocks: list[ContentBlock] = []
    for item in getattr(sdk_response, "output", None) or []:
        itype = getattr(item, "type", None)
        if itype == "reasoning":
            reasoning_text = _reasoning_summary_text(item)
            if reasoning_text:
                blocks.append(
                    ThinkingBlock(
                        text=reasoning_text,
                        signature=getattr(item, "id", None),
                        source_field="reasoning",
                    )
                )
        elif itype == "message":
            for block in getattr(item, "content", None) or []:
                if getattr(block, "type", None) == "output_text":
                    text = getattr(block, "text", "") or ""
                    if text:
                        blocks.append(TextBlock(text=text))
        elif itype == "function_call":
            args_str = getattr(item, "arguments", "") or ""
            arguments = parse_tool_arguments(args_str)
            blocks.append(
                ToolCallBlock(
                    id=getattr(item, "call_id", None) or getattr(item, "id", ""),
                    name=getattr(item, "name", "") or "",
                    arguments=arguments,
                )
            )

    tool_calls = [block for block in blocks if isinstance(block, ToolCallBlock)]
    stop_reason = _map_responses_stop(sdk_response, tool_calls)
    usage = openai_usage(
        getattr(sdk_response, "usage", None),
        input_field="input_tokens",
        output_field="output_tokens",
        details_field="input_tokens_details",
    )

    yield from emit_response(
        blocks,
        stop_reason=stop_reason,
        usage=usage,
        sdk_response=sdk_response,
        request_kwargs=kwargs,
    )


def _api_key(req: LLMRequest) -> str | None:
    # OpenAI SDKs reject an empty key; a key-free endpoint gets a placeholder.
    return resolve_api_key(req.provider, placeholder="not-needed")


def _include_items(
    value: Any,
    *,
    include_encrypted_reasoning: bool,
) -> list[Any] | None:
    """Return Responses include items, optionally enabling reasoning continuity."""

    if value is None:
        include: list[Any] = []
    elif isinstance(value, str):
        include = [value]
    else:
        include = list(value)
    if include_encrypted_reasoning and REASONING_ENCRYPTED_CONTENT not in include:
        include.append(REASONING_ENCRYPTED_CONTENT)
    return include or None


def _to_responses_input(
    req: LLMRequest, *, include_reasoning: bool = True
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in req.messages:
        if message.role == "system":
            items.append(
                {
                    "type": "message",
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": text_of(message.content)}
                    ],
                }
            )
        elif message.role == "user":
            visual_blocks: list[dict[str, Any]] = []
            for tool_result in message.tool_results:
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_result.tool_call_id,
                        "output": text_of(tool_result.content),
                    }
                )
                images = [b for b in tool_result.content if isinstance(b, ImageBlock)]
                if images:
                    visual_blocks.append(
                        {
                            "type": "input_text",
                            "text": TOOL_RESULT_VISUAL_CAPTION.format(
                                tool_name=tool_result.tool_name
                            ),
                        }
                    )
                    for image in images:
                        visual_blocks.append(
                            {
                                "type": "input_image",
                                "image_url": encode_image_data_url(
                                    image.mime_type, image.data
                                ),
                            }
                        )
            if visual_blocks:
                items.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": visual_blocks,
                    }
                )
            if any(isinstance(b, (TextBlock, ImageBlock)) for b in message.content):
                items.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": _to_responses_user_content(message),
                    }
                )
        elif message.role == "assistant":
            text = text_of(message.content)
            tool_calls = list(message.tool_calls)
            # Reasoning models served over the Responses API (e.g.
            # deepseek-via-zenmux) reject a follow-up turn whose assistant
            # message had thinking unless the prior reasoning item is
            # echoed back ahead of the message/tool_call it preceded.
            if include_reasoning and (text or tool_calls):
                reasoning_item = _reasoning_item(message)
                if reasoning_item is not None:
                    items.append(reasoning_item)
            if text:
                items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }
                )
            for tool_call in tool_calls:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call.id,
                        "name": tool_call.name,
                        "arguments": json.dumps(dict(tool_call.arguments)),
                    }
                )
    return items


def _reasoning_summary_text(item: Any) -> str:
    """Join an output ``reasoning`` item's ``summary_text`` parts.

    The Responses API carries a model's thinking as a list of
    ``summary`` parts on the reasoning item. We flatten the text parts
    into one string; the structured shape is rebuilt on replay.
    """
    parts: list[str] = []
    for part in getattr(item, "summary", None) or []:
        if getattr(part, "type", None) == "summary_text":
            text = getattr(part, "text", "") or ""
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def _reasoning_item(message: LLMMessage) -> dict[str, Any] | None:
    """Rebuild the reasoning item to echo back for an assistant turn.

    Mirrors the inbound shape: one ``summary_text`` part per stored
    `ThinkingBlock`. The original item id (kept on `ThinkingBlock.signature`)
    and encrypted content are replayed when present so the wire item matches
    what the model emitted.
    """
    blocks = [block for block in message.thinking_blocks if block.text]
    extra_items = _reasoning_items_from_extra(message.extra)
    if not blocks and not extra_items:
        return None
    signature = next((block.signature for block in blocks if block.signature), None)
    extra_item = _matching_reasoning_item(signature, extra_items)
    if signature is None and extra_item is not None:
        signature = _string_or_none(extra_item.get("id"))
    summary = _summary_from_blocks(blocks)
    if not summary and extra_item is not None:
        summary = _summary_from_extra(extra_item)
    encrypted_content = _encrypted_content_for(signature, extra_items)
    if not summary and not encrypted_content:
        return None
    item: dict[str, Any] = {"type": "reasoning"}
    # Some Responses-compatible endpoints require the `summary` field even when
    # only encrypted reasoning continuity is available for replay.
    item["summary"] = summary
    if signature:
        item["id"] = signature
    if encrypted_content:
        item["encrypted_content"] = encrypted_content
    return item


def _summary_from_blocks(blocks: list[ThinkingBlock]) -> list[dict[str, str]]:
    return [
        {"type": "summary_text", "text": block.text} for block in blocks if block.text
    ]


def _matching_reasoning_item(
    signature: str | None,
    items: list[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    if signature:
        for item in items:
            if _string_or_none(item.get("id")) == signature:
                return item
        return None
    return next(iter(items), None)


def _summary_from_extra(item: Mapping[str, Any]) -> list[dict[str, str]]:
    raw_summary = item.get("summary")
    if not isinstance(raw_summary, list):
        return []
    summary: list[dict[str, str]] = []
    for part in raw_summary:
        if not isinstance(part, Mapping):
            continue
        if part.get("type") != "summary_text":
            continue
        text = part.get("text")
        if text is None:
            continue
        summary.append({"type": "summary_text", "text": str(text)})
    return summary


def _reasoning_items_from_extra(extra: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_items = extra.get(REASONING_ITEMS_EXTRA)
    if not isinstance(raw_items, list):
        return []
    return [item for item in raw_items if isinstance(item, Mapping)]


def _encrypted_content_for(
    signature: str | None,
    items: list[Mapping[str, Any]],
) -> str | None:
    if signature:
        for item in items:
            if _string_or_none(item.get("id")) == signature:
                value = item.get("encrypted_content")
                return str(value) if value else None
        return None
    value = next((item.get("encrypted_content") for item in items), None)
    return str(value) if value else None


def _string_or_none(value: Any) -> str | None:
    return str(value) if value else None


def _to_responses_user_content(message: LLMMessage) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, TextBlock) and block.text:
            parts.append({"type": "input_text", "text": block.text})
        elif isinstance(block, ImageBlock):
            parts.append(
                {
                    "type": "input_image",
                    "image_url": encode_image_data_url(block.mime_type, block.data),
                }
            )
    return parts or [{"type": "input_text", "text": ""}]


def _to_responses_tools(tools: list[LLMTool]) -> list[dict[str, Any]]:
    # Responses uses flat function tools (no chat-style {"function": {...}} wrapper).
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        for tool in tools
    ]


def _map_responses_stop(raw: Any, tool_calls: list[ToolCallBlock]) -> StopReason:
    if tool_calls:
        return "tool_use"
    incomplete = getattr(raw, "incomplete_details", None)
    if incomplete is not None:
        reason = getattr(incomplete, "reason", None)
        if reason == "max_output_tokens":
            return "max_tokens"
        if reason:
            return "error"
    status = getattr(raw, "status", None)
    if status == "incomplete":
        return "error"
    return "end_turn"


register_adapter("openai-responses", stream)
