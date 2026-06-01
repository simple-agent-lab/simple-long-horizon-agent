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

    extra["reasoning"]    : dict          (e.g. {"effort": "low"})
    extra["extra_headers"]: dict          (request headers)
    extra["metadata"]     : dict
    extra["store"]        : bool
    extra["user"]         : str
    extra["previous_response_id"] : str
    extra["top_p"]        : float
"""

from __future__ import annotations

import json
import os
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
from . import TOOL_RESULT_VISUAL_CAPTION, capture_request, sdk_dump
from ..stream import register_adapter
from ..types import (
    RAW_ARGUMENTS_KEY,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMTool,
    StopReason,
    StreamEvent,
    TokenUsage,
)


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
    if req.system_prompt:
        kwargs["instructions"] = req.system_prompt
    if tools:
        kwargs["tools"] = tools
    temperature = (
        req.temperature
        if req.temperature is not None
        else req.provider.default_temperature
    )
    if temperature is not None:
        kwargs["temperature"] = temperature
    max_tokens = req.max_tokens or req.provider.default_max_tokens
    if max_tokens is not None:
        kwargs["max_output_tokens"] = max_tokens
    if req.timeout_seconds:
        kwargs["timeout"] = req.timeout_seconds
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
            try:
                arguments = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                arguments = {RAW_ARGUMENTS_KEY: args_str}
            blocks.append(
                ToolCallBlock(
                    id=getattr(item, "call_id", None) or getattr(item, "id", ""),
                    name=getattr(item, "name", "") or "",
                    arguments=arguments,
                )
            )

    tool_calls = [block for block in blocks if isinstance(block, ToolCallBlock)]
    stop_reason = _map_responses_stop(sdk_response, tool_calls)
    usage = _responses_usage(getattr(sdk_response, "usage", None))

    for block in blocks:
        if isinstance(block, ThinkingBlock) and block.text:
            yield StreamEvent(kind="thinking_delta", payload={"delta": block.text})
        elif isinstance(block, TextBlock) and block.text:
            yield StreamEvent(kind="text_delta", payload={"delta": block.text})
        elif isinstance(block, ToolCallBlock):
            yield StreamEvent(kind="tool_call_start", payload={"tool_call": block})
            yield StreamEvent(kind="tool_call_complete", payload={"tool_call": block})
    yield StreamEvent(kind="usage_update", payload={"usage": usage})

    response = LLMResponse(
        content=tuple(blocks),
        stop_reason=stop_reason,
        usage=usage,
        # The served model the API resolved to (e.g. an alias -> dated
        # snapshot); complete() only back-fills the requested model.
        model=getattr(sdk_response, "model", "") or "",
        raw={"request": capture_request(kwargs), "response": sdk_dump(sdk_response)},
    )
    yield StreamEvent(kind="done", payload={"response": response})


def _api_key(req: LLMRequest) -> str | None:
    env = req.provider.api_key_env
    if not env:
        return "not-needed"
    api_key = os.environ.get(env)
    if not api_key:
        raise RuntimeError(
            f"Provider {req.provider.id!r} requires env var {env!r}; not set."
        )
    return api_key


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
            # Reasoning models served over the Responses API (e.g.
            # deepseek-via-zenmux) reject a follow-up turn whose assistant
            # message had thinking unless the prior reasoning item is
            # echoed back ahead of the message/tool_call it preceded.
            if include_reasoning:
                reasoning_item = _reasoning_item(message)
                if reasoning_item is not None:
                    items.append(reasoning_item)
            text = text_of(message.content)
            if text:
                items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }
                )
            for tool_call in message.tool_calls:
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
    is replayed when present so the wire item matches what the model emitted.
    """
    blocks = [block for block in message.thinking_blocks if block.text]
    if not blocks:
        return None
    item: dict[str, Any] = {
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": block.text} for block in blocks],
    }
    signature = next((block.signature for block in blocks if block.signature), None)
    if signature:
        item["id"] = signature
    return item


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


def _responses_usage(usage: Any) -> TokenUsage:
    if usage is None:
        return TokenUsage()
    cached = 0
    details = getattr(usage, "input_tokens_details", None)
    if details is not None:
        cached = int(getattr(details, "cached_tokens", 0) or 0)
    # `input_tokens` already includes `cached_tokens` as a subset; normalize
    # to the project's additive-cache convention so context_tokens is correct.
    return TokenUsage.from_inclusive_input(
        total_input=int(getattr(usage, "input_tokens", 0) or 0),
        output=int(getattr(usage, "output_tokens", 0) or 0),
        cached_read=cached,
    )


register_adapter("openai-responses", stream)
