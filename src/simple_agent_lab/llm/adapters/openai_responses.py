"""OpenAI Responses API adapter.

Uses the official `openai` SDK's `client.responses.create(...)`. Blocking-only.
Same SDK as `openai-chat`, but the wire shape is different: `instructions`
replaces the leading system message, prior turns and tool calls/results are
flat `input` items, and function tools are unwrapped (no `{"type":"function",
"function": {...}}` nesting).

The SDK import is deferred to `stream()`.

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

from ..stream import register_adapter
from ..types import (
    ContentBlock,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMTool,
    StopReason,
    StreamEvent,
    ToolCall,
    Usage,
)


def stream(req: LLMRequest) -> Iterator[StreamEvent]:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - import error path
        raise RuntimeError(
            "openai-responses adapter requires the 'openai' package. "
            "Install with: uv pip install 'simple-agent-lab[openai]' "
            "or: pip install openai"
        ) from exc

    client = OpenAI(
        api_key=_api_key(req),
        base_url=req.provider.base_url,
    )

    items = _to_responses_input(req)
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
    for key in ("reasoning", "metadata", "store", "user", "previous_response_id", "top_p"):
        if key in req.extra:
            kwargs[key] = req.extra[key]

    raw = client.responses.create(**kwargs)

    blocks: list[ContentBlock] = []
    for item in getattr(raw, "output", None) or []:
        itype = getattr(item, "type", None)
        if itype == "message":
            for block in getattr(item, "content", None) or []:
                if getattr(block, "type", None) == "output_text":
                    text = getattr(block, "text", "") or ""
                    if text:
                        blocks.append(ContentBlock(kind="text", text=text))
        elif itype == "function_call":
            args_str = getattr(item, "arguments", "") or ""
            try:
                arguments = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                arguments = {"_raw_arguments": args_str}
            blocks.append(
                ContentBlock(
                    kind="tool_call",
                    tool_call=ToolCall(
                        id=getattr(item, "call_id", None) or getattr(item, "id", ""),
                        name=getattr(item, "name", "") or "",
                        arguments=arguments,
                    ),
                )
            )

    tool_calls = [
        block.tool_call
        for block in blocks
        if block.kind == "tool_call" and block.tool_call is not None
    ]
    stop_reason = _map_responses_stop(raw, tool_calls)
    usage = _responses_usage(getattr(raw, "usage", None))

    for block in blocks:
        if block.kind == "text" and block.text:
            yield StreamEvent(kind="text_delta", payload={"delta": block.text})
        elif block.kind == "tool_call" and block.tool_call is not None:
            yield StreamEvent(kind="tool_call_start", payload={"tool_call": block.tool_call})
            yield StreamEvent(kind="tool_call_complete", payload={"tool_call": block.tool_call})
    yield StreamEvent(kind="usage_update", payload={"usage": usage})

    response = LLMResponse(
        content=blocks,
        stop_reason=stop_reason,
        usage=usage,
        raw={
            "provider": "openai-responses",
            "model": req.provider.model,
            "id": getattr(raw, "id", None),
            "status": getattr(raw, "status", None),
        },
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


def _to_responses_input(req: LLMRequest) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in req.messages:
        if message.role == "system":
            items.append(
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": _message_text(message)}],
                }
            )
        elif message.role == "user":
            items.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": _to_responses_user_content(message),
                }
            )
        elif message.role == "assistant":
            text = _message_text(message)
            if text:
                items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }
                )
            for tool_call in message.tool_calls or []:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call.id,
                        "name": tool_call.name,
                        "arguments": json.dumps(dict(tool_call.arguments)),
                    }
                )
        elif message.role == "tool_result":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id or "",
                    "output": _message_text(message),
                }
            )
    return items


def _to_responses_user_content(message: LLMMessage) -> list[dict[str, Any]]:
    if isinstance(message.content, str):
        return [{"type": "input_text", "text": message.content}]
    parts: list[dict[str, Any]] = []
    for block in message.content:
        if block.kind == "text" and block.text:
            parts.append({"type": "input_text", "text": block.text})
        elif block.kind == "image" and block.data:
            mime = block.mime_type or "image/png"
            parts.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{mime};base64,{block.data}",
                }
            )
    return parts or [{"type": "input_text", "text": ""}]


def _message_text(message: LLMMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(block.text for block in message.content if block.kind == "text")


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


def _map_responses_stop(raw: Any, tool_calls: list[ToolCall]) -> StopReason:
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


def _responses_usage(usage: Any) -> Usage:
    if usage is None:
        return Usage()
    cached = 0
    details = getattr(usage, "input_tokens_details", None)
    if details is not None:
        cached = int(getattr(details, "cached_tokens", 0) or 0)
    return Usage(
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        cache_read_tokens=cached,
    )


register_adapter("openai-responses", stream)
