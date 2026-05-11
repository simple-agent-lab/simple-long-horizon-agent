"""OpenAI Chat Completions API adapter.

Uses the official `openai` SDK. Blocking-only. Same call path also serves
OpenAI-compatible endpoints (Ollama, vLLM, OpenRouter, LM Studio, ...)
when `Provider.base_url` is set.

Reasoning content is treated as a first-class content block. On the way
in, the adapter reads `message.reasoning_content` (DeepSeek / mimo style)
and surfaces it as a `ContentBlock(kind="thinking", ...)` ahead of the
text and tool_call blocks in `LLMResponse.content`. On the way out, prior
assistant thinking blocks are replayed via the same `reasoning_content`
field on the outbound assistant dict so multi-turn tool-use chains stay
continuous. The replay step is gated by `Provider.replay_reasoning`
(default True; flip to False for endpoints that reject the field).

The SDK import is deferred to `stream()` so the module registers even
without `[openai]` installed.

Provider config:

    Provider(
        id="gpt",
        api="openai-chat",
        model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        base_url=None,
    )

Local Ollama is just::

    Provider(
        id="ollama",
        api="openai-chat",
        model="llama3:8b",
        base_url="http://localhost:11434/v1",
        api_key_env="",   # not needed locally
    )

Pass-through request options via `LLMRequest.extra`:

    extra["seed"]            : int
    extra["top_p"]           : float
    extra["stop"]            : str | list[str]
    extra["user"]            : str
    extra["response_format"] : dict
    extra["parallel_tool_calls"] : bool
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
            "openai-chat adapter requires the 'openai' package. "
            "Install with: uv pip install 'simple-agent-lab[openai]' "
            "or: pip install openai"
        ) from exc

    client = OpenAI(
        api_key=_api_key(req),
        base_url=req.provider.base_url,
    )

    messages = _to_chat_messages(req)
    tools = _to_chat_tools(req.tools)

    kwargs: dict[str, Any] = {
        "model": req.provider.model,
        "messages": messages,
    }
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
        kwargs["max_tokens"] = max_tokens
    if req.timeout_seconds:
        kwargs["timeout"] = req.timeout_seconds
    for key in ("seed", "top_p", "stop", "user", "response_format", "parallel_tool_calls"):
        if key in req.extra:
            kwargs[key] = req.extra[key]

    raw = client.chat.completions.create(**kwargs)
    if not raw.choices:
        raise RuntimeError("openai-chat: response had no choices")
    choice = raw.choices[0]
    message = choice.message

    reasoning_text = _extract_reasoning(message)
    text = getattr(message, "content", None) or ""
    tool_calls: list[ToolCall] = []
    for tool_call in getattr(message, "tool_calls", None) or []:
        function = getattr(tool_call, "function", None)
        name = getattr(function, "name", "") if function else ""
        args_str = getattr(function, "arguments", "") if function else ""
        try:
            arguments = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            arguments = {"_raw_arguments": args_str}
        tool_calls.append(
            ToolCall(id=getattr(tool_call, "id", ""), name=name, arguments=arguments)
        )

    blocks: list[ContentBlock] = []
    if reasoning_text:
        blocks.append(ContentBlock(kind="thinking", thinking=reasoning_text))
    if text:
        blocks.append(ContentBlock(kind="text", text=text))
    for tool_call in tool_calls:
        blocks.append(ContentBlock(kind="tool_call", tool_call=tool_call))

    stop_reason = _map_openai_finish(getattr(choice, "finish_reason", None))
    usage = _openai_chat_usage(getattr(raw, "usage", None))

    if reasoning_text:
        yield StreamEvent(kind="thinking_delta", payload={"delta": reasoning_text})
    if text:
        yield StreamEvent(kind="text_delta", payload={"delta": text})
    for tool_call in tool_calls:
        yield StreamEvent(kind="tool_call_start", payload={"tool_call": tool_call})
        yield StreamEvent(kind="tool_call_complete", payload={"tool_call": tool_call})
    yield StreamEvent(kind="usage_update", payload={"usage": usage})

    response = LLMResponse(
        content=blocks,
        stop_reason=stop_reason,
        usage=usage,
        raw={
            "provider": "openai-chat",
            "model": req.provider.model,
            "id": getattr(raw, "id", None),
            "finish_reason": getattr(choice, "finish_reason", None),
        },
    )
    yield StreamEvent(kind="done", payload={"response": response})


def _api_key(req: LLMRequest) -> str | None:
    env = req.provider.api_key_env
    if not env:
        # Local/compat endpoints (Ollama) don't require a key. The SDK still
        # demands a non-empty value, so pass a placeholder.
        return "not-needed"
    api_key = os.environ.get(env)
    if not api_key:
        raise RuntimeError(
            f"Provider {req.provider.id!r} requires env var {env!r}; not set."
        )
    return api_key


def _extract_reasoning(message: Any) -> str:
    """Read reasoning_content from the SDK message object.

    Not part of OpenAI's official Chat schema, so pydantic v2 SDK
    responses expose it through `__getattr__` (extra="allow"); plain
    attribute access also covers the SimpleNamespace stubs used in tests.
    """
    value = getattr(message, "reasoning_content", None)
    return value if isinstance(value, str) else ""


def _to_chat_messages(req: LLMRequest) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if req.system_prompt:
        out.append({"role": "system", "content": req.system_prompt})
    for message in req.messages:
        if message.role == "system":
            out.append({"role": "system", "content": _message_text(message)})
        elif message.role == "user":
            out.append({"role": "user", "content": _to_chat_user_content(message)})
        elif message.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant"}
            text = _message_text(message)
            entry["content"] = text if text else None
            if message.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(dict(tool_call.arguments)),
                        },
                    }
                    for tool_call in message.tool_calls
                ]
            if req.provider.replay_reasoning:
                reasoning = _message_reasoning(message)
                if reasoning:
                    entry["reasoning_content"] = reasoning
            out.append(entry)
        elif message.role == "tool_result":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id or "",
                    "content": _message_text(message),
                }
            )
    return out


def _to_chat_user_content(message: LLMMessage) -> Any:
    if isinstance(message.content, str):
        return message.content
    parts: list[dict[str, Any]] = []
    for block in message.content:
        if block.kind == "text" and block.text:
            parts.append({"type": "text", "text": block.text})
        elif block.kind == "image" and block.data:
            mime = block.mime_type or "image/png"
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{block.data}"},
                }
            )
    return parts if parts else ""


def _message_text(message: LLMMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(block.text for block in message.content if block.kind == "text")


def _message_reasoning(message: LLMMessage) -> str:
    return "\n\n".join(
        block.thinking for block in message.thinking_blocks() if block.thinking
    )


def _to_chat_tools(tools: list[LLMTool]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


_OPENAI_STOP_MAP: dict[str, StopReason] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "error",
}


def _map_openai_finish(raw: str | None) -> StopReason:
    if raw is None:
        return "end_turn"
    return _OPENAI_STOP_MAP.get(raw, "end_turn")


def _openai_chat_usage(usage: Any) -> Usage:
    if usage is None:
        return Usage()
    cached = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = int(getattr(details, "cached_tokens", 0) or 0)
    return Usage(
        input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        cache_read_tokens=cached,
    )


register_adapter("openai-chat", stream)
