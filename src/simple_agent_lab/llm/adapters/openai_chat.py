"""OpenAI Chat Completions API adapter.

Uses the official `openai` SDK. Blocking-only. Same call path also serves
OpenAI-compatible endpoints (Ollama, vLLM, OpenRouter, LM Studio, ...)
when `Provider.base_url` is set.

Reasoning content is treated as a first-class block. On the way in, the
adapter reads `message.reasoning_content` (DeepSeek / mimo style) and
surfaces it as a `ThinkingBlock` ahead of the text and tool_call blocks
on `LLMResponse.content`. On the way out, prior assistant thinking
blocks are replayed via the same `reasoning_content` field on the
outbound assistant dict (gated by `Provider.replay_reasoning`).

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

from ...messages import (
    ContentBlock,
    ImageBlock,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
    text_of,
    tool_result_text,
)
from ..stream import register_adapter
from ..types import (
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
    wire = {"request": kwargs, "response": _wire_dump(raw)}
    if not raw.choices:
        raise RuntimeError("openai-chat: response had no choices")
    choice = raw.choices[0]
    message = choice.message

    reasoning_text = _extract_reasoning(message)
    text = getattr(message, "content", None) or ""
    tool_calls: list[ToolCallBlock] = []
    for tool_call in getattr(message, "tool_calls", None) or []:
        function = getattr(tool_call, "function", None)
        name = getattr(function, "name", "") if function else ""
        args_str = getattr(function, "arguments", "") if function else ""
        try:
            arguments = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            arguments = {"_raw_arguments": args_str}
        tool_calls.append(
            ToolCallBlock(id=getattr(tool_call, "id", ""), name=name, arguments=arguments)
        )

    blocks: list[ContentBlock] = []
    if reasoning_text:
        blocks.append(ThinkingBlock(text=reasoning_text))
    if text:
        blocks.append(TextBlock(text=text))
    blocks.extend(tool_calls)

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
        content=tuple(blocks),
        stop_reason=stop_reason,
        usage=usage,
        raw={
            "provider": "openai-chat",
            "model": req.provider.model,
            "id": getattr(raw, "id", None),
            "finish_reason": getattr(choice, "finish_reason", None),
        },
        wire=wire,
    )
    yield StreamEvent(kind="done", payload={"response": response})


def _wire_dump(raw: Any) -> Any:
    """Best-effort serialization snapshot of an SDK response object."""
    dump = getattr(raw, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:
            pass
    return raw


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
            out.append({"role": "system", "content": text_of(message.content)})
        elif message.role == "user":
            # OpenAI Chat wants one `role="tool"` entry per tool result and a
            # separate `role="user"` entry for any leftover text/image. Split
            # the bundled message accordingly. Tool-result images can't ride
            # inside a `role="tool"` content (the wire schema only accepts a
            # string), so we surface them in an adjacent user-message after
            # the role=tool entry — providers that accept image input in user
            # messages then see the visual output.
            visual_blocks: list[dict[str, Any]] = []
            for tool_result in message.tool_results:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_result.tool_call_id,
                        "content": tool_result_text(tool_result),
                    }
                )
                images = [b for b in tool_result.content if isinstance(b, ImageBlock)]
                if images:
                    visual_blocks.append(
                        {"type": "text",
                         "text": f"Visual output from {tool_result.tool_name}:"}
                    )
                    for image in images:
                        visual_blocks.append(_openai_image_block(image))
            if visual_blocks:
                out.append({"role": "user", "content": visual_blocks})
            if any(
                isinstance(b, (TextBlock, ImageBlock)) for b in message.content
            ):
                out.append({"role": "user", "content": _to_chat_user_content(message)})
        elif message.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant"}
            text = text_of(message.content)
            entry["content"] = text if text else None
            tool_calls = message.tool_calls
            if tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(dict(tool_call.arguments)),
                        },
                    }
                    for tool_call in tool_calls
                ]
            if req.provider.replay_reasoning:
                reasoning = _reasoning_text(message)
                if reasoning:
                    entry["reasoning_content"] = reasoning
            out.append(entry)
    return out


def _to_chat_user_content(message: LLMMessage) -> Any:
    parts: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, TextBlock) and block.text:
            parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageBlock):
            parts.append(_openai_image_block(block))
    return parts if parts else ""


def _openai_image_block(block: ImageBlock) -> dict[str, Any]:
    mime = block.mime_type or "image/png"
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{block.data}"},
    }


def _reasoning_text(message: LLMMessage) -> str:
    return "\n\n".join(block.text for block in message.thinking_blocks if block.text)


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


def _openai_chat_usage(usage: Any) -> TokenUsage:
    if usage is None:
        return TokenUsage()
    cached = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = int(getattr(details, "cached_tokens", 0) or 0)
    return TokenUsage(
        input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        cache_read_tokens=cached,
    )


register_adapter("openai-chat", stream)
