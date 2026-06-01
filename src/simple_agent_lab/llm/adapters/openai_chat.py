"""OpenAI Chat Completions API adapter.

Uses the official `openai` SDK. Blocking-only. Same call path also serves
OpenAI-compatible endpoints (Ollama, vLLM, OpenRouter, LM Studio, ...)
when `Provider.base_url` is set.

Reasoning content is treated as a first-class block. On the way in, the
adapter reads the first available reasoning field on the SDK message
(`reasoning_content`, OpenAI / DeepSeek-direct style; or `reasoning`,
what some DeepSeek-via-gateway deployments emit) and surfaces it as a
`ThinkingBlock` ahead of the text and tool_call blocks on
`LLMResponse.content`. The wire field name is remembered on
`ThinkingBlock.source_field` for trace debugging. On the way out, prior
assistant thinking is always replayed under the canonical
`reasoning_content` field, regardless of which key brought it in --
strict gateways (e.g. deepseek-via-zenmux) emit `reasoning` but still
require us to echo back the canonical name. Gated by
`Provider.replay_reasoning`.

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
from collections.abc import Sequence
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
            "openai-chat adapter requires the 'openai' package. "
            "Install the package dependencies with: uv sync "
            "or: pip install openai"
        ) from exc

    client = OpenAI(
        api_key=_api_key(req),
        base_url=req.provider.base_url,
    )

    messages = to_openai_chat_messages(
        req.messages,
        system_prompt=req.system_prompt,
        include_reasoning_content=req.provider.replay_reasoning,
    )
    tools = to_openai_chat_tools(req.tools)

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
    for key in (
        "seed",
        "top_p",
        "stop",
        "user",
        "response_format",
        "parallel_tool_calls",
    ):
        if key in req.extra:
            kwargs[key] = req.extra[key]

    sdk_response = client.chat.completions.create(**kwargs)
    if not sdk_response.choices:
        raise RuntimeError("openai-chat: response had no choices")
    choice = sdk_response.choices[0]
    message = choice.message

    reasoning_text, reasoning_field = _extract_reasoning(message)
    text = getattr(message, "content", None) or ""
    tool_calls: list[ToolCallBlock] = []
    for tool_call in getattr(message, "tool_calls", None) or []:
        function = getattr(tool_call, "function", None)
        name = getattr(function, "name", "") if function else ""
        args_str = getattr(function, "arguments", "") if function else ""
        try:
            arguments = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            arguments = {RAW_ARGUMENTS_KEY: args_str}
        tool_calls.append(
            ToolCallBlock(
                id=getattr(tool_call, "id", ""), name=name, arguments=arguments
            )
        )

    blocks: list[ContentBlock] = []
    if reasoning_text:
        blocks.append(ThinkingBlock(text=reasoning_text, source_field=reasoning_field))
    if text:
        blocks.append(TextBlock(text=text))
    blocks.extend(tool_calls)

    stop_reason = _map_openai_finish(getattr(choice, "finish_reason", None))
    usage = _openai_chat_usage(getattr(sdk_response, "usage", None))

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
        # The served model the API resolved to (e.g. an alias -> dated
        # snapshot); complete() only back-fills the requested model.
        model=getattr(sdk_response, "model", "") or "",
        raw={"request": capture_request(kwargs), "response": sdk_dump(sdk_response)},
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


DEFAULT_REASONING_FIELD = "reasoning_content"
# Field names we recognize for inbound reasoning, in priority order.
# `reasoning_content` is what OpenAI / DeepSeek-direct emit; `reasoning`
# is what some DeepSeek-via-gateway deployments emit (and then require
# us to echo back symmetrically).
_REASONING_FIELDS: tuple[str, ...] = ("reasoning_content", "reasoning")


def _extract_reasoning(message: Any) -> tuple[str, str | None]:
    """Read the model's reasoning from the SDK message object.

    Returns ``(text, source_field)`` where ``source_field`` is the wire
    field name the value was read from (so the next outbound turn can
    replay it under the same key) or ``None`` when no reasoning was
    present.

    Neither field is part of OpenAI's official Chat schema, so pydantic
    v2 SDK responses expose them through ``__getattr__`` (extra="allow");
    plain attribute access also covers the SimpleNamespace stubs used in
    tests.
    """
    for field_name in _REASONING_FIELDS:
        value = getattr(message, field_name, None)
        if isinstance(value, str) and value:
            return value, field_name
    return "", None


def to_openai_chat_messages(
    messages: Sequence[LLMMessage],
    *,
    system_prompt: str | None = None,
    include_reasoning_content: bool = True,
) -> list[dict[str, Any]]:
    """Render LLMMessages into the OpenAI Chat wire / training shape.

    Shared between the live `openai-chat` adapter and the offline
    training-export module — both want the same JSON shape. The
    `include_reasoning_content` toggle controls whether prior assistant
    `thinking_blocks` are replayed as a sibling `reasoning_content`
    field (DeepSeek / mimo style); the adapter sets it from
    `Provider.replay_reasoning`.
    """
    out: list[dict[str, Any]] = []
    if system_prompt:
        out.append({"role": "system", "content": system_prompt})
    for message in messages:
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
                        "content": text_of(tool_result.content),
                    }
                )
                images = [b for b in tool_result.content if isinstance(b, ImageBlock)]
                if images:
                    visual_blocks.append(
                        {
                            "type": "text",
                            "text": TOOL_RESULT_VISUAL_CAPTION.format(
                                tool_name=tool_result.tool_name
                            ),
                        }
                    )
                    for image in images:
                        visual_blocks.append(_openai_image_block(image))
            if visual_blocks:
                out.append({"role": "user", "content": visual_blocks})
            if any(isinstance(b, (TextBlock, ImageBlock)) for b in message.content):
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
            if include_reasoning_content:
                reasoning = _reasoning_text(message)
                if reasoning:
                    entry[DEFAULT_REASONING_FIELD] = reasoning
            out.append(entry)
    return out


def to_openai_chat_tools(tools: Sequence[LLMTool]) -> list[dict[str, Any]]:
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


def _to_chat_user_content(message: LLMMessage) -> Any:
    parts: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, TextBlock) and block.text:
            parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageBlock):
            parts.append(_openai_image_block(block))
    return parts if parts else ""


def _openai_image_block(block: ImageBlock) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": encode_image_data_url(block.mime_type, block.data)},
    }


def _reasoning_text(message: LLMMessage) -> str:
    return "\n\n".join(block.text for block in message.thinking_blocks if block.text)


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
    # `prompt_tokens` already includes `cached_tokens` as a subset; normalize
    # to the project's additive-cache convention so context_tokens is correct.
    return TokenUsage.from_inclusive_input(
        total_input=int(getattr(usage, "prompt_tokens", 0) or 0),
        output=int(getattr(usage, "completion_tokens", 0) or 0),
        cached_read=cached,
    )


register_adapter("openai-chat", stream)
