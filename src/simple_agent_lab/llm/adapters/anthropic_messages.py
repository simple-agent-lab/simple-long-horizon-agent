"""Anthropic Messages API adapter.

Uses the official `anthropic` SDK. Blocking-only: calls
`client.messages.create(...)` and emits a one-shot `text_delta`,
`thinking_delta`, `tool_call_start`/`tool_call_complete` pair per tool call,
`usage_update`, and a final `done`. Token-by-token streaming can be added
later by switching to `client.messages.stream(...)`.

The SDK is imported lazily inside `stream()` so module registration stays cheap.
Calling the adapter without the SDK raises a clear error.

Provider config:

    Provider(
        id="claude",
        api="anthropic-messages",
        model="claude-opus-4-7",
        api_key_env="ANTHROPIC_API_KEY",
        base_url=None,                    # optional, Bedrock proxy etc.
    )

Pass-through request options via `LLMRequest.extra`:

    extra["extra_headers"] : dict          (e.g. beta headers)
    extra["extra_body"]    : dict
    extra["metadata"]      : dict
    extra["top_p"]         : float
    extra["top_k"]         : int
    extra["stop_sequences"]: list[str]
    extra["thinking"]      : dict          (reasoning depth; see note)
    extra["output_config"] : dict          (adaptive-thinking effort)

Reasoning has no single shape; it is passed through verbatim so the caller
matches it to the model. Older models take
``thinking={"type": "enabled", "budget_tokens": N}``; newer ones (Opus 4.7+)
reject that with a 400 and instead take ``thinking={"type": "adaptive"}``
plus ``output_config={"effort": "high"}``.
"""

from __future__ import annotations

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
)
from . import capture_request, sdk_dump
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


DEFAULT_MAX_TOKENS = 4096


def stream(req: LLMRequest) -> Iterator[StreamEvent]:
    try:
        from anthropic import Anthropic
    except ImportError as exc:  # pragma: no cover - import error path
        raise RuntimeError(
            "anthropic-messages adapter requires the 'anthropic' package. "
            "Install the package dependencies with: uv sync "
            "or: pip install anthropic"
        ) from exc

    client = Anthropic(
        api_key=_api_key(req),
        base_url=req.provider.base_url,
    )

    system, messages = _to_anthropic_messages(req)
    tools = _to_anthropic_tools(req.tools)
    max_tokens = req.max_tokens or req.provider.default_max_tokens or DEFAULT_MAX_TOKENS

    kwargs: dict[str, Any] = {
        "model": req.provider.model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = tools
    temperature = (
        req.temperature
        if req.temperature is not None
        else req.provider.default_temperature
    )
    if temperature is not None:
        kwargs["temperature"] = temperature
    if req.timeout_seconds:
        kwargs["timeout"] = req.timeout_seconds
    for key in (
        "extra_headers",
        "extra_body",
        "metadata",
        "top_p",
        "top_k",
        "stop_sequences",
        "thinking",
        "output_config",
    ):
        if key in req.extra:
            kwargs[key] = req.extra[key]

    sdk_response = client.messages.create(**kwargs)

    # Preserve the wire order: replay rejects assistant messages where
    # thinking does not lead text and tool_use.
    blocks: list[ContentBlock] = []
    for block in getattr(sdk_response, "content", []) or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            text = getattr(block, "text", "") or ""
            if text:
                blocks.append(TextBlock(text=text))
        elif btype == "thinking":
            blocks.append(
                ThinkingBlock(
                    text=getattr(block, "thinking", "") or "",
                    signature=getattr(block, "signature", None),
                )
            )
        elif btype == "redacted_thinking":
            blocks.append(
                ThinkingBlock(
                    text=getattr(block, "data", "") or "",
                    signature=getattr(block, "signature", None),
                    redacted=True,
                )
            )
        elif btype == "tool_use":
            blocks.append(
                ToolCallBlock(
                    id=getattr(block, "id", ""),
                    name=getattr(block, "name", ""),
                    arguments=dict(getattr(block, "input", {}) or {}),
                )
            )

    stop_reason = _map_anthropic_stop(getattr(sdk_response, "stop_reason", None))
    usage = _anthropic_usage(getattr(sdk_response, "usage", None))

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
        return None
    api_key = os.environ.get(env)
    if not api_key:
        raise RuntimeError(
            f"Provider {req.provider.id!r} requires env var {env!r}; not set."
        )
    return api_key


def _to_anthropic_messages(req: LLMRequest) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    if req.system_prompt:
        system_parts.append(req.system_prompt)

    messages: list[dict[str, Any]] = []
    for message in req.messages:
        if message.role == "system":
            text = text_of(message.content)
            if text:
                system_parts.append(text)
            continue
        if message.role == "user" and message.tool_results:
            wire_blocks: list[dict[str, Any]] = []
            for tool_result in message.tool_results:
                inner = _anthropic_tool_result_content(tool_result)
                wire_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_result.tool_call_id,
                        "content": inner,
                    }
                )
            # Any TextBlock / ImageBlock siblings ride along in the same wire user message.
            for block in message.content:
                if isinstance(block, TextBlock) and block.text:
                    wire_blocks.append({"type": "text", "text": block.text})
                elif isinstance(block, ImageBlock):
                    wire_blocks.append(_anthropic_image_block(block))
            _apply_message_extra(wire_blocks, message)
            messages.append({"role": "user", "content": wire_blocks})
            continue
        if message.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if req.provider.replay_reasoning:
                for thinking_block in message.thinking_blocks:
                    if thinking_block.redacted:
                        entry: dict[str, Any] = {
                            "type": "redacted_thinking",
                            "data": thinking_block.text,
                        }
                        if thinking_block.signature:
                            entry["signature"] = thinking_block.signature
                        blocks.append(entry)
                    elif thinking_block.text:
                        entry = {"type": "thinking", "thinking": thinking_block.text}
                        if thinking_block.signature:
                            entry["signature"] = thinking_block.signature
                        blocks.append(entry)
            text = text_of(message.content)
            if text:
                blocks.append({"type": "text", "text": text})
            for tool_call in message.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "input": dict(tool_call.arguments),
                    }
                )
            if blocks:
                _apply_message_extra(blocks, message)
                messages.append({"role": "assistant", "content": blocks})
            continue
        # user
        content = _to_anthropic_user_content(message)
        if content:
            _apply_message_extra(content, message)
            messages.append({"role": "user", "content": content})

    system = "\n\n".join(part for part in system_parts if part) or None
    return system, messages


def _apply_message_extra(
    wire_blocks: list[dict[str, Any]], message: LLMMessage
) -> None:
    """Translate `LLMMessage.extra` provider-namespaced hints to wire shape.

    Currently understood: `anthropic.cache_breakpoint=True` → attach
    `cache_control: {"type": "ephemeral"}` to the last wire block of
    this message (Anthropic's documented anchor pattern).
    """
    if not message.extra or not wire_blocks:
        return
    if message.extra.get("anthropic.cache_breakpoint"):
        wire_blocks[-1]["cache_control"] = {"type": "ephemeral"}


def _to_anthropic_user_content(message: LLMMessage) -> Any:
    blocks: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, TextBlock) and block.text:
            blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageBlock):
            blocks.append(_anthropic_image_block(block))
    return blocks


def _anthropic_image_block(block: ImageBlock) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": block.mime_type or "image/png",
            "data": block.data,
        },
    }


def _anthropic_tool_result_content(block: ToolResultBlock) -> Any:
    """Render the inner content of a tool_result block for Anthropic wire.

    Anthropic accepts either a plain string (text-only) or a list of
    `text` / `image` content blocks. We use the string form when there
    are no images so simple cases stay compact; multimodal results get
    the list form.
    """
    has_image = any(isinstance(b, ImageBlock) for b in block.content)
    if not has_image:
        return text_of(block.content)
    rendered: list[dict[str, Any]] = []
    for inner in block.content:
        if isinstance(inner, TextBlock) and inner.text:
            rendered.append({"type": "text", "text": inner.text})
        elif isinstance(inner, ImageBlock):
            rendered.append(_anthropic_image_block(inner))
    return rendered


def _to_anthropic_tools(tools: list[LLMTool]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }
        for tool in tools
    ]


_ANTHROPIC_STOP_MAP: dict[str, StopReason] = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "end_turn",
    "refusal": "error",
    "model_context_window_exceeded": "max_tokens",
}


def _map_anthropic_stop(raw: str | None) -> StopReason:
    if raw is None:
        return "end_turn"
    return _ANTHROPIC_STOP_MAP.get(raw, "end_turn")


def _anthropic_usage(usage: Any) -> TokenUsage:
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        cache_write_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
    )


register_adapter("anthropic-messages", stream)
