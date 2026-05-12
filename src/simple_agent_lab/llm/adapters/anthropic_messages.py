"""Anthropic Messages API adapter.

Uses the official `anthropic` SDK. Blocking-only: calls
`client.messages.create(...)` and emits a one-shot `text_delta`,
`thinking_delta`, `tool_call_start`/`tool_call_complete` pair per tool call,
`usage_update`, and a final `done`. Token-by-token streaming can be added
later by switching to `client.messages.stream(...)`.

The SDK is imported lazily inside `stream()` so installing
`simple-agent-lab` without `[anthropic]` still lets the module register.
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
    text_of,
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


DEFAULT_MAX_TOKENS = 4096


def stream(req: LLMRequest) -> Iterator[StreamEvent]:
    try:
        from anthropic import Anthropic
    except ImportError as exc:  # pragma: no cover - import error path
        raise RuntimeError(
            "anthropic-messages adapter requires the 'anthropic' package. "
            "Install with: uv pip install 'simple-agent-lab[anthropic]' "
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
    for key in ("extra_headers", "extra_body", "metadata", "top_p", "top_k", "stop_sequences"):
        if key in req.extra:
            kwargs[key] = req.extra[key]

    raw = client.messages.create(**kwargs)

    # Preserve the wire order: replay rejects assistant messages where
    # thinking does not lead text and tool_use.
    blocks: list[ContentBlock] = []
    for block in getattr(raw, "content", []) or []:
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

    stop_reason = _map_anthropic_stop(getattr(raw, "stop_reason", None))
    usage = _anthropic_usage(getattr(raw, "usage", None))

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
        raw={
            "provider": "anthropic-messages",
            "model": req.provider.model,
            "id": getattr(raw, "id", None),
            "stop_reason": getattr(raw, "stop_reason", None),
        },
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
        if message.role == "tool_result":
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.tool_call_id or "",
                            "content": text_of(message.content),
                        }
                    ],
                }
            )
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
                messages.append({"role": "assistant", "content": blocks})
            continue
        # user
        content = _to_anthropic_user_content(message)
        if content:
            messages.append({"role": "user", "content": content})

    system = "\n\n".join(part for part in system_parts if part) or None
    return system, messages


def _to_anthropic_user_content(message: LLMMessage) -> Any:
    blocks: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, TextBlock) and block.text:
            blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageBlock):
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block.mime_type or "image/png",
                        "data": block.data,
                    },
                }
            )
    return blocks


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
