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
    extra["thinking"]      : dict          (raw reasoning override; see note)
    extra["output_config"] : dict          (raw adaptive-thinking override)

Prefer the typed ``LLMRequest.reasoning`` / ``Provider.default_reasoning`` knob:
the adapter maps it to the shape each model takes — older models want
``thinking={"type": "enabled", "budget_tokens": N}`` (effort -> budget via
``_EFFORT_BUDGET_TOKENS``), while 4.6+ models reject that with a 400 and instead
take ``thinking={"type": "adaptive"}`` + ``output_config={"effort": ...}``
(picked by ``_is_adaptive_thinking_model``). Reasoning has no single wire shape,
so ``extra["thinking"]``/``extra["output_config"]`` stay as escape hatches that
are passed through verbatim and take precedence over the normalized knob.
"""

from __future__ import annotations

import re
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
from ._spine import emit_response, resolve_effort, resolve_temperature
from ..env import resolve_api_key
from ..stream import register_adapter
from ..types import (
    LLMMessage,
    LLMRequest,
    LLMTool,
    StopReason,
    StreamEvent,
    TokenUsage,
)


DEFAULT_MAX_TOKENS = 4096

# Anthropic rejects a thinking budget below this floor.
ANTHROPIC_MIN_THINKING_BUDGET = 1024
# Effort -> thinking budget for models that take ``budget_tokens`` (pre-4.6).
# Values mirror LiteLLM's defaults; each is clamped up to the floor above.
_EFFORT_BUDGET_TOKENS: dict[str, int] = {
    "minimal": 1024,
    "low": 1024,
    "medium": 2048,
    "high": 4096,
    "xhigh": 8192,
}
# Model ids carry the version as ``...-<major>-<minor>`` (e.g. claude-opus-4-7)
# or ``claude-3-7-sonnet``; the first number pair is the family version.
_MODEL_VERSION_RE = re.compile(r"(\d+)-(\d+)")


def _is_adaptive_thinking_model(model: str) -> bool:
    """Claude 4.6+ takes ``thinking={"type": "adaptive"}`` + ``output_config``.

    Older models take ``thinking={"type": "enabled", "budget_tokens": N}`` and
    reject adaptive with a 400. Version is read off the model id; an
    unparseable id is treated as old (the conservative, widely-accepted shape).
    """
    match = _MODEL_VERSION_RE.search(model)
    if not match:
        return False
    return (int(match.group(1)), int(match.group(2))) >= (4, 6)


def _reasoning_kwargs(effort: str, model: str) -> dict[str, Any]:
    """Translate a normalized effort to the wire shape this model expects."""
    if _is_adaptive_thinking_model(model):
        return {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort},
        }
    budget = max(ANTHROPIC_MIN_THINKING_BUDGET, _EFFORT_BUDGET_TOKENS[effort])
    return {"thinking": {"type": "enabled", "budget_tokens": budget}}


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
    temperature = resolve_temperature(req)
    if temperature is not None:
        kwargs["temperature"] = temperature
    if req.timeout_seconds:
        kwargs["timeout"] = req.timeout_seconds
    # Translate the normalized reasoning knob to the shape this model takes.
    # A raw extra["thinking"]/extra["output_config"] (below) wins over it.
    effort = resolve_effort(req)
    if effort:
        kwargs.update(_reasoning_kwargs(effort, req.provider.model))
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

    yield from emit_response(
        blocks,
        stop_reason=stop_reason,
        usage=usage,
        sdk_response=sdk_response,
        request_kwargs=kwargs,
    )


def _api_key(req: LLMRequest) -> str | None:
    # The Anthropic SDK accepts a None key for keyless endpoints (placeholder is
    # None, unlike the OpenAI adapters). Shared resolver lives in `llm.env`.
    return resolve_api_key(req.provider, placeholder=None)


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
