"""Shared translation spine for the provider adapters.

The provider seam (`register_adapter` / `stream`) is intentionally tiny, but
the three real adapters — `openai_chat`, `openai_responses`,
`anthropic_messages` — used to each re-implement the same cross-provider
concerns. Those concerns live here once, behind the seam, so an adapter
carries only what is genuinely provider-specific (its wire field names and
its SDK call):

  * `emit_response` — the identical tail of every `stream()`: replay the
    response blocks as one-shot stream deltas, emit the usage update, build
    the `LLMResponse`, and yield the terminal `done` event.
  * `parse_tool_arguments` — recover a tool call's arguments from the JSON
    string both OpenAI APIs hand back, falling back to a raw-string capture
    when the model emits malformed JSON.
  * `resolve_temperature` / `resolve_effort` — fold the per-request value with
    its `Provider` default, the same way for every adapter.
  * `openai_usage` — normalize an OpenAI-shape usage object (Chat or
    Responses) into the project's additive-cache `TokenUsage`; only the SDK
    field names differ between the two APIs.

What is *not* here: message→wire rendering and stop-reason mapping. Those
differ enough per provider that a shared version would be mostly branching,
so each adapter still owns them.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from ...messages import ContentBlock, TextBlock, ThinkingBlock, ToolCallBlock
from ..types import (
    RAW_ARGUMENTS_KEY,
    LLMRequest,
    LLMResponse,
    StopReason,
    StreamEvent,
    TokenUsage,
)


# Caption emitted in the adjacent user message when a provider's wire shape
# can't carry images inside its tool-result entry (OpenAI Chat / Responses).
# Shared so all OpenAI-shape adapters surface the visual with the same hint.
TOOL_RESULT_VISUAL_CAPTION = "Visual output from {tool_name}:"


def capture_request(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Snapshot a provider request for ``LLMResponse.raw``.

    Returns a shallow copy of the outbound kwargs so every turn's
    ``raw["request"]`` preserves the full request body including the
    messages / input history.
    """
    return dict(kwargs)


def sdk_dump(value: Any) -> Any:
    """Best-effort serialization snapshot of an SDK response object.

    pydantic v2 SDKs (openai, anthropic) expose `model_dump()`. Fall
    back to the raw object when no dump method exists, which is enough
    for `print_trace(raw=True)` to render it via `repr`.
    """
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:
            pass
    return value


def parse_tool_arguments(args_str: str | None) -> dict[str, Any]:
    """Decode a tool call's JSON argument string (OpenAI Chat / Responses).

    Both OpenAI APIs return tool-call arguments as a JSON string. A model can
    emit malformed JSON; rather than fail the whole turn we stash the raw text
    under ``RAW_ARGUMENTS_KEY`` so the caller can surface it and the model can
    self-correct next turn — the same recovery both OpenAI adapters need.
    """
    if not args_str:
        return {}
    try:
        return json.loads(args_str)
    except json.JSONDecodeError:
        return {RAW_ARGUMENTS_KEY: args_str}


def resolve_temperature(req: LLMRequest) -> float | None:
    """Per-request temperature, falling back to the provider default."""
    if req.temperature is not None:
        return req.temperature
    return req.provider.default_temperature


def resolve_effort(req: LLMRequest) -> str | None:
    """Normalized reasoning effort, falling back to the provider default.

    The wire shape differs per provider (top-level ``reasoning_effort`` for
    Chat, ``reasoning={"effort": ...}`` for Responses, a thinking budget /
    adaptive shape for Anthropic), so each adapter still maps the returned
    value itself — but the precedence rule is shared.
    """
    return req.reasoning or req.provider.default_reasoning


def openai_usage(
    usage: Any,
    *,
    input_field: str,
    output_field: str,
    details_field: str,
) -> TokenUsage:
    """Normalize an OpenAI-shape usage object into the project's `TokenUsage`.

    Chat and Responses report the same numbers under different field names
    (``prompt_tokens`` / ``prompt_tokens_details`` vs ``input_tokens`` /
    ``input_tokens_details``). Both report the input count *including* the
    cached portion, so we route through `TokenUsage.from_inclusive_input` to
    keep the project's additive-cache convention (so `context_tokens` doesn't
    double-count the cache).
    """
    if usage is None:
        return TokenUsage()
    details = getattr(usage, details_field, None)
    cached = (
        int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
    )
    return TokenUsage.from_inclusive_input(
        total_input=int(getattr(usage, input_field, 0) or 0),
        output=int(getattr(usage, output_field, 0) or 0),
        cached_read=cached,
    )


def emit_response(
    blocks: list[ContentBlock],
    *,
    stop_reason: StopReason,
    usage: TokenUsage,
    sdk_response: Any,
    request_kwargs: dict[str, Any],
) -> Iterator[StreamEvent]:
    """Replay standardized response blocks as the adapter's stream tail.

    Every adapter ends the same way: emit a one-shot delta per content block
    (thinking, then text, then a start/complete pair per tool call — the order
    is whatever the adapter put in `blocks`), then the usage update, then the
    `LLMResponse` wrapped in the terminal `done` event. `blocks` becomes
    `LLMResponse.content` verbatim, so the adapter controls ordering by how it
    builds the list.
    """
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
        raw={
            "request": capture_request(request_kwargs),
            "response": sdk_dump(sdk_response),
        },
    )
    yield StreamEvent(kind="done", payload={"response": response})
