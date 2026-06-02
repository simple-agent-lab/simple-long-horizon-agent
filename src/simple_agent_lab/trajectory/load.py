"""Rebuild a runtime `State` from a serialized trace record.

`trace_record` (see `run_trace.py`) is one-way by default — it flattens the
event log through `json_safe`/`asdict` for export. Replay needs the inverse:
turn a persisted ``trajectory.jsonl`` record back into a `State` so a run can
be forked and resumed (see `simple_agent_lab.replay`).

Only the **event log** is reconstructed. `State.__post_init__` rebuilds the
message list and the active-context snapshot from the events, so the record's
top-level ``messages`` mirror is not consumed here — the events are the
source of truth, exactly as they are at runtime.

Heterogeneous trace payloads that are typed `Any`/`dict` at runtime
(`ModelRequestEvent.context_view` / `tools` / `llm_payload`) are kept as the
plain JSON structures they serialized to; nothing reads them as typed objects.
"""

from __future__ import annotations

from typing import Any, cast

from ..messages import (
    AssistantMessage,
    ContentBlock,
    ImageBlock,
    Message,
    MessageSidecar,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    TokenUsage,
    ToolCallBlock,
    ToolResultBlock,
    UserMessage,
    VisibleBlock,
)
from ..protocols import (
    AgentEndEvent,
    AgentStartEvent,
    ContextCompressionEvent,
    Event,
    MessageEvent,
    ModelRequestEvent,
    ModelResponseEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from ..state import State
from ..tools import ToolResult


def _token_usage_from_dict(data: Any) -> TokenUsage | None:
    if not data:
        return None
    return TokenUsage(
        input_tokens=data.get("input_tokens", 0),
        output_tokens=data.get("output_tokens", 0),
        cache_read_tokens=data.get("cache_read_tokens", 0),
        cache_write_tokens=data.get("cache_write_tokens", 0),
    )


def _block_from_dict(data: dict[str, Any]) -> ContentBlock:
    kind = data.get("kind")
    if kind == "text":
        return TextBlock(text=data.get("text", ""))
    if kind == "image":
        return ImageBlock(data=data["data"], mime_type=data["mime_type"])
    if kind == "thinking":
        return ThinkingBlock(
            text=data.get("text", ""),
            signature=data.get("signature"),
            redacted=data.get("redacted", False),
            source_field=data.get("source_field"),
        )
    if kind == "tool_call":
        return ToolCallBlock(
            id=data["id"],
            name=data["name"],
            arguments=dict(data.get("arguments", {})),
        )
    if kind == "tool_result":
        return ToolResultBlock(
            tool_call_id=data["tool_call_id"],
            tool_name=data["tool_name"],
            content=_visible_content(data.get("content")),
            is_error=data.get("is_error", False),
        )
    raise ValueError(f"unknown content block kind {kind!r}")


def _visible_block_from_dict(data: dict[str, Any]) -> VisibleBlock:
    """A tool_result block only ever carries visible (text/image) blocks."""
    block = _block_from_dict(data)
    if not isinstance(block, (TextBlock, ImageBlock)):
        raise ValueError(f"tool_result content cannot hold a {block.kind!r} block")
    return block


def _visible_content(blocks: Any) -> tuple[VisibleBlock, ...]:
    return tuple(_visible_block_from_dict(b) for b in (blocks or ()))


def _content_from_list(blocks: Any) -> tuple[ContentBlock, ...]:
    return tuple(_block_from_dict(b) for b in (blocks or ()))


def _sidecar_from_dict(data: Any) -> MessageSidecar:
    return cast(MessageSidecar, dict(data or {}))


def message_from_dict(data: dict[str, Any]) -> Message:
    """Reconstruct a `Message` from its `asdict` record (dispatch on ``role``)."""
    role = data.get("role")
    content = _content_from_list(data.get("content"))
    sender = data.get("sender", "")
    target = data.get("target", "all")
    kind = data.get("kind", "message")
    sidecar = _sidecar_from_dict(data.get("sidecar"))

    if role == "user":
        return UserMessage(
            content=content, sender=sender, target=target, kind=kind, sidecar=sidecar
        )
    if role == "system":
        return SystemMessage(
            content=content, sender=sender, target=target, kind=kind, sidecar=sidecar
        )
    if role == "assistant":
        return AssistantMessage(
            content=content,
            sender=sender,
            target=target,
            kind=kind,
            usage=_token_usage_from_dict(data.get("usage")),
            model=data.get("model", ""),
            sidecar=sidecar,
        )
    raise ValueError(f"unknown message role {role!r}")


def _tool_result_from_dict(data: dict[str, Any]) -> ToolResult:
    return ToolResult(
        content=_visible_content(data.get("content")),
        details=data.get("details"),
        is_error=data.get("is_error", False),
        terminate=data.get("terminate", False),
    )


def event_from_dict(data: dict[str, Any]) -> Event:
    """Reconstruct one `Event` from its `asdict` record (dispatch on ``kind``)."""
    kind = data.get("kind")
    # `index`/`elapsed` are stamped on every event; preserve the originals so
    # the rebuilt prefix keeps its chronological metadata.
    stamp = {"index": data.get("index", -1), "elapsed": data.get("elapsed", 0.0)}

    if kind == "message":
        return MessageEvent(message=message_from_dict(data["message"]), **stamp)
    if kind == "agent_start":
        return AgentStartEvent(**stamp)
    if kind == "agent_end":
        return AgentEndEvent(reason=data["reason"], **stamp)
    if kind == "turn_start":
        return TurnStartEvent(agent=data["agent"], **stamp)
    if kind == "turn_end":
        return TurnEndEvent(
            agent=data["agent"], terminated=data.get("terminated", False), **stamp
        )
    if kind == "model_request":
        return ModelRequestEvent(
            agent=data["agent"],
            visible_count=data["visible_count"],
            llm_message_count=data["llm_message_count"],
            context_view=dict(data.get("context_view", {})),
            tools=list(data.get("tools", [])),
            llm_payload=list(data.get("llm_payload", [])),
            **stamp,
        )
    if kind == "model_response":
        return ModelResponseEvent(
            agent=data["agent"],
            output_kind=data["output_kind"],
            target=data["target"],
            tool_call_count=data["tool_call_count"],
            usage=_token_usage_from_dict(data.get("usage")),
            model=data.get("model", ""),
            **stamp,
        )
    if kind == "context_compression":
        return ContextCompressionEvent(
            agent=data["agent"],
            summary_message_index=data["summary_message_index"],
            compressed_message_indices=list(data.get("compressed_message_indices", [])),
            active_context_indices=list(data.get("active_context_indices", [])),
            before_tokens=data["before_tokens"],
            after_tokens=data["after_tokens"],
            **stamp,
        )
    if kind == "tool_execution_start":
        return ToolExecutionStartEvent(
            tool_call_id=data["tool_call_id"], tool_name=data["tool_name"], **stamp
        )
    if kind == "tool_execution_update":
        return ToolExecutionUpdateEvent(
            tool_call_id=data["tool_call_id"],
            tool_name=data["tool_name"],
            partial=_tool_result_from_dict(data["partial"]),
            **stamp,
        )
    if kind == "tool_execution_end":
        return ToolExecutionEndEvent(
            tool_call_id=data["tool_call_id"],
            tool_name=data["tool_name"],
            is_error=data["is_error"],
            terminate=data["terminate"],
            **stamp,
        )
    raise ValueError(f"unknown event kind {kind!r}")


def state_from_trace_record(record: dict[str, Any]) -> State:
    """Rebuild a `State` from a `trace_record` dict.

    The returned state carries the full event log; its message list and
    active-context snapshot are rebuilt from those events by
    `State.__post_init__`. `task` is the record's text summary (the full
    task message survives in the events), which is enough for fork/resume —
    `run()` reads messages, not `state.task`.
    """
    events = [event_from_dict(e) for e in record.get("events", [])]
    return State(task=record.get("task", ""), events=events)
