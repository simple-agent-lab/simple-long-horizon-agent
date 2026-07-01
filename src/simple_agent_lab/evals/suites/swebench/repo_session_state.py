"""SWE-bench Pro repo-session continuation artifacts.

Repo-session experiments run one fresh instance container per task, so the
agent's carry-over context has to cross the container boundary as data. This
module serializes the active runtime context into a compact JSON payload and
rebuilds a ``State`` from that payload inside the next container.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from simple_agent_lab.messages import (
    AssistantMessage,
    ContentBlock,
    ContentInput,
    ImageBlock,
    Message,
    MessageKind,
    MessageSidecar,
    TextBlock,
    ThinkingBlock,
    TokenUsage,
    ToolCallBlock,
    ToolResultBlock,
    assistant_message,
    runtime_message,
    user_message,
)
from simple_agent_lab.protocols import ContextCompressionEvent
from simple_agent_lab.state import State
from simple_agent_lab.trace.jsonl import json_safe

SESSION_STATE_INPUT_KEY = "input/session_state.json"
SESSION_STATE_OUTPUT_KEY = "out/session_state.json"
SESSION_CONFIG_KEY = "input/repo_session_config.json"
SESSION_STATE_SCHEMA = "simple-agent-lab.swebench-pro-repo-session-state.v1"


def start_repo_session_state(repo: str, *, agent_name: str) -> State:
    """Create the persistent transcript seed for one repository session."""

    task = (
        f"SWE-bench Pro repo session for {repo}. Solve instances for this "
        "repository in commit-time order. Carry useful context across tasks, "
        "but each instance's patch must address only the current problem."
    )
    state = State(task)
    state.data["repo_session"] = {"repo": repo, "agent_name": agent_name}
    return state


def append_repo_session_task(
    state: State,
    *,
    agent_name: str,
    instance_id: str,
    task: str,
) -> None:
    """Append one SWE-bench instance prompt to an existing repo session."""

    state.send(
        "task",
        "user",
        agent_name,
        task,
        sidecar={"details": {"swebench": {"instance_id": instance_id}}},
    )


def state_to_session_payload(state: State) -> dict[str, Any]:
    """Return a JSON-safe continuation payload for the state's active context."""

    active_items = state.active_context_items()
    return {
        "schema": SESSION_STATE_SCHEMA,
        "task": _content_input_to_record(state.task),
        "messages": [_message_to_record(message) for _, message in active_items],
        "active_context_indices": list(range(len(active_items))),
        "data": json_safe(state.data),
        "meta": {
            "source_message_count": len(state.messages),
            "source_event_count": len(state.events),
        },
    }


def state_from_session_payload(payload: Mapping[str, Any]) -> State:
    """Rebuild a ``State`` from ``state_to_session_payload`` data."""

    schema = str(payload.get("schema") or "")
    if schema and schema != SESSION_STATE_SCHEMA:
        raise ValueError(f"Unsupported repo session state schema: {schema!r}")

    state = State(task=_content_input_from_record(payload.get("task", "")))
    data = payload.get("data")
    if isinstance(data, Mapping):
        state.data.update(dict(data))

    messages = payload.get("messages", [])
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ValueError("repo session payload 'messages' must be a list")
    for record in messages:
        if not isinstance(record, Mapping):
            raise ValueError("repo session message records must be objects")
        state.record(_message_from_record(record))

    indices = payload.get("active_context_indices")
    if indices is None:
        indices = list(range(len(state.messages)))
    if not isinstance(indices, Sequence) or isinstance(indices, (str, bytes)):
        raise ValueError("repo session payload 'active_context_indices' must be a list")
    active = [int(index) for index in indices]
    state.record_event(
        ContextCompressionEvent(
            agent=str(state.data.get("repo_session", {}).get("agent_name") or ""),
            summary_message_index=active[-1] if active else -1,
            compressed_message_indices=[],
            active_context_indices=active,
            before_tokens=0,
            after_tokens=0,
            strategy="repo-session-state-restore",
        )
    )
    return state


def _content_input_to_record(content: ContentInput) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    return [_block_to_record(block) for block in content]


def _content_input_from_record(value: Any) -> ContentInput:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(_block_from_record(block) for block in value)
    return str(value or "")


def _message_to_record(message: Message) -> dict[str, Any]:
    record = {
        "role": message.role,
        "sender": message.sender,
        "target": message.target,
        "kind": message.kind,
        "content": [_block_to_record(block) for block in message.content],
        "sidecar": json_safe(message.sidecar),
    }
    if isinstance(message, AssistantMessage):
        record["model"] = message.model
        if message.usage is not None:
            record["usage"] = {
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
                "cache_read_tokens": message.usage.cache_read_tokens,
                "cache_write_tokens": message.usage.cache_write_tokens,
            }
    return record


def _message_from_record(record: Mapping[str, Any]) -> Message:
    role = str(record.get("role") or "")
    content = tuple(_block_from_record(block) for block in record.get("content", []))
    sender = str(record.get("sender") or role or "agent")
    target = str(record.get("target") or "all")
    kind = cast(MessageKind, str(record.get("kind") or "message"))
    sidecar = cast(MessageSidecar, _mapping(record.get("sidecar")))
    if role == "user":
        return user_message(
            content, sender=sender, target=target, kind=kind, sidecar=sidecar
        )
    if role == "system":
        return runtime_message(
            content, sender=sender, target=target, kind=kind, sidecar=sidecar
        )
    if role == "assistant":
        usage = record.get("usage")
        return assistant_message(
            content,
            sender=sender,
            target=target,
            kind=kind,
            sidecar=sidecar,
            usage=_usage_from_record(usage) if isinstance(usage, Mapping) else None,
            model=str(record.get("model") or ""),
        )
    raise ValueError(f"Unsupported repo session message role: {role!r}")


def _block_to_record(block: ContentBlock) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"kind": "text", "text": block.text}
    if isinstance(block, ImageBlock):
        return {"kind": "image", "data": block.data, "mime_type": block.mime_type}
    if isinstance(block, ThinkingBlock):
        return {
            "kind": "thinking",
            "text": block.text,
            "signature": block.signature,
            "redacted": block.redacted,
            "source_field": block.source_field,
        }
    if isinstance(block, ToolCallBlock):
        return {
            "kind": "tool_call",
            "id": block.id,
            "name": block.name,
            "arguments": json_safe(dict(block.arguments)),
        }
    if isinstance(block, ToolResultBlock):
        return {
            "kind": "tool_result",
            "tool_call_id": block.tool_call_id,
            "tool_name": block.tool_name,
            "content": [_block_to_record(item) for item in block.content],
            "is_error": block.is_error,
        }
    raise TypeError(f"Unsupported content block: {type(block)!r}")


def _block_from_record(record: Any) -> ContentBlock:
    if not isinstance(record, Mapping):
        raise ValueError("content block records must be objects")
    kind = str(record.get("kind") or "")
    if kind == "text":
        return TextBlock(str(record.get("text") or ""))
    if kind == "image":
        return ImageBlock(
            data=str(record.get("data") or ""),
            mime_type=str(record.get("mime_type") or "image/png"),
        )
    if kind == "thinking":
        return ThinkingBlock(
            text=str(record.get("text") or ""),
            signature=(
                str(record["signature"])
                if record.get("signature") is not None
                else None
            ),
            redacted=bool(record.get("redacted", False)),
            source_field=(
                str(record["source_field"])
                if record.get("source_field") is not None
                else None
            ),
        )
    if kind == "tool_call":
        return ToolCallBlock(
            id=str(record.get("id") or ""),
            name=str(record.get("name") or ""),
            arguments=_mapping(record.get("arguments")),
        )
    if kind == "tool_result":
        return ToolResultBlock(
            tool_call_id=str(record.get("tool_call_id") or ""),
            tool_name=str(record.get("tool_name") or ""),
            content=tuple(
                block
                for block in (
                    _block_from_record(item) for item in record.get("content", [])
                )
                if isinstance(block, (TextBlock, ImageBlock))
            ),
            is_error=bool(record.get("is_error", False)),
        )
    raise ValueError(f"Unsupported content block kind: {kind!r}")


def _usage_from_record(record: Mapping[str, Any]) -> TokenUsage:
    return TokenUsage(
        input_tokens=int(record.get("input_tokens", 0) or 0),
        output_tokens=int(record.get("output_tokens", 0) or 0),
        cache_read_tokens=int(record.get("cache_read_tokens", 0) or 0),
        cache_write_tokens=int(record.get("cache_write_tokens", 0) or 0),
    )


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}
