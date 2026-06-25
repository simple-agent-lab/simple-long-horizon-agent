"""Layer 3 — Claude Code transcript export.

The sibling of ``openai_export`` for a different consumer: where that module
serializes a run into OpenAI fine-tuning JSONL, this one serializes a run into
the on-disk *session transcript* shape Claude Code writes under
``~/.claude/projects/<cwd>/<sessionId>.jsonl`` — one JSON record per line, each
message linked to its predecessor by ``uuid`` / ``parentUuid``.

Two shape choices follow Claude Code rather than this runtime's own schema:

- **Compaction splits sessions.** Claude Code starts a fresh session file on
  ``/compact``. Each :class:`ContextCompressionEvent` therefore ends the current
  session and opens the next, whose first line is a ``summary`` record carrying
  the fold's summary text and a ``leafUuid`` pointing at the last message of the
  previous session. So one run with one compaction exports *two* ``.jsonl``
  files, isolating pre- and post-compaction history exactly as Claude Code does.

- **Sub-agents become sidechains.** A sub-agent's events (recorded by
  ``task_tool`` into the spawning ``tool_result``'s sidecar) are flattened into
  ``isSidechain: true`` records in the *same* session file as the tool call,
  each with its own ``parentUuid`` chain — instead of staying buried inside a
  tool result. This is the uniform, first-class representation Claude Code uses.

The inner ``message`` field is Anthropic-wire shaped (text / thinking /
tool_use / tool_result blocks), produced here rather than via the anthropic
adapter because that path is keyed to the provider-facing ``LLMMessage`` and
drops the per-message identity this export is built around. Like
``openai_export`` this is a leaf module; nothing else imports it.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..messages import (
    AssistantMessage,
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
    message_from_record,
    text_of,
)
from ..protocols import ContextCompressionEvent, Event, MessageEvent
from .run_trace import RunTrace

# Placeholder Claude Code client version stamped on each record's ``version``
# field. Callers that care can override via ``write_claude_code_sessions``.
DEFAULT_CC_VERSION = "simple-agent-lab"


@dataclass(frozen=True)
class ClaudeCodeSession:
    """One exported session: its id and the ordered records that fill its file."""

    session_id: str
    records: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Inner message payload (Anthropic wire shape)
# ---------------------------------------------------------------------------


def _content_blocks_payload(blocks: Sequence[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, TextBlock):
            if block.text:
                out.append({"type": "text", "text": block.text})
        elif isinstance(block, ThinkingBlock):
            if block.redacted:
                entry: dict[str, Any] = {
                    "type": "redacted_thinking",
                    "data": block.text,
                }
            elif block.text:
                entry = {"type": "thinking", "thinking": block.text}
            else:
                continue
            if block.signature:
                entry["signature"] = block.signature
            out.append(entry)
        elif isinstance(block, ToolCallBlock):
            out.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": dict(block.arguments),
                }
            )
        elif isinstance(block, ToolResultBlock):
            entry = {
                "type": "tool_result",
                "tool_use_id": block.tool_call_id,
                "content": _tool_result_content(block),
            }
            if block.is_error:
                entry["is_error"] = True
            out.append(entry)
        elif isinstance(block, ImageBlock):
            out.append(_image_payload(block))
    return out


def _tool_result_content(block: ToolResultBlock) -> Any:
    """String for text-only results, list of blocks when an image is present."""
    if not any(isinstance(inner, ImageBlock) for inner in block.content):
        return text_of(block.content)
    rendered: list[dict[str, Any]] = []
    for inner in block.content:
        if isinstance(inner, TextBlock) and inner.text:
            rendered.append({"type": "text", "text": inner.text})
        elif isinstance(inner, ImageBlock):
            rendered.append(_image_payload(inner))
    return rendered


def _image_payload(block: ImageBlock) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": block.mime_type or "image/png",
            "data": block.data,
        },
    }


def _message_payload(message: Message) -> dict[str, Any]:
    """The Anthropic-shaped ``message`` field for one transcript record."""
    role = "assistant" if isinstance(message, AssistantMessage) else "user"
    payload: dict[str, Any] = {
        "role": role,
        "content": _content_blocks_payload(message.content),
    }
    if isinstance(message, AssistantMessage):
        payload["type"] = "message"
        if message.model:
            payload["model"] = message.model
        if message.usage is not None:
            payload["usage"] = {
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
                "cache_read_input_tokens": message.usage.cache_read_tokens,
                "cache_creation_input_tokens": message.usage.cache_write_tokens,
            }
    return payload


# ---------------------------------------------------------------------------
# Record envelope
# ---------------------------------------------------------------------------


def _iso(base_epoch: float, elapsed: float) -> str:
    return (
        datetime.fromtimestamp(base_epoch + elapsed, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True)
class _Env:
    """Per-export constants stamped on every record."""

    cwd: str
    version: str
    git_branch: str
    base_epoch: float


def _message_record(
    *,
    message: Message,
    uuid: str,
    parent_uuid: str | None,
    session_id: str,
    is_sidechain: bool,
    elapsed: float,
    env: _Env,
) -> dict[str, Any]:
    payload = _message_payload(message)
    return {
        "parentUuid": parent_uuid,
        "uuid": uuid,
        "sessionId": session_id,
        "timestamp": _iso(env.base_epoch, elapsed),
        "type": "assistant" if payload["role"] == "assistant" else "user",
        "isSidechain": is_sidechain,
        "userType": "external",
        "cwd": env.cwd,
        "version": env.version,
        "gitBranch": env.git_branch,
        "message": payload,
    }


# ---------------------------------------------------------------------------
# Session assembly
# ---------------------------------------------------------------------------


@dataclass
class _MsgEntry:
    message_index: int
    uuid: str
    message: Message
    elapsed: float


def _collect(events: Sequence[Event]) -> tuple[list[_MsgEntry], list[int]]:
    """Walk the event log into (message entries in order, sorted fold boundaries)."""
    entries: list[_MsgEntry] = []
    boundaries: list[int] = []
    message_index = 0
    for event in events:
        if isinstance(event, MessageEvent):
            entries.append(
                _MsgEntry(
                    message_index=message_index,
                    uuid=event.uuid or f"m{message_index}",
                    message=event.message,
                    elapsed=event.elapsed,
                )
            )
            message_index += 1
        elif isinstance(event, ContextCompressionEvent):
            boundaries.append(event.summary_message_index)
    boundaries.sort()
    return entries, boundaries


def _sub_events_of(message: Message) -> list[list[Mapping[str, Any]]]:
    """Sub-agent event lists carried on a tool_result message's sidecar."""
    details = message.sidecar.get("details") if message.sidecar else None
    if not isinstance(details, dict):
        return []
    found: list[list[Mapping[str, Any]]] = []
    for call_details in details.values():
        if not isinstance(call_details, dict):
            continue
        sub_events = call_details.get("sub_events")
        if isinstance(sub_events, list) and sub_events:
            found.append(sub_events)
    return found


def _sidechain_records(
    sub_events: Sequence[Mapping[str, Any]],
    *,
    session_id: str,
    env: _Env,
) -> list[dict[str, Any]]:
    """Flatten one sub-agent's event list into sidechain message records.

    The sub-agent's own ``parentUuid`` chain (first message roots at ``None``),
    marked ``isSidechain: true`` and written into the parent's session file.
    """
    records: list[dict[str, Any]] = []
    parent_uuid: str | None = None
    for raw in sub_events:
        if raw.get("kind") != "message" or "message" not in raw:
            continue
        message = message_from_record(raw["message"])
        uuid = raw.get("uuid") or f"sub.{len(records)}"
        records.append(
            _message_record(
                message=message,
                uuid=uuid,
                parent_uuid=parent_uuid,
                session_id=session_id,
                is_sidechain=True,
                elapsed=float(raw.get("elapsed", 0.0)),
                env=env,
            )
        )
        parent_uuid = uuid
    return records


def claude_code_sessions(
    trace: RunTrace,
    *,
    cwd: str = ".",
    version: str = DEFAULT_CC_VERSION,
    git_branch: str = "",
    base_epoch: float = 0.0,
) -> list[ClaudeCodeSession]:
    """Build the Claude Code session transcripts for a run (pure, no IO).

    Returns one :class:`ClaudeCodeSession` per compaction boundary plus one
    (session 0 holds everything before the first fold). Sub-agent traces are
    inlined as sidechain records in the session that spawned them.
    """
    env = _Env(cwd=cwd, version=version, git_branch=git_branch, base_epoch=base_epoch)
    entries, boundaries = _collect(trace.events)
    boundary_set = set(boundaries)

    # Group every message into its session: the count of fold boundaries at or
    # before its index. A summary message (its index *is* a boundary) heads the
    # session it opens; messages recorded after it share that session.
    sessions: dict[int, list[_MsgEntry]] = {}
    for entry in entries:
        k = bisect_right(boundaries, entry.message_index)
        sessions.setdefault(k, []).append(entry)

    out: list[ClaudeCodeSession] = []
    last_uuid_of_prev: str | None = None
    for k in range(len(boundaries) + 1):
        session_id = trace.trace_id if k == 0 else f"{trace.trace_id}-compact{k}"
        records: list[dict[str, Any]] = []
        parent_uuid: str | None = None
        for entry in sessions.get(k, []):
            if entry.message_index in boundary_set:
                # The fold's summary: a Claude Code `summary` record that bridges
                # back to the previous session, not a chained message turn.
                records.append(
                    {
                        "type": "summary",
                        "summary": text_of(entry.message.content),
                        "leafUuid": last_uuid_of_prev,
                    }
                )
                continue
            records.append(
                _message_record(
                    message=entry.message,
                    uuid=entry.uuid,
                    parent_uuid=parent_uuid,
                    session_id=session_id,
                    is_sidechain=False,
                    elapsed=entry.elapsed,
                    env=env,
                )
            )
            parent_uuid = entry.uuid
            for sub_events in _sub_events_of(entry.message):
                records.extend(
                    _sidechain_records(sub_events, session_id=session_id, env=env)
                )
        # Track the tail of this session's main chain for the next summary's
        # leafUuid (sidechain records don't bridge sessions).
        if parent_uuid is not None:
            last_uuid_of_prev = parent_uuid
        out.append(ClaudeCodeSession(session_id=session_id, records=records))
    return out


def write_claude_code_sessions(
    trace: RunTrace,
    out_dir: str | Path,
    *,
    cwd: str = ".",
    version: str = DEFAULT_CC_VERSION,
    git_branch: str = "",
    base_epoch: float = 0.0,
) -> list[Path]:
    """Write each session to ``<out_dir>/<sessionId>.jsonl`` (one record per line).

    Returns the written paths in session order (session 0 first). Files use the
    compact line-delimited shape Claude Code reads — one JSON object per line.
    """
    sessions = claude_code_sessions(
        trace, cwd=cwd, version=version, git_branch=git_branch, base_epoch=base_epoch
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for session in sessions:
        path = out / f"{session.session_id}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for record in session.records:
                f.write(json.dumps(record, ensure_ascii=False))
                f.write("\n")
        paths.append(path)
    return paths
