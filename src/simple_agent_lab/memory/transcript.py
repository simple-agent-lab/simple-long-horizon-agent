"""Transcript text helpers for memory backends.

These helpers extract only project-owned, model-visible message content. They
never read provider raw payloads or debug-only sidecars.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from simple_agent_lab.messages import (
    AssistantMessage,
    ImageBlock,
    Message,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    message_text,
)
from simple_agent_lab.state import State


def extract_memory_text(message: Message) -> str:
    """Flatten one runtime message into text suitable for memory indexing."""

    parts: list[str] = []
    for block in message.content:
        if isinstance(block, TextBlock) and block.text:
            parts.append(block.text)
        elif isinstance(block, ImageBlock):
            parts.append(f"[image: {block.mime_type or 'image'}]")
        elif isinstance(block, ToolCallBlock):
            parts.append(
                json.dumps(
                    {
                        "tool": block.name,
                        "args": dict(block.arguments),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        elif isinstance(block, ToolResultBlock):
            inner = _visible_blocks_text(block.content)
            parts.append(f"tool:{block.tool_name}\n{inner}".strip())
    return "\n".join(part for part in parts if part).strip()


def render_transcript_markdown(messages: Iterable[Message]) -> str:
    """Render a readable Markdown transcript for filesystem memory."""

    lines = ["# Trajectory", ""]
    for index, message in enumerate(messages):
        text = extract_memory_text(message)
        if not text:
            continue
        route = f"{message.sender} -> {message.target}"
        lines.extend(
            [
                f"## {index}. {message.role} ({message.kind}, {route})",
                "",
                text,
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def first_user_text(messages: Iterable[Message], *, fallback: str = "") -> str:
    """Return the first user/task text in a transcript."""

    for message in messages:
        if message.role == "user" or message.kind == "task":
            text = extract_memory_text(message) or message_text(message)
            if text:
                return text
    return fallback


def final_submission_from_state(state: State) -> str:
    """Best-effort final artifact text from generic State metadata."""

    for key in ("model_patch", "patch", "submission"):
        value = state.data.get(key)
        if isinstance(value, str) and value:
            return value

    for message in reversed(state.messages):
        sidecar = getattr(message, "sidecar", None)
        if not isinstance(sidecar, dict):
            continue
        extra = sidecar.get("extra")
        if isinstance(extra, dict):
            value = extra.get("submission")
            if isinstance(value, str) and value:
                return value
    return ""


def simple_session_summary(
    messages: Iterable[Message], *, max_chars: int = 2200
) -> str:
    """Build a small generic summary without calling a model."""

    messages_list = list(messages)
    sections: list[str] = []
    task = first_user_text(messages_list)
    if task:
        sections.append("Task: " + _compact(task, 900))

    tools: list[str] = []
    important: list[str] = []
    for message in messages_list:
        if isinstance(message, AssistantMessage):
            for call in message.tool_calls:
                tools.append(
                    _compact(
                        f"{call.name} {json.dumps(dict(call.arguments), sort_keys=True)}",
                        180,
                    )
                )
        text = extract_memory_text(message)
        for line in text.splitlines():
            lowered = line.lower()
            if any(
                marker in lowered
                for marker in (
                    "error",
                    "failed",
                    "failure",
                    "exception",
                    "traceback",
                    "assert",
                )
            ):
                important.append(_compact(line, 220))

    if tools:
        sections.append("Tools: " + "; ".join(tools[:18]))
    if important:
        sections.append("Important output: " + "; ".join(important[:10]))

    final = next(
        (
            extract_memory_text(message)
            for message in reversed(messages_list)
            if message.role == "assistant" and message.kind == "final"
        ),
        "",
    )
    if final:
        sections.append("Final: " + _compact(final, 500))

    return _compact("\n".join(sections), max_chars)


def _visible_blocks_text(blocks: Iterable[TextBlock | ImageBlock]) -> str:
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, TextBlock) and block.text:
            parts.append(block.text)
        elif isinstance(block, ImageBlock):
            parts.append(f"[image: {block.mime_type or 'image'}]")
    return "\n".join(parts).strip()


def _compact(text: str, limit: int) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(0, limit - 14)] + "...[truncated]"
