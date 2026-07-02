"""Transcript text helpers for memory backends.

These helpers extract only project-owned, model-visible message content. They
never read provider raw payloads or debug-only sidecars.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from simple_agent_lab.messages import (
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
        # Skip memory's own injected context (the policy/summary block recalled at
        # session start): it is framework scaffolding, not run evidence, and
        # feeding it back into the distiller would echo the instructions the
        # distiller is told to ignore. ``index`` still advances so section ids stay
        # stable and unique as anchors.
        if message.sender == "memory":
            continue
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


def _visible_blocks_text(blocks: Iterable[TextBlock | ImageBlock]) -> str:
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, TextBlock) and block.text:
            parts.append(block.text)
        elif isinstance(block, ImageBlock):
            parts.append(f"[image: {block.mime_type or 'image'}]")
    return "\n".join(parts).strip()
