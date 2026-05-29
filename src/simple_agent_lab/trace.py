"""Trace utilities for runtime transcripts.

`core.py` owns the inspectable runtime: `Agent + Message + State +
context_view() + run()`. This module owns downstream trace consumers:
human-readable trace printing and OpenAI Chat fine-tuning JSONL export.

Each OpenAI training export line has the form::

    {"messages": [...], "tools": [...]}

matching OpenAI's supervised fine-tuning data format for chat models
(including tool-use turns). The message list reuses the same
`to_openai_chat_messages` shape the openai-chat adapter sends on the
wire, so what trains is what runs.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .llm.adapters.openai_chat import to_openai_chat_messages, to_openai_chat_tools
from .llm.bridge import messages_to_llm_messages, tool_to_llm_tool
from .messages import AssistantMessage, message_text
from .protocols import (
    ContextCompressionEvent,
    MessageEvent,
    ModelRequestEvent,
    ModelResponseEvent,
)
from .state import State
from .tools import AgentTool, Tool


def print_trace(state: State, *, raw: bool = False) -> None:
    """Print the standardized trace.

    `raw=True` also dumps each model call's `raw` payload -- the provider
    request snapshot (with messages history pruned) and the SDK response
    dump -- so the trace doubles as an HTTP-level diff tool.
    """
    print("\ntrace")
    print("-----")
    for event in state.events:
        kind = event.kind.value
        t = f"@{event.elapsed:.3f}s"
        if isinstance(event, MessageEvent):
            message = event.message
            route = f"{message.sender} -> {message.target}"
            print(
                f"{event.index:02d} {t:<10} {kind:<21} {message.kind:<10} "
                f"{route:<24} {message_text(message)}"
            )
            extra = (message.data or {}).get("extra")
            if extra:
                preview = ", ".join(f"{k}={v!r}" for k, v in extra.items())
                print(f"   {'':10} {'extra':<21} {preview[:200]}")
            if isinstance(message, AssistantMessage):
                for thinking_block in message.thinking:
                    preview = thinking_block.text.replace("\n", " ")
                    if len(preview) > 200:
                        preview = preview[:200] + "..."
                    tag = "redacted_thinking" if thinking_block.redacted else "thinking"
                    print(f"   {'':10} {tag:<21} {preview}")
                if raw:
                    raw_payload = (message.data or {}).get("raw")
                    if raw_payload:
                        _print_raw(raw_payload)
        elif isinstance(event, ModelRequestEvent):
            candidate = event.candidate_id
            suffix = f" candidate={candidate}" if candidate is not None else ""
            print(
                f"{event.index:02d} {t:<10} {kind:<21} "
                f"agent={event.agent} "
                f"visible={event.visible_count} "
                f"llm_messages={event.llm_message_count}{suffix}"
            )
        elif isinstance(event, ModelResponseEvent):
            candidate = event.candidate_id
            suffix = f" candidate={candidate}" if candidate is not None else ""
            print(
                f"{event.index:02d} {t:<10} {kind:<21} "
                f"agent={event.agent} "
                f"kind={event.output_kind} "
                f"target={event.target} "
                f"tool_calls={event.tool_call_count}{suffix}"
            )
        elif isinstance(event, ContextCompressionEvent):
            print(
                f"{event.index:02d} {t:<10} {kind:<21} "
                f"agent={event.agent} "
                f"compressed={event.compressed_message_indices} "
                f"summary_idx={event.summary_message_index} "
                f"tokens={event.before_tokens}->{event.after_tokens}"
            )
        else:
            extras = " ".join(
                f"{field.name}={getattr(event, field.name)}"
                for field in dataclasses.fields(event)
                if field.name not in ("index", "elapsed")
            )
            print(f"{event.index:02d} {t:<10} {kind:<21} {extras}")


def openai_training_record(
    state: State,
    *,
    tools: Sequence[Tool | AgentTool] = (),
    system_prompt: str | None = None,
    include_reasoning_content: bool = True,
    model_invisible_kinds: set[str] | None = None,
) -> dict[str, Any]:
    """Build an OpenAI Chat fine-tuning record from a runtime `State`.

    `tools` is the set of `AgentTool`s the run had available -- it gets
    serialized into the record's `tools` field so the trained model
    can be conditioned on the same tool surface.

    `system_prompt` is the agent-level system prompt (the runtime
    doesn't carry it on `state.messages` because it's a per-agent
    config, not a transcript entry).

    `include_reasoning_content` controls whether prior assistant
    `thinking` blocks are serialized as a `reasoning_content` sibling
    field on each assistant turn (DeepSeek / mimo style).
    """
    llm_messages = messages_to_llm_messages(
        state.messages, model_invisible_kinds=model_invisible_kinds
    )
    record: dict[str, Any] = {
        "messages": to_openai_chat_messages(
            llm_messages,
            system_prompt=system_prompt,
            include_reasoning_content=include_reasoning_content,
        ),
    }
    if tools:
        record["tools"] = to_openai_chat_tools(
            [tool_to_llm_tool(tool) for tool in tools]
        )
    return record


def append_openai_training_record(
    state: State,
    path: str | Path,
    *,
    tools: Sequence[Tool | AgentTool] = (),
    system_prompt: str | None = None,
    include_reasoning_content: bool = True,
    model_invisible_kinds: set[str] | None = None,
) -> Path:
    """Build a record and append it to `path` as a JSONL line.

    Returns the resolved path. The file is created if missing; if it
    already exists, the new line is appended (so successive runs of the
    same agent accumulate in one file).
    """
    record = openai_training_record(
        state,
        tools=tools,
        system_prompt=system_prompt,
        include_reasoning_content=include_reasoning_content,
        model_invisible_kinds=model_invisible_kinds,
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")
    return out


def _print_raw(raw: Any) -> None:
    for label in ("request", "response"):
        body = raw.get(label) if isinstance(raw, dict) else None
        if body is None:
            continue
        print(f"   raw.{label}:")
        try:
            rendered = json.dumps(body, indent=2, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            rendered = repr(body)
        for line in rendered.splitlines():
            print(f"     {line}")
