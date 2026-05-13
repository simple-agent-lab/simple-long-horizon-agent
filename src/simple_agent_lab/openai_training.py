"""Export agent transcripts as OpenAI Chat fine-tuning JSONL.

Each `State` becomes one JSONL line of the form::

    {"messages": [...], "tools": [...]}

matching OpenAI's supervised fine-tuning data format for chat models
(including tool-use turns). The message list reuses the same
`to_openai_chat_messages` shape the openai-chat adapter sends on the
wire, so what trains is what runs.

This export is the runtime's training-data hand-off; the lab's
internal trajectory / training-data records live in `trajectory.py`
and `training_data.py` and serve a different purpose
(experiment-tracking and a project-specific schema).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .core import State
from .llm.adapters.openai_chat import to_openai_chat_messages, to_openai_chat_tools
from .llm.bridge import messages_to_llm_messages, tool_to_llm_tool
from .tools import AgentTool, Tool


def openai_training_record(
    state: State,
    *,
    tools: Sequence[Tool | AgentTool] = (),
    system_prompt: str | None = None,
    include_reasoning_content: bool = True,
    skip_kinds: set[str] | None = None,
) -> dict[str, Any]:
    """Build an OpenAI Chat fine-tuning record from a runtime `State`.

    `tools` is the set of `AgentTool`s the run had available — it gets
    serialized into the record's `tools` field so the trained model
    can be conditioned on the same tool surface.

    `system_prompt` is the agent-level system prompt (the runtime
    doesn't carry it on `state.messages` because it's a per-agent
    config, not a transcript entry).

    `include_reasoning_content` controls whether prior assistant
    `thinking` blocks are serialized as a `reasoning_content` sibling
    field on each assistant turn (DeepSeek / mimo style).
    """
    llm_messages = messages_to_llm_messages(state.messages, skip_kinds=skip_kinds)
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
    skip_kinds: set[str] | None = None,
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
        skip_kinds=skip_kinds,
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")
    return out
