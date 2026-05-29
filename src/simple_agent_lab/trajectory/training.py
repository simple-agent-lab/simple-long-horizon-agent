"""Layer 3 — ModelTurn: model-visible input/output pairs for training.

Each ``ModelRequestEvent`` paired with the next assistant ``MessageEvent``
becomes one ``ModelTurn`` suitable for supervised fine-tuning. Like the
span layer this module is a pure transform over the event log.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..protocols import Event, MessageEvent, ModelRequestEvent
from .jsonl import json_safe


@dataclass(frozen=True)
class ModelTurn:
    """One model-visible input/output pair captured from a run."""

    step_id: str
    agent: str
    input_messages: list[Any]
    output_message: Any
    tools: list[Any]
    meta: dict[str, Any] | None = None


def model_turns_from_events(
    trace_id: str,
    events: Iterable[Event],
) -> list[ModelTurn]:
    """Extract model-visible input/output pairs from runtime events.

    Each ``ModelRequestEvent`` → next assistant ``MessageEvent`` pair
    becomes one ``ModelTurn`` suitable for supervised fine-tuning.
    """

    turns: list[ModelTurn] = []
    pending: dict[str, Any] | None = None
    model_call_index = 0

    for event in events:
        if isinstance(event, ModelRequestEvent):
            model_call_index += 1
            pending = {
                "agent": str(event.agent or ""),
                "input_messages": event.llm_payload,
                "tools": event.tools,
                "request_event_index": event.index,
                "meta": {
                    "visible_count": event.visible_count,
                    "model_message_count": event.llm_message_count,
                },
            }
            continue

        if not isinstance(event, MessageEvent) or pending is None:
            continue
        message = event.message
        if message.role != "assistant":
            continue
        agent = pending["agent"] or message.sender
        if message.sender != agent:
            continue
        turns.append(
            ModelTurn(
                step_id=f"{trace_id}.model{model_call_index}",
                agent=agent,
                input_messages=json_safe(pending["input_messages"]),
                output_message=json_safe(message),
                tools=json_safe(pending["tools"]),
                meta={
                    **pending["meta"],
                    "request_event_index": pending["request_event_index"],
                    "message_event_index": event.index,
                },
            )
        )
        pending = None

    return turns
