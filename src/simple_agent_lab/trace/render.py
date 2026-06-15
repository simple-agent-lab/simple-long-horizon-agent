"""Console renderer for the Layer-1 event log.

Not a trace layer itself: `print_trace` is a human-readable *view* over
`state.events`, useful in demos and while debugging a run interactively.
The structured derivations live in their layer modules (`spans`,
`training`, `run_trace`).
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from ..messages import AssistantMessage, message_text
from ..protocols import (
    ContextCompressionEvent,
    MessageEvent,
    ModelRequestEvent,
    ModelResponseEvent,
)
from ..state import State


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
            extra = (message.sidecar or {}).get("extra")
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
                    raw_payload = (message.sidecar or {}).get("raw")
                    if raw_payload:
                        _print_raw(raw_payload)
        elif isinstance(event, ModelRequestEvent):
            print(
                f"{event.index:02d} {t:<10} {kind:<21} "
                f"agent={event.agent} "
                f"visible={event.visible_count} "
                f"llm_messages={event.llm_message_count}"
            )
        elif isinstance(event, ModelResponseEvent):
            print(
                f"{event.index:02d} {t:<10} {kind:<21} "
                f"agent={event.agent} "
                f"kind={event.output_kind} "
                f"target={event.target} "
                f"tool_calls={event.tool_call_count}"
            )
        elif isinstance(event, ContextCompressionEvent):
            strategy = f"strategy={event.strategy} " if event.strategy else ""
            print(
                f"{event.index:02d} {t:<10} {kind:<21} "
                f"agent={event.agent} "
                f"{strategy}"
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
