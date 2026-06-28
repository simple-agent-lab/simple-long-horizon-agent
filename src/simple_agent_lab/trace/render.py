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

from ..messages import AssistantMessage, TokenUsage, message_text
from ..model_metadata import PriceBook, RunCost
from ..protocols import (
    ContextCompressionEvent,
    MessageEvent,
    ModelRequestEvent,
    ModelResponseEvent,
)
from ..state import State


def print_trace(
    state: State,
    *,
    raw: bool = False,
    costs: bool = True,
    price_book: PriceBook | None = None,
) -> None:
    """Print the standardized trace.

    `raw=True` also dumps each model call's `raw` payload -- the provider
    request snapshot (with messages history pruned) and the SDK response
    dump -- so the trace doubles as an HTTP-level diff tool.

    `costs=True` appends a cost summary footer (tokens + dollars, per model)
    derived from the run's `usage`/`model` primitives via `model_metadata.RunCost`.
    `price_book` overrides the built-in rate card (defaults to
    `model_metadata.default_price_book()`, i.e. the table plus any env override).
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

    if costs:
        _print_cost_summary(RunCost.from_run(state.events, state.messages, price_book))


def _format_tokens(tokens: TokenUsage) -> str:
    """The shared per-bucket token line used in both cost rows."""
    return (
        f"in={tokens.input_tokens} out={tokens.output_tokens} "
        f"cache_r={tokens.cache_read_tokens} cache_w={tokens.cache_write_tokens}"
    )


_PRICE_BOOK_HINT = "the SIMPLE_AGENT_LAB_PRICE_BOOK env file or a custom PriceBook"


def _print_cost_summary(run_cost: RunCost) -> None:
    """Render the per-model token + dollar rollup under the event list."""
    if not run_cost.by_model:
        return
    print("\ncost")
    print("----")
    print(
        f"total ${run_cost.total_usd:.4f} "
        f"over {run_cost.calls} call(s) "
        f"({_format_tokens(run_cost.total_tokens)})"
    )
    for entry in run_cost.by_model:
        flag = " (unpriced)" if entry.model in run_cost.unpriced_models else ""
        print(
            f"  {entry.model:<28} ${entry.cost.total_usd:.4f} "
            f"x{entry.calls} {_format_tokens(entry.tokens)}{flag}"
        )
    if run_cost.unpriced_models:
        print(
            "  note: unpriced models contribute $0 — total is a lower bound. "
            f"Add rates via {_PRICE_BOOK_HINT}."
        )


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
