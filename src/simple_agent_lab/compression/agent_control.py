"""Agent-controlled compaction: a `compact` tool paired with the strategy
that applies it.

The shipped threshold strategies are system-controlled — the runtime watches
token counts and decides for the agent. This module is the other end of the
control spectrum: the model itself decides when its context should shrink and
writes the replacement summary, by calling a `compact` tool.

The decision and its application are deliberately split:

- the **tool** runs inside the tool-dispatch worker pool, where mutating
  `State` is out of bounds; it only records the request on a holder shared
  with the strategy and confirms back to the model;
- the paired **strategy** applies the request at the next turn start — the
  loop's one safe compression point, after the pending tool_result bundle has
  been recorded, so a fold can never split an in-flight tool_call from its
  result.

Build both halves together and wire them to the same agent:

    control = make_compact_control(keep_recent=2)
    agent = Agent(
        "worker",
        step,
        tools=(control.tool, ...),
        context_policy=ContextPolicy(strategy=control.strategy),
    )

To combine with a threshold fallback, put the agent-controlled stage first in
a `TieredStrategy((control.strategy, ToolCompactStrategy(...)))`.

One control serves one running agent: the request holder is shared mutable
state between its tool and its strategy, so concurrent runs each need their
own `make_compact_control()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..context_view import CompressionDecision
from ..messages import Message, MessageKind, make_message
from ..tools import AbortFlag, AgentTool, ToolResult, ToolUpdateFn, text_result
from .strategies import DEFAULT_PRESERVE_KINDS, source_note

COMPACT_TOOL_NAME = "compact"


@dataclass(frozen=True)
class CompactRequest:
    """One pending request: the agent's summary plus an optional keep override."""

    summary: str
    keep_recent: int | None = None  # None -> use the strategy default


@dataclass
class _RequestHolder:
    """Single-slot mailbox between the tool (writer) and the strategy (reader)."""

    request: CompactRequest | None = None

    def put(self, request: CompactRequest) -> None:
        self.request = request

    def take(self) -> CompactRequest | None:
        request, self.request = self.request, None
        return request


@dataclass(frozen=True)
class AgentCompactStrategy:
    """Apply a pending `compact` request; stay idle otherwise.

    A request is consumed exactly once — even when nothing is old enough to
    fold, the pass is a silent no-op rather than a deferred compaction that
    would fire at a surprising later turn. The replacement is the agent's own
    summary text plus a `source_note` citing the folded transcript indices,
    so a `recall` tool can fetch the originals back.
    """

    holder: _RequestHolder
    keep_recent: int = 2
    preserve_kinds: tuple[MessageKind, ...] = DEFAULT_PRESERVE_KINDS

    def __call__(
        self,
        active: list[tuple[int, Message]],
        agent_name: str,
    ) -> CompressionDecision | None:
        request = self.holder.take()
        if request is None or not active:
            return None
        keep = self.keep_recent if request.keep_recent is None else request.keep_recent
        droppable = [item for item in active if item[1].kind not in self.preserve_kinds]
        to_compress = droppable[:-keep] if keep else droppable
        if not to_compress:
            return None
        compress_indices = tuple(index for index, _ in to_compress)
        return CompressionDecision(
            compress_indices=compress_indices,
            replacement=make_message(
                "system",
                request.summary + "\n\n" + source_note(compress_indices),
                sender="runtime",
                target=agent_name,
                kind="summary",
            ),
        )


@dataclass(frozen=True)
class CompactControl:
    """The paired halves of agent-controlled compaction.

    Give `tool` to the agent's `tools` and `strategy` to its
    `ContextPolicy.strategy` (or a `TieredStrategy` stage).
    """

    tool: AgentTool
    strategy: AgentCompactStrategy


def make_compact_control(
    *,
    keep_recent: int = 2,
    preserve_kinds: tuple[MessageKind, ...] = DEFAULT_PRESERVE_KINDS,
    tool_name: str = COMPACT_TOOL_NAME,
) -> CompactControl:
    """Build a `compact` tool and its applying strategy around one holder."""
    if keep_recent < 0:
        raise ValueError("keep_recent must be >= 0")

    holder = _RequestHolder()
    strategy = AgentCompactStrategy(
        holder=holder,
        keep_recent=keep_recent,
        preserve_kinds=preserve_kinds,
    )

    def execute(
        call_id: str,
        args: dict[str, Any],
        abort: AbortFlag,
        on_update: ToolUpdateFn | None,
    ) -> ToolResult:
        del call_id, abort, on_update
        summary = str(args.get("summary", "")).strip()
        if not summary:
            return text_result(
                "`summary` is required and must be non-empty — it replaces the "
                "older messages, so write down everything you still need.",
                is_error=True,
            )
        try:
            keep_override = _coerce_keep_recent(args.get("keep_recent"))
        except ValueError as exc:
            return text_result(f"Invalid compact argument: {exc}", is_error=True)
        holder.put(CompactRequest(summary=summary, keep_recent=keep_override))
        keep = keep_recent if keep_override is None else keep_override
        return text_result(
            "Compaction scheduled: before your next turn, older messages "
            f"(keeping protected kinds and the {keep} most recent) will be "
            "replaced by your summary. The replacement cites the folded "
            "transcript indices; if nothing is old enough to fold, the "
            "request is dropped."
        )

    tool = AgentTool(
        name=tool_name,
        description=(
            "Compress your own older context. Call this when the conversation "
            "has grown long and earlier messages are no longer needed verbatim "
            "— e.g. after finishing a sub-task. Write `summary` as the "
            "replacement for those older messages: keep every fact, decision, "
            "constraint, and open question you still need, because everything "
            "not in the summary leaves your view. The replacement cites the "
            "transcript indices it folded, so a `recall` tool (when available) "
            "can retrieve the originals."
        ),
        parameters={
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": (
                        "Replacement text for the older messages — everything "
                        "you still need from them."
                    ),
                },
                "keep_recent": {
                    "type": "integer",
                    "minimum": 0,
                    "description": (
                        "How many of the most recent non-protected messages to "
                        f"keep verbatim (default {keep_recent})."
                    ),
                },
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
        execute=execute,
    )
    return CompactControl(tool=tool, strategy=strategy)


def _coerce_keep_recent(value: Any) -> int | None:
    """Coerce the optional `keep_recent` argument to a non-negative int."""
    if value is None or value == "":
        return None
    # bool is an int subclass; reject it so `True` isn't read as keep 1.
    if isinstance(value, bool):
        raise ValueError(f"keep_recent must be an integer, got {value!r}")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"keep_recent must be an integer, got {value!r}")
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"keep_recent must be an integer, got {value!r}") from None
    if number < 0:
        raise ValueError(f"keep_recent must be >= 0, got {number}")
    return number
