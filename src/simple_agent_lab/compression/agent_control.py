"""Agent-controlled compaction: a `compact` tool paired with the strategy
that applies it.

The shipped threshold strategies are system-controlled — the runtime watches
token counts and decides for the agent. This module is the other end of the
control spectrum: the model itself decides when its context should shrink and
writes the replacement summary, by calling a `compact` tool.

The decision and its application are deliberately split, and the request flows
between them through the **append-only transcript** rather than a side channel:

- the **tool** runs inside the tool-dispatch worker pool, where mutating
  `State` is out of bounds; it stays a pure function and returns its request as
  structured `ToolResult.details` (`{"compact_request": ...}`). The loop records
  that with the tool_result bundle — single-threaded, at the one safe point —
  so the request lands in the event log and the trace like any other tool
  output, with no shared mutable state or locking between tool and strategy;
- the paired **strategy** runs at the next turn start — the loop's one safe
  compression point, after the pending tool_result bundle has been recorded, so
  a fold can never split an in-flight tool_call from its result. It reads the
  most recent `compact_request` back off the `active` view's sidecars and
  applies it.

Exactly-once without a destructive read: the strategy applies a request only
while it is the newest message in the active view (the immediate next turn).
Applying splices a `kind="summary"` whose transcript index outranks it, and the
model's next output outranks it too, so the request is never the max again — it
stays in the log (recoverable, auditable) but fires only once, and a request
that finds nothing to fold is dropped rather than deferred to a later turn.

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

The tool and strategy share no mutable state, so one `make_compact_control()`
can serve any number of concurrent runs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..context_view import CompressionDecision
from ..messages import Message, MessageKind, make_message
from ..tools import (
    AbortFlag,
    AgentTool,
    ToolResult,
    ToolUpdateFn,
    coerce_int,
    text_result,
)
from .strategies import DEFAULT_PRESERVE_KINDS, source_note

COMPACT_TOOL_NAME = "compact"

# Key under a tool_result's `details` sidecar that carries an agent's compaction
# request. The tool writes it; the strategy reads it back off the transcript.
COMPACT_REQUEST_KEY = "compact_request"


@dataclass(frozen=True)
class CompactRequest:
    """One request: the agent's summary plus an optional keep override."""

    summary: str
    keep_recent: int | None = None  # None -> use the strategy default


@dataclass(frozen=True)
class AgentCompactStrategy:
    """Apply the agent's most recent `compact` request; stay idle otherwise.

    The request is read from the `active` view's tool_result sidecars (see
    `_latest_compact_request`), not a side channel. It fires only while it is
    the newest message in the view — the immediate next turn start, before the
    model has produced anything after it. That single index test gives every
    property the old destructive mailbox did, from the append-only order alone:

    - **exactly once** — applying splices a `kind="summary"` whose transcript
      index outranks the request, so the request is no longer the max and never
      re-fires;
    - **no deferral** — if nothing is old enough to fold on that turn the pass
      is a no-op, and once the model moves on the request stops being the max,
      so a stale summary can never fold messages it was not written to describe.

    The replacement is the agent's own summary text plus a `source_note` citing
    the folded transcript indices, so a `recall` tool can fetch the originals.
    """

    keep_recent: int = 2
    preserve_kinds: tuple[MessageKind, ...] = DEFAULT_PRESERVE_KINDS

    def __call__(
        self,
        active: list[tuple[int, Message]],
        agent_name: str,
    ) -> CompressionDecision | None:
        found = _latest_compact_request(active)
        if found is None:
            return None
        request_index, request = found
        # Apply only while the request is the freshest message. Once a fold's
        # summary or the model's next output outranks it, the request is spent —
        # exactly once, no deferred (and so no stale) compaction, no marker.
        if request_index != max(index for index, _ in active):
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
                "user",
                request.summary + "\n\n" + source_note(compress_indices),
                sender="runtime",
                target=agent_name,
                kind="summary",
            ),
            label="agent-compact",
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
    """Build a `compact` tool and the strategy that applies its requests."""
    if keep_recent < 0:
        raise ValueError("keep_recent must be >= 0")

    strategy = AgentCompactStrategy(
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
        request: dict[str, Any] = {"summary": summary}
        if keep_override is not None:
            request["keep_recent"] = keep_override
        keep = keep_recent if keep_override is None else keep_override
        return text_result(
            "Compaction recorded: at your next turn start, older messages "
            f"(keeping protected kinds and the {keep} most recent) will be "
            "replaced by your summary. The replacement cites the folded "
            "transcript indices; if nothing is old enough to fold, the "
            "request is dropped.",
            details={COMPACT_REQUEST_KEY: request},
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


def _latest_compact_request(
    active: list[tuple[int, Message]],
) -> tuple[int, CompactRequest] | None:
    """Find the newest `compact_request` carried in the active view's sidecars.

    Scans newest-first and returns the first request found, so the most recent
    `compact` call wins. When one turn carried several `compact` calls they
    share a tool_result bundle; the first call recorded in that bundle is used.
    """
    for index, message in reversed(active):
        details = message.sidecar.get("details")
        if not isinstance(details, Mapping):
            continue
        for entry in details.values():
            request = _read_request(entry)
            if request is not None:
                return index, request
    return None


def _read_request(entry: Any) -> CompactRequest | None:
    """Parse one `details` entry into a `CompactRequest`, or `None` if it is not one."""
    if not isinstance(entry, Mapping):
        return None
    payload = entry.get(COMPACT_REQUEST_KEY)
    if not isinstance(payload, Mapping):
        return None
    summary = str(payload.get("summary", "")).strip()
    if not summary:
        return None
    keep_recent = payload.get("keep_recent")
    valid_keep = isinstance(keep_recent, int) and not isinstance(keep_recent, bool)
    return CompactRequest(
        summary=summary,
        keep_recent=keep_recent if valid_keep else None,
    )


def _coerce_keep_recent(value: Any) -> int | None:
    """Coerce the optional `keep_recent` argument to a non-negative int."""
    if value is None or value == "":
        return None
    return coerce_int("keep_recent", value, minimum=0)
