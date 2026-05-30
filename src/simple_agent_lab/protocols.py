"""Typed runtime events shared across the agent loop and tool dispatch.

Every event is a frozen, keyword-only dataclass with explicitly named
fields. Each one carries a stable `kind: EventKind` discriminator stored
as a real `Literal[...]` field with `init=False`, so callers can pattern
match with `isinstance` (preferred) or filter on `event.kind`, and so
`dataclasses.asdict` (and any JSON path that goes through it) keeps the
discriminator intact. Construction omits `index` and `elapsed` (they
default to placeholder values); `State.record_event` stamps the real
values when the event is appended.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from .messages import AgentName, Message, MessageKind, TokenUsage

if TYPE_CHECKING:
    from .tools import ToolResult


class EventKind(str, Enum):
    MESSAGE = "message"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    CONTEXT_COMPRESSION = "context_compression"
    TOOL_EXECUTION_START = "tool_execution_start"
    TOOL_EXECUTION_UPDATE = "tool_execution_update"
    TOOL_EXECUTION_END = "tool_execution_end"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, kw_only=True)
class _BaseEvent:
    # Placeholders; `State.record_event` stamps the real values on append.
    index: int = -1
    elapsed: float = 0.0


@dataclass(frozen=True, kw_only=True)
class MessageEvent(_BaseEvent):
    kind: Literal[EventKind.MESSAGE] = field(default=EventKind.MESSAGE, init=False)
    message: Message


@dataclass(frozen=True, kw_only=True)
class AgentStartEvent(_BaseEvent):
    kind: Literal[EventKind.AGENT_START] = field(
        default=EventKind.AGENT_START, init=False
    )


# Why an agent's run loop stopped. Closed set so static checkers catch
# typos and consumers can match exhaustively (same rationale as
# `MessageKind` / `StopReason`); the loop in `core.run_agent` is the sole
# producer.
#   "done"           — the agent emitted a `kind="final"` message
#   "max_turns"      — the turn budget ran out before a final message
#   "tool_terminate" — a tool returned `ToolResult(terminate=True)`
AgentEndReason: TypeAlias = Literal["done", "max_turns", "tool_terminate"]


@dataclass(frozen=True, kw_only=True)
class AgentEndEvent(_BaseEvent):
    kind: Literal[EventKind.AGENT_END] = field(default=EventKind.AGENT_END, init=False)
    reason: AgentEndReason


@dataclass(frozen=True, kw_only=True)
class TurnStartEvent(_BaseEvent):
    kind: Literal[EventKind.TURN_START] = field(
        default=EventKind.TURN_START, init=False
    )
    agent: AgentName


@dataclass(frozen=True, kw_only=True)
class TurnEndEvent(_BaseEvent):
    kind: Literal[EventKind.TURN_END] = field(default=EventKind.TURN_END, init=False)
    agent: AgentName
    terminated: bool = False


@dataclass(frozen=True, kw_only=True)
class ModelRequestEvent(_BaseEvent):
    kind: Literal[EventKind.MODEL_REQUEST] = field(
        default=EventKind.MODEL_REQUEST, init=False
    )
    agent: AgentName
    visible_count: int
    llm_message_count: int
    # JSON-shaped trace records. Callers read by known keys; `Any` is
    # deliberate — these dicts are heterogeneous trace data, not typed
    # records, and static narrowing on every read would just be noise.
    context_view: dict[str, Any]
    tools: list[dict[str, Any]]
    llm_payload: list[Any]


@dataclass(frozen=True, kw_only=True)
class ModelResponseEvent(_BaseEvent):
    kind: Literal[EventKind.MODEL_RESPONSE] = field(
        default=EventKind.MODEL_RESPONSE, init=False
    )
    agent: AgentName
    output_kind: MessageKind
    target: AgentName
    tool_call_count: int
    # Per-call cost primitives, snapshotted from the response message
    # alongside the facts above. `usage` is None when the provider didn't
    # report counts; `model` is "" when unknown. Carrying them here lets the
    # span layer fold cost without walking messages or the raw blob.
    usage: TokenUsage | None = None
    model: str = ""


@dataclass(frozen=True, kw_only=True)
class ContextCompressionEvent(_BaseEvent):
    """One compression pass.

    `active_context_indices` is the new active context (preserved messages +
    summary + recent), in chronological order. `compressed_message_indices`
    lists the messages that were folded into `summary_message_index`. The
    trace viewer can derive "preserved" / "recent" from the order of
    `active_context_indices` relative to `summary_message_index`.
    """

    kind: Literal[EventKind.CONTEXT_COMPRESSION] = field(
        default=EventKind.CONTEXT_COMPRESSION, init=False
    )
    agent: AgentName
    summary_message_index: int
    compressed_message_indices: list[int]
    active_context_indices: list[int]
    before_tokens: int
    after_tokens: int


@dataclass(frozen=True, kw_only=True)
class ToolExecutionStartEvent(_BaseEvent):
    kind: Literal[EventKind.TOOL_EXECUTION_START] = field(
        default=EventKind.TOOL_EXECUTION_START, init=False
    )
    tool_call_id: str
    tool_name: str


@dataclass(frozen=True, kw_only=True)
class ToolExecutionUpdateEvent(_BaseEvent):
    kind: Literal[EventKind.TOOL_EXECUTION_UPDATE] = field(
        default=EventKind.TOOL_EXECUTION_UPDATE, init=False
    )
    tool_call_id: str
    tool_name: str
    partial: ToolResult


@dataclass(frozen=True, kw_only=True)
class ToolExecutionEndEvent(_BaseEvent):
    kind: Literal[EventKind.TOOL_EXECUTION_END] = field(
        default=EventKind.TOOL_EXECUTION_END, init=False
    )
    tool_call_id: str
    tool_name: str
    is_error: bool
    terminate: bool


Event: TypeAlias = (
    MessageEvent
    | AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | ModelRequestEvent
    | ModelResponseEvent
    | ContextCompressionEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
)
