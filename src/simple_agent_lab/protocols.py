"""Typed runtime events shared across the agent loop and tool dispatch.

Every event is a frozen dataclass with explicitly named fields. Each one
carries a stable `kind: EventKind` discriminator so callers can pattern-match
with `isinstance` (preferred) or filter on `event.kind`. Serializers should
go through `trajectory.event_record` (or `dataclasses.asdict`) rather than
poking individual fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

from .messages import AgentName, Message, MessageKind

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


class RuntimeEvent(Protocol):
    """Structural shape of every runtime event.

    Every concrete event has an integer `index` and a `kind` discriminator.
    Callers that need a specific event's fields should narrow with
    `isinstance(event, ConcreteEvent)`; the typed fields are the source
    of truth.
    """

    index: int

    @property
    def kind(self) -> EventKind: ...


@dataclass(frozen=True)
class _BaseEvent:
    index: int


@dataclass(frozen=True)
class MessageEvent:
    index: int
    message: Message

    @property
    def kind(self) -> EventKind:
        return EventKind.MESSAGE


@dataclass(frozen=True)
class AgentStartEvent(_BaseEvent):
    @property
    def kind(self) -> EventKind:
        return EventKind.AGENT_START


@dataclass(frozen=True)
class AgentEndEvent(_BaseEvent):
    reason: str

    @property
    def kind(self) -> EventKind:
        return EventKind.AGENT_END


@dataclass(frozen=True)
class TurnStartEvent(_BaseEvent):
    agent: AgentName

    @property
    def kind(self) -> EventKind:
        return EventKind.TURN_START


@dataclass(frozen=True)
class TurnEndEvent(_BaseEvent):
    agent: AgentName
    terminated: bool = False

    @property
    def kind(self) -> EventKind:
        return EventKind.TURN_END


@dataclass(frozen=True)
class ModelRequestEvent(_BaseEvent):
    agent: AgentName
    visible_count: int
    llm_message_count: int
    # JSON-shaped trace records. Callers read by known keys; `Any` is
    # deliberate — these dicts are heterogeneous trace data, not typed
    # records, and static narrowing on every read would just be noise.
    visible: list[dict[str, Any]]
    context_view: dict[str, Any]
    tools: list[dict[str, Any]]
    llm_payload: list[Any]
    candidate_id: str | None = None

    @property
    def kind(self) -> EventKind:
        return EventKind.MODEL_REQUEST


@dataclass(frozen=True)
class ModelResponseEvent(_BaseEvent):
    agent: AgentName
    output_kind: MessageKind
    target: AgentName
    tool_call_count: int
    candidate_id: str | None = None

    @property
    def kind(self) -> EventKind:
        return EventKind.MODEL_RESPONSE


@dataclass(frozen=True)
class ContextCompressionEvent(_BaseEvent):
    agent: AgentName
    summary_message_index: int
    compressed_message_indices: list[int]
    preserved_message_indices: list[int]
    recent_message_indices: list[int]
    before_tokens: int
    after_tokens: int

    @property
    def kind(self) -> EventKind:
        return EventKind.CONTEXT_COMPRESSION

    @property
    def active_message_indices(self) -> list[int]:
        return [
            *self.preserved_message_indices,
            self.summary_message_index,
            *self.recent_message_indices,
        ]


@dataclass(frozen=True)
class ToolExecutionStartEvent(_BaseEvent):
    tool_call_id: str
    tool_name: str

    @property
    def kind(self) -> EventKind:
        return EventKind.TOOL_EXECUTION_START


@dataclass(frozen=True)
class ToolExecutionUpdateEvent(_BaseEvent):
    tool_call_id: str
    tool_name: str
    partial: ToolResult

    @property
    def kind(self) -> EventKind:
        return EventKind.TOOL_EXECUTION_UPDATE


@dataclass(frozen=True)
class ToolExecutionEndEvent(_BaseEvent):
    tool_call_id: str
    tool_name: str
    is_error: bool
    terminate: bool

    @property
    def kind(self) -> EventKind:
        return EventKind.TOOL_EXECUTION_END


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


class EventRecorder(Protocol):
    """Minimal event sink needed by tool dispatch."""

    def record(self, message: Message) -> MessageEvent:
        """Record and return a message event."""
        ...

    def tool_execution_start(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
    ) -> ToolExecutionStartEvent:
        """Record that a tool call is starting."""
        ...

    def tool_execution_update(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        partial: "ToolResult",
    ) -> ToolExecutionUpdateEvent:
        """Record a partial tool update."""
        ...

    def tool_execution_end(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        is_error: bool,
        terminate: bool,
    ) -> ToolExecutionEndEvent:
        """Record that a tool call finished."""
        ...
