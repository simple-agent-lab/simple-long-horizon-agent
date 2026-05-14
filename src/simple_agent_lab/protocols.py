"""Typed runtime events shared across the agent loop and tool dispatch."""

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
    index: int

    @property
    def kind(self) -> EventKind:
        """Stable event discriminator for trace filtering."""
        ...

    @property
    def message(self) -> Message | None:
        """Message carried by message events, otherwise None."""
        ...

    @property
    def data(self) -> dict[str, Any]:
        """Dict view kept for trace formatting and migration compatibility."""
        ...


@dataclass(frozen=True)
class _BaseEvent:
    index: int

    @property
    def message(self) -> Message | None:
        return None


@dataclass(frozen=True)
class MessageEvent:
    index: int
    message: Message

    @property
    def kind(self) -> EventKind:
        return EventKind.MESSAGE

    @property
    def data(self) -> dict[str, Any]:
        return {"message": self.message}


@dataclass(frozen=True)
class AgentStartEvent(_BaseEvent):
    @property
    def kind(self) -> EventKind:
        return EventKind.AGENT_START

    @property
    def data(self) -> dict[str, Any]:
        return {}


@dataclass(frozen=True)
class AgentEndEvent(_BaseEvent):
    reason: str

    @property
    def kind(self) -> EventKind:
        return EventKind.AGENT_END

    @property
    def data(self) -> dict[str, Any]:
        return {"reason": self.reason}


@dataclass(frozen=True)
class TurnStartEvent(_BaseEvent):
    agent: AgentName

    @property
    def kind(self) -> EventKind:
        return EventKind.TURN_START

    @property
    def data(self) -> dict[str, Any]:
        return {"agent": self.agent}


@dataclass(frozen=True)
class TurnEndEvent(_BaseEvent):
    agent: AgentName
    terminated: bool = False

    @property
    def kind(self) -> EventKind:
        return EventKind.TURN_END

    @property
    def data(self) -> dict[str, Any]:
        data: dict[str, Any] = {"agent": self.agent}
        if self.terminated:
            data["terminated"] = True
        return data


@dataclass(frozen=True)
class ModelRequestEvent(_BaseEvent):
    agent: AgentName
    visible_count: int
    llm_message_count: int
    visible: list[dict[str, Any]]
    context_view: dict[str, Any]
    tools: list[dict[str, Any]]
    llm_payload: list[Any]
    candidate_id: Any = None

    @property
    def kind(self) -> EventKind:
        return EventKind.MODEL_REQUEST

    @property
    def data(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "agent": self.agent,
            "visible_count": self.visible_count,
            "llm_message_count": self.llm_message_count,
            "visible": self.visible,
            "context_view": self.context_view,
            "tools": self.tools,
            "llm_payload": self.llm_payload,
        }
        if self.candidate_id is not None:
            data["candidate_id"] = self.candidate_id
        return data


@dataclass(frozen=True)
class ModelResponseEvent(_BaseEvent):
    agent: AgentName
    output_kind: MessageKind
    target: AgentName
    tool_call_count: int
    candidate_id: Any = None

    @property
    def kind(self) -> EventKind:
        return EventKind.MODEL_RESPONSE

    @property
    def data(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "agent": self.agent,
            "output_kind": self.output_kind,
            "target": self.target,
            "tool_call_count": self.tool_call_count,
        }
        if self.candidate_id is not None:
            data["candidate_id"] = self.candidate_id
        return data


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

    @property
    def data(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "summary_message_index": self.summary_message_index,
            "compressed_message_indices": self.compressed_message_indices,
            "preserved_message_indices": self.preserved_message_indices,
            "recent_message_indices": self.recent_message_indices,
            "active_message_indices": self.active_message_indices,
            "before_tokens": self.before_tokens,
            "after_tokens": self.after_tokens,
        }


@dataclass(frozen=True)
class ToolExecutionStartEvent(_BaseEvent):
    tool_call_id: str
    tool_name: str

    @property
    def kind(self) -> EventKind:
        return EventKind.TOOL_EXECUTION_START

    @property
    def data(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
        }


@dataclass(frozen=True)
class ToolExecutionUpdateEvent(_BaseEvent):
    tool_call_id: str
    tool_name: str
    partial: ToolResult

    @property
    def kind(self) -> EventKind:
        return EventKind.TOOL_EXECUTION_UPDATE

    @property
    def data(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "partial": self.partial,
        }


@dataclass(frozen=True)
class ToolExecutionEndEvent(_BaseEvent):
    tool_call_id: str
    tool_name: str
    is_error: bool
    terminate: bool

    @property
    def kind(self) -> EventKind:
        return EventKind.TOOL_EXECUTION_END

    @property
    def data(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "is_error": self.is_error,
            "terminate": self.terminate,
        }


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
        partial: ToolResult,
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
