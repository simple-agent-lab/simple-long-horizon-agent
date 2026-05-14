"""Append-only runtime state and its derived snapshot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .messages import (
    AgentName,
    ContentInput,
    Message,
    MessageChannel,
    MessageKind,
    Role,
    make_message,
)
from .protocols import (
    AgentEndEvent,
    AgentStartEvent,
    ContextCompressionEvent,
    Event,
    MessageEvent,
    ModelRequestEvent,
    ModelResponseEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)


@dataclass
class StateSnapshot:
    """Derived cache for fast access to the current event projection."""

    messages: list[Message] = field(default_factory=list)
    active_context_message_indices: list[int] | None = None
    latest_context_compression_event_index: int | None = None

    def apply(self, event: Event) -> None:
        if isinstance(event, MessageEvent):
            message_index = len(self.messages)
            self.messages.append(event.message)
            if self.active_context_message_indices is not None:
                self.active_context_message_indices.append(message_index)
            return
        if isinstance(event, ContextCompressionEvent):
            self.active_context_message_indices = event.active_message_indices
            self.latest_context_compression_event_index = event.index

    def active_items(self) -> list[tuple[int, Message]]:
        indices = self.active_context_message_indices
        if indices is None:
            return list(enumerate(self.messages))

        items: list[tuple[int, Message]] = []
        used: set[int] = set()
        for index in indices:
            if index in used or not 0 <= index < len(self.messages):
                continue
            used.add(index)
            items.append((index, self.messages[index]))
        return items

    def active_messages(self) -> list[Message]:
        return [message for _, message in self.active_items()]


@dataclass
class State:
    task: str
    events: list[Event] = field(default_factory=list)
    snapshot: StateSnapshot = field(default_factory=StateSnapshot)
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.events and not self.snapshot.messages:
            self.rebuild_snapshot()

    @property
    def messages(self) -> list[Message]:
        return list(self.snapshot.messages)

    def active_context_items(self) -> list[tuple[int, Message]]:
        return self.snapshot.active_items()

    def active_context_messages(self) -> list[Message]:
        return self.snapshot.active_messages()

    def _append(self, event: Event) -> None:
        self.events.append(event)
        self.snapshot.apply(event)

    def rebuild_snapshot(self) -> StateSnapshot:
        snapshot = StateSnapshot()
        for event in self.events:
            snapshot.apply(event)
        self.snapshot = snapshot
        return snapshot

    def record(self, message: Message) -> MessageEvent:
        event = MessageEvent(len(self.events), message)
        self._append(event)
        return event

    def context_compression(
        self,
        *,
        agent: AgentName,
        summary_message_index: int,
        compressed_message_indices: list[int],
        preserved_message_indices: list[int],
        recent_message_indices: list[int],
        before_tokens: int,
        after_tokens: int,
    ) -> ContextCompressionEvent:
        event = ContextCompressionEvent(
            len(self.events),
            agent=agent,
            summary_message_index=summary_message_index,
            compressed_message_indices=compressed_message_indices,
            preserved_message_indices=preserved_message_indices,
            recent_message_indices=recent_message_indices,
            before_tokens=before_tokens,
            after_tokens=after_tokens,
        )
        self._append(event)
        return event

    def agent_start(self) -> AgentStartEvent:
        event = AgentStartEvent(len(self.events))
        self._append(event)
        return event

    def agent_end(self, *, reason: str) -> AgentEndEvent:
        event = AgentEndEvent(len(self.events), reason=reason)
        self._append(event)
        return event

    def turn_start(self, *, agent: AgentName) -> TurnStartEvent:
        event = TurnStartEvent(len(self.events), agent=agent)
        self._append(event)
        return event

    def turn_end(self, *, agent: AgentName, terminated: bool = False) -> TurnEndEvent:
        event = TurnEndEvent(len(self.events), agent=agent, terminated=terminated)
        self._append(event)
        return event

    def model_request(
        self,
        *,
        agent: AgentName,
        visible_count: int,
        llm_message_count: int,
        visible: list[dict[str, Any]],
        context_view: dict[str, Any],
        tools: list[dict[str, Any]],
        llm_payload: list[Any],
        candidate_id: Any = None,
    ) -> ModelRequestEvent:
        event = ModelRequestEvent(
            len(self.events),
            agent=agent,
            visible_count=visible_count,
            llm_message_count=llm_message_count,
            visible=visible,
            context_view=context_view,
            tools=tools,
            llm_payload=llm_payload,
            candidate_id=candidate_id,
        )
        self._append(event)
        return event

    def model_response(
        self,
        *,
        agent: AgentName,
        output_kind: MessageKind,
        target: AgentName,
        tool_call_count: int,
        candidate_id: Any = None,
    ) -> ModelResponseEvent:
        event = ModelResponseEvent(
            len(self.events),
            agent=agent,
            output_kind=output_kind,
            target=target,
            tool_call_count=tool_call_count,
            candidate_id=candidate_id,
        )
        self._append(event)
        return event

    def tool_execution_start(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
    ) -> ToolExecutionStartEvent:
        event = ToolExecutionStartEvent(
            len(self.events),
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )
        self._append(event)
        return event

    def tool_execution_update(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        partial: Any,
    ) -> ToolExecutionUpdateEvent:
        event = ToolExecutionUpdateEvent(
            len(self.events),
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            partial=partial,
        )
        self._append(event)
        return event

    def tool_execution_end(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        is_error: bool,
        terminate: bool,
    ) -> ToolExecutionEndEvent:
        event = ToolExecutionEndEvent(
            len(self.events),
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            is_error=is_error,
            terminate=terminate,
        )
        self._append(event)
        return event

    def send(
        self,
        kind: MessageKind,
        sender: AgentName,
        target: AgentName,
        content: ContentInput = "",
        role: Role | None = None,
        channel: MessageChannel = "main",
        **data: Any,
    ) -> Message:
        resolved_role = role
        if resolved_role is None:
            if sender in {"system", "state", "runtime"}:
                resolved_role = "system"
            elif sender == "user":
                resolved_role = "user"
            else:
                resolved_role = "assistant"
        message = make_message(
            resolved_role,
            content,
            sender=sender,
            target=target,
            kind=kind,
            channel=channel,
            **data,
        )
        self.record(message)
        return message
