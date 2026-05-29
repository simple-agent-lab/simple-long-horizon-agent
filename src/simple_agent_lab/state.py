"""Append-only runtime state and its derived snapshot."""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

from .messages import (
    AgentName,
    ContentInput,
    Message,
    MessageKind,
    MessageSidecar,
    Role,
    make_message,
)
from .protocols import (
    ContextCompressionEvent,
    Event,
    MessageEvent,
)

EventT = TypeVar("EventT", bound=Event)


@dataclass
class StateSnapshot:
    """Derived cache for fast access to the current event projection."""

    messages: list[Message] = field(default_factory=list)
    active_context_message_indices: list[int] | None = None

    def apply(self, event: Event) -> None:
        if isinstance(event, MessageEvent):
            message_index = len(self.messages)
            self.messages.append(event.message)
            if self.active_context_message_indices is not None:
                self.active_context_message_indices.append(message_index)
            return
        if isinstance(event, ContextCompressionEvent):
            # Copy: the snapshot mutates this list as new messages arrive;
            # the event's field must stay a stable historical record.
            self.active_context_message_indices = list(event.active_message_indices)

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
    # Open scratchpad for harnesses/experiments to hang run-level metadata
    # on a State (e.g. the SWE-bench runner stashes `model_patch`,
    # `workspace`). Intentionally untyped: the core runtime never reads or
    # writes it -- it's an extension point, not part of the message loop.
    # (Distinct from a message's typed `MessageSidecar`.)
    data: dict[str, Any] = field(default_factory=dict)
    _monotonic_origin: float = field(default_factory=time.monotonic, repr=False)

    def __post_init__(self) -> None:
        if self.events and not self.snapshot.messages:
            self.rebuild_snapshot()

    def _elapsed(self) -> float:
        return time.monotonic() - self._monotonic_origin

    @property
    def messages(self) -> list[Message]:
        return list(self.snapshot.messages)

    def active_context_items(self) -> list[tuple[int, Message]]:
        return self.snapshot.active_items()

    def active_context_messages(self) -> list[Message]:
        return self.snapshot.active_messages()

    def record_event(self, event: EventT) -> EventT:
        """Stamp `index`/`elapsed` on `event` and append it to the trace.

        Callers construct events with their domain fields only — for
        example `state.record_event(TurnStartEvent(agent=name))`. The
        `index` and `elapsed` fields default to placeholders and are
        replaced here so every event in `state.events` carries the same
        chronological metadata.
        """
        stamped = dataclasses.replace(
            event, index=len(self.events), elapsed=self._elapsed()
        )
        self.events.append(stamped)
        self.snapshot.apply(stamped)
        return stamped

    def record(self, message: Message) -> MessageEvent:
        """Convenience for the common `MessageEvent` path."""
        return self.record_event(MessageEvent(message=message))

    def rebuild_snapshot(self) -> StateSnapshot:
        snapshot = StateSnapshot()
        for event in self.events:
            snapshot.apply(event)
        self.snapshot = snapshot
        return snapshot

    def send(
        self,
        kind: MessageKind,
        sender: AgentName,
        target: AgentName,
        content: ContentInput = "",
        role: Role | None = None,
        sidecar: MessageSidecar | None = None,
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
            sidecar=sidecar,
        )
        self.record(message)
        return message
