"""Append-only runtime state and its derived snapshot."""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar
from uuid import uuid4

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
    # The active context: indices into `messages` that survive compression
    # and feed the next model step. `None` means no compression has run yet,
    # so all messages are active. (A later visibility filter in
    # `build_context_view` narrows this to the *visible* context the model
    # actually sees -- active is the compression stage, visible the next.)
    active_context_indices: list[int] | None = None
    # Indices into `messages` that are compaction outputs (an N->1 summary or a
    # 1->1 rewrite spliced back at an earlier slot). The explicit record of
    # "where a compaction happened" that `compression.runtime._active_context_tokens`
    # reads to decide whether a usage baseline predates the newest fold -- a
    # message recorded before a compaction boundary carries a stale window size.
    # Tracked here rather than inferred from index ordering so re-pointing the
    # active context for any reason can never confuse the sizing.
    compaction_indices: set[int] = field(default_factory=set)

    def apply(self, event: Event) -> None:
        if isinstance(event, MessageEvent):
            message_index = len(self.messages)
            self.messages.append(event.message)
            if self.active_context_indices is not None:
                self.active_context_indices.append(message_index)
            return
        if isinstance(event, ContextCompressionEvent):
            # Copy: the snapshot mutates this list as new messages arrive;
            # the event's field must stay a stable historical record.
            self.active_context_indices = list(event.active_context_indices)
            self.compaction_indices.add(event.summary_message_index)

    def active_context_items(self) -> list[tuple[int, Message]]:
        indices = self.active_context_indices
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

    def active_context_messages(self) -> list[Message]:
        return [message for _, message in self.active_context_items()]


@dataclass
class State:
    # `str` or a sequence of content blocks (multimodal task: text + images).
    task: ContentInput
    events: list[Event] = field(default_factory=list)
    snapshot: StateSnapshot = field(default_factory=StateSnapshot)
    # Open scratchpad for harnesses/experiments to hang run-level metadata
    # on a State (e.g. the SWE-bench runner stashes `model_patch`,
    # `workspace`). Intentionally untyped: the core runtime never reads or
    # writes it -- it's an extension point, not part of the message loop.
    # (Distinct from a message's typed `MessageSidecar`.)
    data: dict[str, Any] = field(default_factory=dict)
    # Per-run seed for message identity. Deterministic message `uuid`s are
    # derived from it plus message order (`run_id.m{index}`), so a run is
    # reproducible end to end when a fixed `run_id` is supplied (tests, golden
    # fixtures) while distinct runs stay globally unique by default.
    run_id: str = field(default_factory=lambda: uuid4().hex)
    _monotonic_origin: float = field(default_factory=time.monotonic, repr=False)

    def __post_init__(self) -> None:
        if self.events and not self.snapshot.messages:
            self.rebuild_snapshot()

    def message_uuid(self, message_index: int) -> str:
        """Deterministic identity for the message at `message_index`."""
        return f"{self.run_id}.m{message_index}"

    @property
    def compaction_indices(self) -> set[int]:
        """Message indices that are compaction outputs (see `StateSnapshot`)."""
        return self.snapshot.compaction_indices

    def _elapsed(self) -> float:
        return time.monotonic() - self._monotonic_origin

    @property
    def messages(self) -> list[Message]:
        # A fresh list each call, so a concurrent reader — e.g. the `recall`
        # tool running in the parallel tool pool — gets a length-stable
        # snapshot to index into. A shallow copy is both enough and correct
        # here: `Message`s are frozen and are never mutated after they are
        # recorded, and `list()` is a single GIL-atomic copy. A deepcopy would
        # clone immutable data for no safety gain and, being non-atomic Python,
        # would widen rather than close the interleaving window.
        return list(self.snapshot.messages)

    def active_context_items(self) -> list[tuple[int, Message]]:
        return self.snapshot.active_context_items()

    def active_context_messages(self) -> list[Message]:
        return self.snapshot.active_context_messages()

    def record_event(self, event: EventT) -> EventT:
        """Stamp `index`/`elapsed` on `event` and append it to the trace.

        Callers construct events with their domain fields only — for
        example `state.record_event(TurnStartEvent(agent=name))`. The
        `index` and `elapsed` fields default to placeholders and are
        replaced here so every event in `state.events` carries the same
        chronological metadata.
        """
        fields: dict[str, Any] = {"index": len(self.events), "elapsed": self._elapsed()}
        # Stamp message identity from run_id + position, leaving any value a
        # replayed/loaded event already carries untouched. `parent_uuid` links
        # to the message recorded just before this one (None for the first).
        if isinstance(event, MessageEvent) and not event.uuid:
            message_index = len(self.snapshot.messages)
            fields["uuid"] = self.message_uuid(message_index)
            fields["parent_uuid"] = (
                self.message_uuid(message_index - 1) if message_index > 0 else None
            )
        stamped = dataclasses.replace(event, **fields)
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
