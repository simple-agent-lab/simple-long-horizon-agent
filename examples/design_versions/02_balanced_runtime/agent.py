"""Stateful runtime wrapper.

Mirrors pi-mono's `Agent` class: subscribe to events, steer mid-run, queue
follow-ups, abort. The pure-functional `run()` in core.py stays available for
callers that prefer to drive the loop themselves.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Iterator, Optional

from core import (
    Agent,
    AfterAct,
    BeforeAct,
    ConvertToLlm,
    Event,
    Message,
    NextFn,
    Queue,
    QueueMode,
    State,
    TransformFn,
    run,
)


Listener = Callable[[Event], None]


class AgentRuntime:
    """Stateful wrapper around `core.run`.

    Holds the State, two injection queues, an event subscription list, and a
    cancel flag. Use `prompt()` to start a run from a task and `continue_()`
    to resume from the existing state without adding a new task.
    """

    def __init__(
        self,
        agents: list[Agent],
        *,
        transform: TransformFn = lambda messages: messages,
        convert_to_llm: Optional[ConvertToLlm] = None,
        last: Optional[int] = None,
        before_act: Optional[BeforeAct] = None,
        after_act: Optional[AfterAct] = None,
        steering_mode: QueueMode = "one-at-a-time",
        follow_up_mode: QueueMode = "one-at-a-time",
    ) -> None:
        self._agents = {agent.name: agent for agent in agents}
        self._transform = transform
        self._convert_to_llm = convert_to_llm
        self._last = last
        self._before_act = before_act
        self._after_act = after_act
        self._steering = Queue(mode=steering_mode)
        self._follow_up = Queue(mode=follow_up_mode)
        self._listeners: list[Listener] = []
        self._aborted = False
        self.state = State(task="")

    # ------------------------------------------------------------------ events

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def _emit_to_listeners(self, event: Event) -> None:
        for listener in list(self._listeners):
            listener(event)

    # ----------------------------------------------------------------- queues

    def steer(self, message: Message) -> None:
        """Inject `message` after the current turn finishes."""
        self._steering.push(message)

    def follow_up(self, message: Message) -> None:
        """Inject `message` only after the agent would otherwise stop."""
        self._follow_up.push(message)

    def clear_queues(self) -> None:
        self._steering.clear()
        self._follow_up.clear()

    @property
    def steering_mode(self) -> QueueMode:
        return self._steering.mode

    @steering_mode.setter
    def steering_mode(self, mode: QueueMode) -> None:
        self._steering = replace(self._steering, mode=mode)

    @property
    def follow_up_mode(self) -> QueueMode:
        return self._follow_up.mode

    @follow_up_mode.setter
    def follow_up_mode(self, mode: QueueMode) -> None:
        self._follow_up = replace(self._follow_up, mode=mode)

    # ------------------------------------------------------------------ abort

    def abort(self) -> None:
        """Stop the run after the current event yields."""
        self._aborted = True

    # ----------------------------------------------------------------- runs

    def prompt(
        self,
        task: str,
        *,
        target: str,
        next_agent: NextFn,
    ) -> Iterator[Event]:
        """Start a fresh run with a new task message addressed at `target`."""
        self.state = State(task=task)
        self.state.send("task", "user", target, task)
        return self._drive(next_agent)

    def continue_(self, next_agent: NextFn) -> Iterator[Event]:
        """Resume from the existing state. The last message must not be from an agent."""
        if not self.state.messages:
            raise RuntimeError("Cannot continue: state has no messages")
        return self._drive(next_agent)

    def _drive(self, next_agent: NextFn) -> Iterator[Event]:
        self._aborted = False
        stream = run(
            self._agents,
            self.state,
            next_agent,
            transform=self._transform,
            convert_to_llm=self._convert_to_llm,
            last=self._last,
            before_act=self._before_act,
            after_act=self._after_act,
            steering=self._steering,
            follow_up=self._follow_up,
        )
        for event in stream:
            self._emit_to_listeners(event)
            yield event
            if self._aborted:
                self.state.emit("agent_end", reason="aborted")
                return
