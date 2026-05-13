"""Stateful runtime helpers built on the tiny core loop.

`core.py` keeps the inspectable path: Agent + Message + State + context_view()
+ run(). This module holds the stateful wrapper and trace printer used by demos
and interactive callers.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterator

from .context_view import ContextPolicy
from .core import Agent, Event, EventKind, NextFn, State, TransformFn, run
from .messages import AssistantMessage, message_text
from .tools import AgentTool


Listener = Callable[[Event], None]


class AgentRuntime:
    """Small stateful wrapper around run().

    It owns State, a listener list, and a cancel flag. It intentionally does not
    own injection queues; extra user input should be recorded explicitly and then
    driven through `resume()`.
    """

    def __init__(
        self,
        agents: list[Agent],
        *,
        transform: TransformFn = lambda messages: messages,
        last: int | None = None,
        context_policy: ContextPolicy | None = None,
        tools: list[AgentTool] | None = None,
    ) -> None:
        self._agents = {agent.name: agent for agent in agents}
        self._transform = transform
        self._last = last
        self._context_policy = context_policy
        self.tools: dict[str, AgentTool] = {tool.name: tool for tool in (tools or [])}
        self._listeners: list[Listener] = []
        self._aborted = False
        self.state = State(task="")

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def abort(self) -> None:
        self._aborted = True

    def prompt(
        self,
        task: str,
        *,
        target: str,
        next_agent: NextFn,
    ) -> Iterator[Event]:
        self.state = State(task=task)
        self.state.send("task", "user", target, task)
        return self._drive(next_agent)

    def resume(self, next_agent: NextFn) -> Iterator[Event]:
        if not self.state.messages:
            raise RuntimeError("Cannot resume: state has no messages")
        return self._drive(next_agent)

    def _drive(self, next_agent: NextFn) -> Iterator[Event]:
        self._aborted = False
        stream = run(
            self._agents,
            self.state,
            next_agent,
            transform=self._transform,
            last=self._last,
            context_policy=self._context_policy,
            tools=self.tools,
            abort=lambda: self._aborted,
        )
        for event in stream:
            for listener in list(self._listeners):
                listener(event)
            yield event
            if self._aborted:
                end_event = self.state.emit(EventKind.AGENT_END, reason="aborted")
                for listener in list(self._listeners):
                    listener(end_event)
                yield end_event
                return


def print_trace(state: State, *, raw: bool = False) -> None:
    """Print the standardized trace.

    `raw=True` also dumps each model call's `raw` payload -- the provider
    request snapshot (with messages history pruned) and the SDK response
    dump -- so the trace doubles as an HTTP-level diff tool.
    """
    print("\ntrace")
    print("-----")
    for event in state.events:
        kind = event.kind.value
        if event.kind is EventKind.MESSAGE:
            message = event.message
            if message is None:
                continue
            route = f"{message.sender} -> {message.target}"
            print(
                f"{event.index:02d} {kind:<21} {message.kind:<10} "
                f"{route:<24} {message_text(message)}"
            )
            extra = (message.data or {}).get("extra")
            if extra:
                preview = ", ".join(f"{k}={v!r}" for k, v in extra.items())
                print(f"   {'extra':<21} {preview[:200]}")
            if isinstance(message, AssistantMessage):
                for thinking_block in message.thinking:
                    preview = thinking_block.text.replace("\n", " ")
                    if len(preview) > 200:
                        preview = preview[:200] + "..."
                    tag = "redacted_thinking" if thinking_block.redacted else "thinking"
                    print(f"   {tag:<21} {preview}")
                if raw:
                    raw_payload = (message.data or {}).get("raw")
                    if raw_payload:
                        _print_raw(raw_payload)
        elif event.kind is EventKind.MODEL_REQUEST:
            candidate = event.data.get("candidate_id")
            suffix = f" candidate={candidate}" if candidate is not None else ""
            print(
                f"{event.index:02d} {kind:<21} "
                f"agent={event.data.get('agent')} "
                f"visible={event.data.get('visible_count')} "
                f"llm_messages={event.data.get('llm_message_count')}{suffix}"
            )
        elif event.kind is EventKind.MODEL_RESPONSE:
            candidate = event.data.get("candidate_id")
            suffix = f" candidate={candidate}" if candidate is not None else ""
            print(
                f"{event.index:02d} {kind:<21} "
                f"agent={event.data.get('agent')} "
                f"kind={event.data.get('output_kind')} "
                f"target={event.data.get('target')} "
                f"tool_calls={event.data.get('tool_call_count')}{suffix}"
            )
        else:
            extras = " ".join(f"{key}={value}" for key, value in event.data.items())
            print(f"{event.index:02d} {kind:<21} {extras}")


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
