"""Small memory boundary.

Memory is deliberately an assembly source, not a runtime. A memory
implementation can expose ordinary AgentTool values and declare lifecycle hooks
for a future hook-aware runtime to consume.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from simple_agent_lab.messages import (
    ContentInput,
    Message,
    normalize_content,
    system_message,
    text_of,
)
from simple_agent_lab.state import State
from simple_agent_lab.tools import AgentTool


@dataclass(frozen=True)
class MemoryContext:
    """Run metadata passed to a memory extension."""

    agent: str
    task: str
    session_id: str = ""
    run_id: str = ""
    memory_name: str = ""
    step_index: int | None = None
    state: State | None = None
    data: Mapping[str, Any] = field(default_factory=dict)


class Memory:
    """Base class for optional memory behavior around an agent run."""

    def initial(self, ctx: MemoryContext) -> tuple[Message, ...]:
        """Messages to record before the run starts."""
        del ctx
        return ()

    def recall(self, ctx: MemoryContext, query: str) -> tuple[Message, ...]:
        """Messages to record before a model request."""
        del ctx, query
        return ()

    def tools(self, ctx: MemoryContext) -> tuple[AgentTool, ...]:
        """Additional tools to expose for this run."""
        del ctx
        return ()

    def record(self, ctx: MemoryContext, messages: tuple[Message, ...]) -> None:
        """Observe messages recorded during one completed turn."""
        del ctx, messages

    def finish(self, ctx: MemoryContext) -> None:
        """Best-effort post-run learning or persistence."""
        del ctx

    def bind(self, ctx: MemoryContext) -> "MemoryBinding":
        """Produce tools and future runtime hooks for this memory instance."""

        def context_for(state: State | None = None) -> MemoryContext:
            return _context_with_state(ctx, state)

        def before_run(state: State) -> tuple[Message, ...]:
            try:
                return tuple(self.initial(context_for(state)))
            except Exception:
                return ()

        def before_model_request(state: State) -> tuple[Message, ...]:
            query = _latest_user_text(state) or _task_text(state.task)
            try:
                return tuple(self.recall(context_for(state), query))
            except Exception:
                return ()

        def after_turn(state: State, messages: tuple[Message, ...]) -> None:
            try:
                self.record(context_for(state), messages)
            except Exception:
                pass

        def after_run(state: State) -> None:
            try:
                self.finish(context_for(state))
            except Exception:
                pass

        return MemoryBinding(
            tools=self.tools(ctx),
            hooks=MemoryHooks(
                before_run=before_run,
                before_model_request=before_model_request,
                after_turn=after_turn,
                after_run=after_run,
            ),
        )


class NoMemory(Memory):
    """No-op memory implementation."""


BeforeRunHook = Callable[[State], Iterable[Message]]
BeforeModelRequestHook = Callable[[State], Iterable[Message]]
AfterTurnHook = Callable[[State, tuple[Message, ...]], None]
AfterRunHook = Callable[[State], None]


@dataclass(frozen=True)
class MemoryHooks:
    """Future runtime hooks requested by one memory instance."""

    before_run: BeforeRunHook | None = None
    before_model_request: BeforeModelRequestHook | None = None
    after_turn: AfterTurnHook | None = None
    after_run: AfterRunHook | None = None


@dataclass(frozen=True)
class MemoryBinding:
    """Assembly material produced by binding one memory instance."""

    tools: tuple[AgentTool, ...] = ()
    hooks: MemoryHooks = field(default_factory=MemoryHooks)


def memory_context_message(text: str, *, target: str) -> Message:
    """Build the default model-visible memory context message."""

    return system_message(text, sender="memory", target=target, kind="context")


def _context_with_state(ctx: MemoryContext, state: State | None) -> MemoryContext:
    if state is None:
        return ctx
    return MemoryContext(
        agent=ctx.agent,
        task=_task_text(state.task),
        session_id=ctx.session_id,
        run_id=ctx.run_id,
        memory_name=ctx.memory_name,
        step_index=ctx.step_index,
        state=state,
        data=ctx.data,
    )


def _task_text(task: ContentInput) -> str:
    if isinstance(task, str):
        return task
    return text_of(normalize_content(task)).strip()


def _latest_user_text(state: State) -> str:
    for message in reversed(state.messages):
        if message.role == "user":
            return _message_text(message)
    return ""


def _message_text(message: Message) -> str:
    parts = []
    for block in message.content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts).strip()
