"""Small memory boundary.

Memory is deliberately an assembly source, not a runtime. A memory
implementation can expose ordinary AgentTool values and declare lifecycle hooks
for the core runtime to consume.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from simple_long_horizon_agent.hooks import (
    HookContext,
    HookDecision,
    HookMap,
    HookPoint,
)
from simple_long_horizon_agent.messages import (
    ContentInput,
    Message,
    normalize_content,
    runtime_message,
    text_of,
)
from simple_long_horizon_agent.state import State
from simple_long_horizon_agent.tools import AgentTool


@dataclass(frozen=True)
class MemoryContext:
    """Run metadata passed to memory."""

    agent: str
    task: str
    session_id: str = ""
    run_id: str = ""
    memory_name: str = ""
    state: State | None = None
    data: Mapping[str, Any] = field(default_factory=dict)


class Memory:
    """Base class for optional memory behavior around an agent run.

    The surface is exactly what the core hook points can drive today:
    `initial(...)` at `SESSION_START`, `tools(...)` at assembly, and
    `finish(...)` at `SESSION_END`. Add a method only when a hook point
    exists to call it.
    """

    def initial(self, ctx: MemoryContext) -> tuple[Message, ...]:
        """Messages to record before the run starts."""
        del ctx
        return ()

    def tools(self, ctx: MemoryContext) -> tuple[AgentTool, ...]:
        """Additional tools to expose for this run."""
        del ctx
        return ()

    def finish(self, ctx: MemoryContext) -> None:
        """Best-effort post-run learning or persistence."""
        del ctx

    def bind(self, ctx: MemoryContext) -> "MemoryBinding":
        """Produce tools and runtime hooks for this memory instance."""

        def context_for(state: State | None = None) -> MemoryContext:
            return _context_with_state(ctx, state)

        def on_session_start(hook_ctx: HookContext) -> HookDecision | None:
            try:
                messages = tuple(self.initial(context_for(hook_ctx.state)))
            except Exception as exc:
                # Best-effort: a memory initial() failure must not crash the run,
                # but it must not vanish either. The project has no logger, so
                # surface a compact note as a recorded MessageEvent (via
                # fire_hooks) — keeping the skipped injection visible in the
                # trace instead of silently returning None.
                return HookDecision(
                    emit_messages=(
                        memory_context_message(
                            "Memory initialization was skipped after an error: "
                            f"{type(exc).__name__}: {exc}",
                            target=ctx.agent,
                        ),
                    )
                )
            if not messages:
                return None
            return HookDecision(emit_messages=messages)

        def on_session_end(hook_ctx: HookContext) -> HookDecision | None:
            try:
                self.finish(context_for(hook_ctx.state))
            except Exception:
                pass
            return None

        return MemoryBinding(
            tools=self.tools(ctx),
            hooks={
                HookPoint.SESSION_START: (on_session_start,),
                HookPoint.SESSION_END: (on_session_end,),
            },
        )


@dataclass(frozen=True)
class MemoryBinding:
    """Assembly material produced by binding one memory instance."""

    tools: tuple[AgentTool, ...] = ()
    hooks: HookMap = field(default_factory=dict)


def memory_context_message(text: str, *, target: str) -> Message:
    """Build the default model-visible memory context message."""

    return runtime_message(text, sender="memory", target=target, kind="context")


def _context_with_state(ctx: MemoryContext, state: State | None) -> MemoryContext:
    if state is None:
        return ctx
    return replace(
        ctx,
        task=_task_text(state.task),
        state=state,
    )


def _task_text(task: ContentInput) -> str:
    if isinstance(task, str):
        return task
    return text_of(normalize_content(task)).strip()
