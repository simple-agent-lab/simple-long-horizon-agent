"""Lifecycle hooks — the one place the agent loop asks for permission.

Events (`protocols.py`) are observe-only: `run()` yields them, consumers read
them, nothing flows back. Hooks are the opposite arrow. At a few named points
in the loop, the runtime calls every registered hook with a `HookContext` and
acts on the `HookDecision` it returns. Hooks are **append-only** — like the
event log itself, they never edit an existing message; they only add. So a
hook can wave a tool call through, **block** it (which adds an error result),
or **emit** messages (which add context) — never modify what's already there.

The shape deliberately mirrors compression: a `Hook` is a pluggable callable
that returns a frozen decision object (or `None` for "no opinion"), exactly as
a `CompressionStrategy` returns a `CompressionDecision | None`. The registry is
just a plain map from point to hooks (no wrapper type — like `tools` is a bare
tuple). A thin `fire_hooks` reduces the decisions and emits one observable
`HookFiredEvent` so hook activity still shows up in the trace.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, TypeAlias

from .messages import AgentName, Message, ToolCallBlock
from .protocols import Event, HookFiredEvent

if TYPE_CHECKING:
    from .state import State


class HookPoint(str, Enum):
    """A named place in the loop where the runtime consults hooks.

    String-valued (like `EventKind`) so callers can match on the value and so
    it serializes cleanly into the trace via the `HookFiredEvent.point` field.
    """

    SESSION_START = "session_start"
    SESSION_END = "session_end"
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, kw_only=True)
class HookContext:
    """Read-only payload handed to a hook.

    `point` discriminates which firing this is; fields not relevant to a point
    are left `None` (one payload with optional fields, not a class per case).
    `state` is passed so a hook can inspect the transcript / `state.data`
    scratchpad — the only sanctioned way to affect the loop is the returned
    `HookDecision`, not mutating `state`.
    """

    point: HookPoint
    agent: AgentName
    state: State
    # PRE_TOOL_USE / POST_TOOL_USE only. The tool name is `tool_call.name`.
    tool_call: ToolCallBlock | None = None


@dataclass(frozen=True, kw_only=True)
class HookDecision:
    """What a hook returns. `None` from a hook is shorthand for this default.

    Hooks are **append-only**: they never edit an existing message or tool
    call, they only add. So the vocabulary is just two additive moves:

    - block: a non-empty `block_reason`. At PRE_TOOL_USE, the tool does not run;
      the reason is *added* as an (error) tool result so the model can
      self-correct, exactly like a tool that raised. (Blocking always carries a
      reason — that's why it *is* the reason, not a separate flag.) This is the
      transparent way to "redirect" a call: the model retries, every step on the
      record. Other hook points ignore `block_reason`.
    - emit: `emit_messages` are *added* to the transcript. The runtime records
      them as `MessageEvent`s, so the append rides the loop's normal yield
      discipline and replays cleanly (see `fire_hooks`). Only honored at points
      outside tool dispatch — see the `PRE_TOOL_USE` note in `fire_hooks`.

    Returning `None` (or a bare `HookDecision()`) is pure observation.
    """

    block_reason: str = ""
    emit_messages: tuple[Message, ...] = ()


# A hook: payload in, optional decision out. `None` means "no opinion" — the
# same convention as a `CompressionStrategy` returning `None`.
Hook = Callable[[HookContext], "HookDecision | None"]

# The registry: a bare map from point to its ordered hooks. An empty map (the
# `Agent.hooks` default) is a zero-cost no-op, which is what keeps hooks fully
# backward compatible. Build one as a dict literal:
#     {HookPoint.PRE_TOOL_USE: [my_hook]}
HookMap: TypeAlias = Mapping[HookPoint, Sequence[Hook]]


def fire_hooks(
    hooks: HookMap,
    ctx: HookContext,
    state: State,
) -> tuple[HookDecision, list[Event]]:
    """Run every hook for `ctx.point`, reduce to one decision, record + yield.

    Returns `(decision, events)` — the same `list[Event]` contract as
    `maybe_compress_context`, so a generator caller yields the events and then
    acts on `decision.block_reason`. No hook for the point → a no-op (default
    decision, empty list), so unhooked runs record nothing extra.

    Reduction follows the append-only vocabulary: `block_reason` is selective
    (first non-empty wins and short-circuits — a veto is terminal), while
    `emit_messages` is additive (collected from every hook, in order). Emitted
    messages are recorded here as `MessageEvent`s and returned for the caller
    to yield, so the append stays inside the loop's yield discipline (no drift
    between the yielded stream and `state.events`) and replays cleanly.

    Emission is skipped at `PRE_TOOL_USE`: a message inserted between a
    tool_call and its result would orphan the pair on the wire. `POST_TOOL_USE`
    fires only after the tool-result bundle has been appended, so emitted
    reminder/context messages are safe and visible on the next model turn.
    """
    point_hooks = hooks.get(ctx.point, ())
    if not point_hooks:
        return HookDecision(), []

    block_reason = ""
    emitted: list[Message] = []
    for hook in point_hooks:
        outcome = hook(ctx)
        if outcome is None:
            continue
        emitted.extend(outcome.emit_messages)
        if outcome.block_reason:
            block_reason = outcome.block_reason
            break

    if ctx.point == HookPoint.PRE_TOOL_USE:
        emitted = []

    events: list[Event] = [
        state.record_event(
            HookFiredEvent(
                point=str(ctx.point),
                agent=ctx.agent,
                target=ctx.tool_call.name if ctx.tool_call else "",
                block_reason=block_reason,
                emitted=len(emitted),
            )
        )
    ]
    events.extend(state.record(message) for message in emitted)
    return HookDecision(block_reason=block_reason), events
