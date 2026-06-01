"""Fork a recorded run and re-run an agent from any point in its trace.

The runtime is append-only and stateless between turns: `run()` rebuilds
the model context every turn from `state.active_context_messages()` (see
`core.run`), and `State.__post_init__` reconstructs the derived snapshot
from a list of events. Together those two facts make mid-trace replay
nearly free — slice the event log at a chosen message boundary, drop a
fresh `State` around the kept prefix, and hand it back to `run()`.

The natural "position" in a trace is a **message index** (an offset into
`state.messages`), because the loop is driven by messages, not raw events.
Forking after a non-assistant message (a task, tool-result, or injected
user message) leaves the model as the next actor, so the resumed loop
calls `agent.generate(...)` again and the divergence is yours to inspect.

`fork_at_message` optionally *replaces* the message at the cut point, which
turns replay into edit-and-continue: change a tool result (or the model's
own prior output) and watch how the agent reacts.

Note on side effects: this forks the in-memory transcript only. Tools that
mutate state outside the message log (a container filesystem, a database)
are NOT rewound — to replay those faithfully you must restore that external
state to the chosen point first.

The `replay_side_effects` helper covers the common "rebuild, don't snapshot"
strategy: re-execute the tool calls recorded in the kept prefix against a
fresh environment (e.g. a container booted from the run's baseline image),
applying their side effects again without ever calling the model. Wire it
into `resume` via the `on_fork` hook and the external world is restored to
the fork point before the model loop continues. For pure-function tools and
model-only debugging, the transcript is the whole story and no rebuild is
needed.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Callable, Iterator

from .core import Agent, _execute_one, run
from .messages import Message, ToolCallBlock, message_tool_calls
from .protocols import Event, MessageEvent
from .state import State
from .tools import AbortFlag, AgentTool, ToolResult, ToolUpdateFn


def message_event_indices(state: State) -> list[int]:
    """Event-log positions that produced each message, in message order.

    `result[k]` is the index into `state.events` of the `MessageEvent`
    whose message is `state.messages[k]`. Handy for mapping a chosen
    message back onto the raw event stream (e.g. in a trace viewer).
    """
    return [
        i for i, event in enumerate(state.events) if isinstance(event, MessageEvent)
    ]


def fork_at_message(
    state: State,
    message_index: int,
    *,
    replace_tail: Message | None = None,
) -> State:
    """Return a new `State` truncated to the first `message_index + 1` messages.

    Every event after the chosen `MessageEvent` is dropped — later model
    outputs, tool executions, compression, and turn bookkeeping all go
    away — so the fork looks exactly as the run did the instant that
    message was recorded. The new `State` rebuilds its snapshot from the
    kept events via `State.__post_init__`.

    `replace_tail`, when given, swaps the message at the cut point for a new
    one (edit-and-continue): tweak a tool result or the model's prior turn,
    then resume to see the agent take a different path.

    Raises `IndexError` if `message_index` is out of range.
    """
    msg_positions = message_event_indices(state)
    if not 0 <= message_index < len(msg_positions):
        raise IndexError(
            f"message_index {message_index} out of range "
            f"(run has {len(msg_positions)} messages)"
        )

    cut = msg_positions[message_index]  # index of the boundary MessageEvent
    # `dataclasses.replace` with no changes copies each frozen event so the
    # fork never shares mutable event instances with the source state.
    kept: list[Event] = [
        dataclasses.replace(event) for event in state.events[: cut + 1]
    ]

    if replace_tail is not None:
        kept[cut] = dataclasses.replace(kept[cut], message=replace_tail)

    return State(task=state.task, events=kept, data=dict(state.data))


def recorded_tool_calls(messages: Sequence[Message]) -> list[ToolCallBlock]:
    """Every tool call recorded in `messages`, flattened in transcript order.

    A parallel-tool assistant turn contributes its calls in the order they
    appear in the message, so replaying the result preserves the original
    sequence of side effects.
    """
    calls: list[ToolCallBlock] = []
    for message in messages:
        calls.extend(message_tool_calls(message))
    return calls


def replay_side_effects(
    messages: Sequence[Message],
    tools: Sequence[AgentTool],
    *,
    abort: AbortFlag = lambda: False,
    on_update: ToolUpdateFn | None = None,
) -> list[ToolResult]:
    """Re-execute the tool calls recorded in `messages` for their side effects.

    This is the "rebuild, don't snapshot" half of replay: rather than
    persisting a filesystem/container checkpoint at every step, boot a fresh
    environment from the run's baseline and replay the recorded calls to
    bring the external world back to the fork point. The returned
    `ToolResult`s come from *this* execution, so a caller can diff them
    against the recorded `ToolResultBlock`s to surface nondeterminism
    (a command that read the clock, the network, or a random source).

    Calls run **sequentially in recorded order** — side effects often depend
    on their predecessors (a write after a `cd`, an edit after a checkout),
    so this deliberately ignores each tool's `execution_mode` and never
    parallelizes. Unknown tool names yield an error result rather than
    raising, mirroring `dispatch_tool_calls`.

    Pass `messages` from the *forked* state (the kept prefix). Fork at a
    user / tool-result boundary so every recorded call's effect had in fact
    been applied by that point; forking mid-turn (an assistant message whose
    calls were not yet dispatched) would over-apply that turn's effects.
    """
    tool_by_name = {tool.name: tool for tool in tools}
    return [
        _execute_one(call, tool_by_name, abort, on_update)
        for call in recorded_tool_calls(messages)
    ]


def resume(
    agent: Agent,
    state: State,
    message_index: int,
    *,
    max_turns: int = 10,
    abort: AbortFlag = lambda: False,
    replace_tail: Message | None = None,
    on_fork: Callable[[State], None] | None = None,
) -> tuple[State, Iterator[Event]]:
    """Fork `state` at `message_index` and re-run `agent` from there.

    Returns `(forked_state, events)` mirroring `Agent.run`: the original
    `state` is never mutated, and iterating `events` drives the resumed
    loop while populating `forked_state`. The resumed trace opens with a
    fresh `AgentStartEvent`, so the replay boundary is visible in the new
    event log.

    Pass `replace_tail` to edit the message at the cut point before
    resuming (see `fork_at_message`).

    `on_fork`, when given, is called with the forked state *before* the
    model loop starts — the seam for restoring external state to the fork
    point (e.g. ``on_fork=lambda s: replay_side_effects(s.messages,
    agent.tools)`` to rebuild a container filesystem). `resume` stays
    neutral about how the world is rewound; the hook owns that.
    """
    forked = fork_at_message(state, message_index, replace_tail=replace_tail)
    if on_fork is not None:
        on_fork(forked)
    events = run(agent, forked, max_turns=max_turns, abort=abort)
    return forked, events
