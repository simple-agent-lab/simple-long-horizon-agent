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
state to the chosen point first (see the eval/container notes in the docs).
For pure-function tools and model-only debugging, the transcript is the
whole story and no extra work is needed.
"""

from __future__ import annotations

import dataclasses
from typing import Iterator

from .core import Agent, run
from .messages import Message
from .protocols import Event, MessageEvent
from .state import State
from .tools import AbortFlag


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


def resume(
    agent: Agent,
    state: State,
    message_index: int,
    *,
    max_turns: int = 10,
    abort: AbortFlag = lambda: False,
    replace_tail: Message | None = None,
) -> tuple[State, Iterator[Event]]:
    """Fork `state` at `message_index` and re-run `agent` from there.

    Returns `(forked_state, events)` mirroring `Agent.run`: the original
    `state` is never mutated, and iterating `events` drives the resumed
    loop while populating `forked_state`. The resumed trace opens with a
    fresh `AgentStartEvent`, so the replay boundary is visible in the new
    event log.

    Pass `replace_tail` to edit the message at the cut point before
    resuming (see `fork_at_message`).
    """
    forked = fork_at_message(state, message_index, replace_tail=replace_tail)
    events = run(agent, forked, max_turns=max_turns, abort=abort)
    return forked, events
