from __future__ import annotations

import unittest

from simple_agent_lab import (
    Agent,
    Message,
    State,
    assistant_message,
    fork_at_message,
    message_event_indices,
    message_text,
    resume,
    user_message,
)


def _drain(events) -> None:
    for _ in events:
        pass


class _ScriptedAgent:
    """Generate function that replays a fixed list of replies in order.

    Records every `visible` context it was asked to generate from, so a
    test can assert what the model saw on the resumed turn.
    """

    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self._turn = 0
        self.seen_contexts: list[list[Message]] = []

    def __call__(self, visible: list[Message]) -> Message:
        self.seen_contexts.append(list(visible))
        text = self._replies[min(self._turn, len(self._replies) - 1)]
        self._turn += 1
        return assistant_message(text, sender="writer", target="user", kind="final")


def _run_three_message_trace() -> State:
    """A minimal completed run: task message + request/response + final."""
    agent = Agent("writer", _ScriptedAgent(["original answer"]), role="Write.")
    state, events = agent.run("first task", max_turns=3)
    _drain(events)
    return state


class ReplayTest(unittest.TestCase):
    def test_message_event_indices_maps_messages_to_events(self) -> None:
        state = _run_three_message_trace()
        positions = message_event_indices(state)

        # One event position per message, in order, each a MessageEvent.
        self.assertEqual(len(positions), len(state.messages))
        for msg_index, event_index in enumerate(positions):
            self.assertEqual(state.events[event_index].kind, "message")
            self.assertIs(state.events[event_index].message, state.messages[msg_index])

    def test_fork_truncates_to_chosen_message(self) -> None:
        state = _run_three_message_trace()
        # Message 0 is the seeded task; fork there drops everything after it.
        forked = fork_at_message(state, 0)

        self.assertEqual(len(forked.messages), 1)
        self.assertEqual(message_text(forked.messages[0]), "first task")
        # Source state is untouched.
        self.assertGreater(len(state.messages), 1)

    def test_fork_out_of_range_raises(self) -> None:
        state = _run_three_message_trace()
        with self.assertRaises(IndexError):
            fork_at_message(state, len(state.messages))

    def test_resume_regenerates_from_fork_point(self) -> None:
        state = _run_three_message_trace()

        # Resume from the task message with a different scripted reply: the
        # agent runs again and produces the new answer instead of the old.
        replay_agent = Agent("writer", _ScriptedAgent(["replayed answer"]))
        forked, events = resume(replay_agent, state, 0, max_turns=3)
        _drain(events)

        final = next(m for m in reversed(forked.messages) if m.kind == "final")
        self.assertEqual(message_text(final), "replayed answer")
        # Original run is unchanged.
        original_final = next(m for m in reversed(state.messages) if m.kind == "final")
        self.assertEqual(message_text(original_final), "original answer")

    def test_resume_opens_with_fresh_agent_start(self) -> None:
        state = _run_three_message_trace()
        replay_agent = Agent("writer", _ScriptedAgent(["x"]))
        forked, events = resume(replay_agent, state, 0)
        _drain(events)
        # The kept prefix ends with the task MessageEvent; the resumed loop
        # appends a new AgentStartEvent right after it.
        kinds = [e.kind for e in forked.events]
        self.assertIn("agent_start", kinds)
        self.assertEqual(forked.events[-1].kind, "agent_end")

    def test_edit_and_continue_replaces_tail_message(self) -> None:
        state = _run_three_message_trace()

        edited = user_message(
            "edited task context", sender="user", target="writer", kind="task"
        )
        replay_agent = _ScriptedAgent(["answer after edit"])
        forked, events = resume(
            Agent("writer", replay_agent), state, 0, replace_tail=edited
        )
        _drain(events)

        # The cut-point message now carries the edited content...
        self.assertEqual(message_text(forked.messages[0]), "edited task context")
        # ...and the resumed model saw that edited context on its first turn.
        first_seen = replay_agent.seen_contexts[0]
        self.assertEqual(message_text(first_seen[0]), "edited task context")


if __name__ == "__main__":
    unittest.main()
