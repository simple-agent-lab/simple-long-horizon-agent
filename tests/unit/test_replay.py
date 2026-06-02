from __future__ import annotations

import json
import unittest

from simple_agent_lab import (
    Agent,
    Message,
    State,
    ToolCallBlock,
    assistant_message,
    fork_at_message,
    message_event_indices,
    message_text,
    recorded_tool_calls,
    replay_side_effects,
    resume,
    user_message,
)
from simple_agent_lab.tools import AgentTool, text_result


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


class _ToolThenFinalAgent:
    """Calls `touch` once per path, in order, then emits a final message."""

    def __init__(self, paths: list[str]) -> None:
        self._paths = paths
        self._turn = 0

    def __call__(self, visible: list[Message]) -> Message:
        del visible
        if self._turn < len(self._paths):
            path = self._paths[self._turn]
            self._turn += 1
            return assistant_message(
                [ToolCallBlock(id=f"c{path}", name="touch", arguments={"path": path})],
                sender="writer",
                target="user",
                kind="step",
            )
        return assistant_message("done", sender="writer", target="user", kind="final")


def _touch_tool(fs: set[str]) -> AgentTool:
    """A stateful tool standing in for a container filesystem mutation."""

    def touch(call_id, args, abort, on_update):  # noqa: ANN001 - test stub
        del call_id, abort, on_update
        fs.add(args["path"])
        return text_result(f"created {args['path']}")

    return AgentTool(
        name="touch",
        description="create a file",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        execute=touch,
    )


class ReplaySideEffectsTest(unittest.TestCase):
    def _run_with_two_touches(self) -> tuple[State, set[str], AgentTool]:
        fs: set[str] = set()
        tool = _touch_tool(fs)
        agent = Agent("writer", _ToolThenFinalAgent(["a", "b"]), tools=(tool,))
        state, events = agent.run("make files", max_turns=5)
        _drain(events)
        return state, fs, tool

    def test_recorded_tool_calls_in_transcript_order(self) -> None:
        state, _fs, _tool = self._run_with_two_touches()
        calls = recorded_tool_calls(state.messages)
        self.assertEqual([c.name for c in calls], ["touch", "touch"])
        self.assertEqual([c.arguments["path"] for c in calls], ["a", "b"])

    def test_replay_rebuilds_external_state(self) -> None:
        state, fs, tool = self._run_with_two_touches()
        self.assertEqual(fs, {"a", "b"})  # original run created both

        # Simulate a fresh container: external state wiped.
        fs.clear()

        # Fork at the second tool-result message; replaying the kept prefix's
        # recorded calls rebuilds the filesystem to that point.
        forked = fork_at_message(state, 4)
        results = replay_side_effects(forked.messages, [tool])

        self.assertEqual(fs, {"a", "b"})
        self.assertEqual(len(results), 2)
        self.assertFalse(any(r.is_error for r in results))

    def test_unknown_tool_yields_error_result_not_raise(self) -> None:
        state, _fs, _tool = self._run_with_two_touches()
        forked = fork_at_message(state, 4)
        # No tools provided: every recorded call is unknown.
        results = replay_side_effects(forked.messages, [])
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.is_error for r in results))

    def test_resume_runs_on_fork_before_loop(self) -> None:
        state, fs, tool = self._run_with_two_touches()
        fs.clear()

        seen_lengths: list[int] = []

        def rebuild(forked: State) -> None:
            seen_lengths.append(len(forked.messages))
            replay_side_effects(forked.messages, [tool])

        # Resume from message 2 (first tool result); the hook rebuilds {"a"}
        # before the model loop continues.
        replay_agent = Agent("writer", _ToolThenFinalAgent([]), tools=(tool,))
        forked, events = resume(replay_agent, state, 2, on_fork=rebuild)
        _drain(events)

        self.assertEqual(seen_lengths, [3])  # task + step + tool-result
        self.assertEqual(fs, {"a"})

    def test_resume_from_trace_record_rebuilds_from_disk(self) -> None:
        from simple_agent_lab import resume_from_trace_record
        from simple_agent_lab.trajectory import run_trace_from_state, trace_record

        state, fs, tool = self._run_with_two_touches()
        # A persisted trace, round-tripped through JSON like a real jsonl line.
        record = json.loads(
            json.dumps(
                trace_record(
                    run_trace_from_state(state=state, trace_id="t", producer="p")
                )
            )
        )

        # Fresh environment: nothing exists, no in-memory State retained.
        fs.clear()
        replay_agent = Agent("writer", _ToolThenFinalAgent([]), tools=(tool,))
        forked, events = resume_from_trace_record(
            replay_agent,
            record,
            4,
            on_fork=lambda s: replay_side_effects(s.messages, [tool]),
        )
        _drain(events)

        self.assertEqual(fs, {"a", "b"})  # rebuilt from the recorded calls
        final = next(m for m in reversed(forked.messages) if m.kind == "final")
        self.assertEqual(message_text(final), "done")


if __name__ == "__main__":
    unittest.main()
