"""Recoverable compression and agent-controlled compaction.

Two halves of one loop (ADR recoverable-compression-and-agent-compaction):

- `make_recall_tool` reads original transcript messages back off the
  append-only `State`, using the indices that compression summaries cite.
- `make_compact_control` pairs a `compact` tool (the model requests
  compression itself, with its own replacement summary) with the strategy
  that applies the request at the next turn start.
"""

from __future__ import annotations

import unittest

from simple_agent_lab import (
    Agent,
    ContextCompressionEvent,
    ContextPolicy,
    Message,
    State,
    TextBlock,
    ToolCallBlock,
    assistant_message,
    make_compact_control,
    run,
    text_of,
)
from simple_agent_lab.compression import format_index_ranges, source_note
from simple_agent_lab.tools import make_recall_tool, tool_result_text


def _no_abort() -> bool:
    return False


class IndexCitationTest(unittest.TestCase):
    def test_format_index_ranges_collapses_runs(self) -> None:
        self.assertEqual(format_index_ranges((2, 3, 4, 7)), "2-4, 7")
        self.assertEqual(format_index_ranges((5,)), "5")
        self.assertEqual(format_index_ranges((9, 1, 2)), "1-2, 9")
        self.assertEqual(format_index_ranges(()), "")

    def test_source_note_cites_indices(self) -> None:
        note = source_note((1, 2, 3))
        self.assertIn("transcript messages 1-3", note)
        self.assertIn("recall", note)


class RecallToolTest(unittest.TestCase):
    def _state(self) -> State:
        state = State("t")
        state.send("task", "user", "worker", "t")  # 0
        state.send("message", "user", "worker", "the secret is 42")  # 1
        state.record(
            assistant_message(
                [TextBlock("calling"), ToolCallBlock("c1", "echo", {"x": 1})],
                sender="worker",
                target="user",
                kind="step",
            )
        )  # 2
        return state

    def test_returns_original_messages_by_index(self) -> None:
        state = self._state()
        tool = make_recall_tool(state)
        result = tool.execute("call", {"indices": [1, 2]}, _no_abort, None)
        self.assertFalse(result.is_error)
        text = tool_result_text(result)
        self.assertIn("[transcript message 1]", text)
        self.assertIn("the secret is 42", text)
        self.assertIn("[transcript message 2]", text)
        self.assertIn("tool_call c1 -> echo", text)

    def test_recalls_messages_compressed_out_of_the_active_view(self) -> None:
        # Compression re-points the active view, but the append-only
        # transcript keeps the original — recall must still find it.
        state = self._state()
        state.record_event(
            ContextCompressionEvent(
                agent="worker",
                summary_message_index=2,
                compressed_message_indices=[1],
                active_context_indices=[0, 2],
                before_tokens=100,
                after_tokens=10,
            )
        )
        self.assertNotIn(
            "the secret is 42",
            [text_of(message.content) for message in state.active_context_messages()],
        )
        tool = make_recall_tool(state)
        result = tool.execute("call", {"indices": [1]}, _no_abort, None)
        self.assertIn("the secret is 42", tool_result_text(result))

    def test_rejects_out_of_range_and_malformed_indices(self) -> None:
        state = self._state()
        tool = make_recall_tool(state)

        result = tool.execute("call", {"indices": [99]}, _no_abort, None)
        self.assertTrue(result.is_error)
        self.assertIn("out of range", tool_result_text(result))
        self.assertIn("0-2", tool_result_text(result))

        for bad_args in (
            {},
            {"indices": []},
            {"indices": "1"},
            {"indices": [True]},
            {"indices": [1.5]},
            {"indices": [-1]},
        ):
            result = tool.execute("call", bad_args, _no_abort, None)
            self.assertTrue(result.is_error, msg=repr(bad_args))

    def test_truncates_long_messages_per_message(self) -> None:
        state = State("t")
        state.send("task", "user", "worker", "t")
        state.send("message", "user", "worker", "y" * 500)
        tool = make_recall_tool(state, max_chars_per_message=100)
        result = tool.execute("call", {"indices": [1]}, _no_abort, None)
        text = tool_result_text(result)
        self.assertIn("truncated; 500 chars total", text)
        self.assertNotIn("y" * 200, text)

    def test_per_call_budget_bounds_a_multi_index_recall(self) -> None:
        # Without a per-call cap a single recall could re-inject
        # max_chars_per_message * max_indices worth of text, undoing the
        # compaction that just ran. The budget stops the batch early.
        state = State("t")
        state.send("task", "user", "worker", "t")
        for _ in range(5):
            state.send("message", "user", "worker", "z" * 500)  # indices 1-5
        tool = make_recall_tool(
            state, max_chars_per_message=200, max_total_chars=400
        )
        result = tool.execute("call", {"indices": [1, 2, 3, 4, 5]}, _no_abort, None)
        text = tool_result_text(result)
        # First message is always returned; the rest is summarized as truncated.
        self.assertIn("recall truncated", text)
        self.assertIn("4 more message(s)", text)
        # details report only what was actually returned, not all requested.
        self.assertEqual(result.details["indices"], [1])

    def test_first_message_returned_even_when_it_alone_fills_the_budget(self) -> None:
        state = State("t")
        state.send("task", "user", "worker", "t")
        state.send("message", "user", "worker", "z" * 500)  # index 1
        tool = make_recall_tool(
            state, max_chars_per_message=400, max_total_chars=400
        )
        result = tool.execute("call", {"indices": [1]}, _no_abort, None)
        self.assertFalse(result.is_error)
        self.assertEqual(result.details["indices"], [1])
        self.assertNotIn("recall truncated", tool_result_text(result))

    def test_rejects_total_budget_below_per_message_cap(self) -> None:
        with self.assertRaises(ValueError):
            make_recall_tool(
                State("t"), max_chars_per_message=4000, max_total_chars=1000
            )


class CompactControlTest(unittest.TestCase):
    def test_compact_request_is_applied_at_the_next_turn_start(self) -> None:
        control = make_compact_control(keep_recent=2)

        def worker(visible: list[Message]) -> Message:
            if any(message.kind == "summary" for message in visible):
                return assistant_message(
                    "done", sender="worker", target="user", kind="final"
                )
            return assistant_message(
                [
                    TextBlock("context is long, folding"),
                    ToolCallBlock(
                        "c1",
                        "compact",
                        {"summary": "Investigated A and B; B is the answer."},
                    ),
                ],
                sender="worker",
                target="user",
                kind="step",
            )

        state = State("long task")
        state.send("task", "user", "worker", "long task")  # 0 (preserved kind)
        state.send("message", "user", "worker", "older note one")  # 1
        state.send("message", "user", "worker", "older note two")  # 2

        agent = Agent(
            "worker",
            worker,
            tools=(control.tool,),
            context_policy=ContextPolicy(strategy=control.strategy),
        )
        events = list(run(agent, state, max_turns=4))

        compressions = [
            event for event in events if isinstance(event, ContextCompressionEvent)
        ]
        # Exactly one fold: the request fires once at the next turn start and
        # is consumed; later turns stay idle.
        self.assertEqual(len(compressions), 1)
        compression = compressions[0]
        # The two older notes folded; the task (preserved kind) and the
        # keep_recent=2 tail (the compact exchange itself) stayed verbatim.
        self.assertEqual(compression.compressed_message_indices, [1, 2])
        replacement = state.messages[compression.summary_message_index]
        self.assertEqual(replacement.kind, "summary")
        replacement_text = text_of(replacement.content)
        self.assertIn("B is the answer", replacement_text)
        self.assertIn("transcript messages 1-2", replacement_text)
        # The run still finishes normally after the fold.
        self.assertEqual(state.messages[-1].kind, "final")

    def test_strategy_is_idle_without_a_pending_request(self) -> None:
        control = make_compact_control()
        active = [
            (0, State("t").send("task", "user", "worker", "t")),
        ]
        self.assertIsNone(control.strategy(active, "worker"))

    def test_request_with_nothing_to_fold_is_dropped(self) -> None:
        control = make_compact_control(keep_recent=2)
        result = control.tool.execute("call", {"summary": "keep this"}, _no_abort, None)
        self.assertFalse(result.is_error)

        state = State("t")
        task_message = state.send("task", "user", "worker", "t")
        recent = state.send("message", "user", "worker", "recent")
        active = [(0, task_message), (1, recent)]
        # Only one droppable message and keep_recent=2 -> nothing to fold;
        # the request is consumed, not deferred.
        self.assertIsNone(control.strategy(active, "worker"))
        self.assertIsNone(control.strategy(active, "worker"))

    def test_keep_recent_override_per_call(self) -> None:
        control = make_compact_control(keep_recent=2)
        control.tool.execute(
            "call", {"summary": "fold everything", "keep_recent": 0}, _no_abort, None
        )
        state = State("t")
        items = [
            (0, state.send("task", "user", "worker", "t")),
            (1, state.send("message", "user", "worker", "a")),
            (2, state.send("message", "user", "worker", "b")),
        ]
        decision = control.strategy(items, "worker")
        assert decision is not None
        self.assertEqual(decision.compress_indices, (1, 2))

    def test_tool_rejects_empty_summary_and_bad_keep_recent(self) -> None:
        control = make_compact_control()
        result = control.tool.execute("call", {}, _no_abort, None)
        self.assertTrue(result.is_error)
        self.assertIsNone(control.strategy([], "worker"))

        result = control.tool.execute(
            "call", {"summary": "s", "keep_recent": -1}, _no_abort, None
        )
        self.assertTrue(result.is_error)
        result = control.tool.execute(
            "call", {"summary": "s", "keep_recent": 1.5}, _no_abort, None
        )
        self.assertTrue(result.is_error)


if __name__ == "__main__":
    unittest.main()
