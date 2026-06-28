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
from simple_agent_lab.compression import (
    continuation_preamble,
    format_index_ranges,
)
from simple_agent_lab.messages import (
    ToolResultBlock,
    make_message,
    tool_results_message,
)
from simple_agent_lab.tools import make_recall_tool, tool_result_text


def _no_abort() -> bool:
    return False


def _compact_request_item(
    index: int,
    control: object,
    *,
    summary: str,
    keep_recent: int | None = None,
    call_id: str = "c1",
    target: str = "worker",
) -> tuple[int, Message]:
    """Run the `compact` tool and wrap its result the way the loop records it.

    The request rides in the tool_result bundle's `details` sidecar, exactly as
    `dispatch_tool_calls` stores it, so the strategy reads it back off `active`.
    """
    args: dict[str, object] = {"summary": summary}
    if keep_recent is not None:
        args["keep_recent"] = keep_recent
    result = control.tool.execute(call_id, args, _no_abort, None)  # type: ignore[attr-defined]
    block = ToolResultBlock(
        tool_call_id=call_id,
        tool_name="compact",
        content=tuple(result.content),
        is_error=result.is_error,
    )
    message = tool_results_message(
        [block], target=target, sidecar={"details": {call_id: result.details}}
    )
    return index, message


class IndexCitationTest(unittest.TestCase):
    def test_format_index_ranges_collapses_runs(self) -> None:
        self.assertEqual(format_index_ranges((2, 3, 4, 7)), "2-4, 7")
        self.assertEqual(format_index_ranges((5,)), "5")
        self.assertEqual(format_index_ranges((9, 1, 2)), "1-2, 9")
        self.assertEqual(format_index_ranges(()), "")

    def test_continuation_preamble_frames_a_session_handoff(self) -> None:
        preamble = continuation_preamble()
        self.assertIn("continues a previous", preamble)
        self.assertIn("working memory", preamble)


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
        tool = make_recall_tool(state, max_chars_per_message=200, max_total_chars=400)
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
        tool = make_recall_tool(state, max_chars_per_message=400, max_total_chars=400)
        result = tool.execute("call", {"indices": [1]}, _no_abort, None)
        self.assertFalse(result.is_error)
        self.assertEqual(result.details["indices"], [1])
        self.assertNotIn("recall truncated", tool_result_text(result))

    def test_rejects_total_budget_below_per_message_cap(self) -> None:
        with self.assertRaises(ValueError):
            make_recall_tool(
                State("t"), max_chars_per_message=4000, max_total_chars=1000
            )

    def test_deduplicates_indices_preserving_order(self) -> None:
        state = State("t")
        state.send("task", "user", "worker", "t")  # 0
        state.send("message", "user", "worker", "alpha")  # 1
        state.send("message", "user", "worker", "beta")  # 2
        tool = make_recall_tool(state)
        result = tool.execute("call", {"indices": [2, 1, 2, 1]}, _no_abort, None)
        self.assertFalse(result.is_error)
        # First-occurrence order kept; each message rendered once.
        self.assertEqual(result.details["indices"], [2, 1])
        text = tool_result_text(result)
        self.assertEqual(text.count("alpha"), 1)
        self.assertEqual(text.count("beta"), 1)
        self.assertLess(text.index("beta"), text.index("alpha"))

    def test_max_indices_cap_counts_raw_input_before_dedup(self) -> None:
        state = State("t")
        state.send("task", "user", "worker", "t")
        state.send("message", "user", "worker", "x")  # index 1
        tool = make_recall_tool(state, max_indices=3)
        # Four entries, all duplicates of a valid index: still rejected on the
        # raw length so a pathological array can't slip through dedup.
        result = tool.execute("call", {"indices": [1, 1, 1, 1]}, _no_abort, None)
        self.assertTrue(result.is_error)
        self.assertIn("at most 3 indices", tool_result_text(result))


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
        # Framed as a continuation, then the agent's summary.
        self.assertTrue(replacement_text.startswith("[This session continues"))
        self.assertIn("B is the answer", replacement_text)
        # The run still finishes normally after the fold.
        self.assertEqual(state.messages[-1].kind, "final")

    def test_strategy_is_idle_without_a_request(self) -> None:
        control = make_compact_control()
        active = [
            (0, State("t").send("task", "user", "worker", "t")),
        ]
        # No tool_result carries a compact_request, so the strategy stays idle.
        self.assertIsNone(control.strategy(active, "worker"))

    def test_request_with_nothing_to_fold_is_dropped(self) -> None:
        control = make_compact_control(keep_recent=2)
        task = State("t").send("task", "user", "worker", "t")
        request = _compact_request_item(1, control, summary="keep this")
        # Task is a preserved kind, so the only droppable item is the request
        # itself and keep_recent=2 protects it -> nothing to fold, every pass.
        active = [(0, task), request]
        self.assertIsNone(control.strategy(active, "worker"))
        self.assertIsNone(control.strategy(active, "worker"))

    def test_keep_recent_override_per_call(self) -> None:
        control = make_compact_control(keep_recent=2)
        state = State("t")
        request = _compact_request_item(
            3, control, summary="fold everything", keep_recent=0
        )
        items = [
            (0, state.send("task", "user", "worker", "t")),
            (1, state.send("message", "user", "worker", "a")),
            (2, state.send("message", "user", "worker", "b")),
            request,
        ]
        decision = control.strategy(items, "worker")
        assert decision is not None
        # keep_recent=0 folds every droppable message, the request bundle included.
        self.assertEqual(decision.compress_indices, (1, 2, 3))

    def test_multiple_requests_in_one_turn_apply_the_first(self) -> None:
        # Two compact calls share one tool_result bundle (parallel tool pool);
        # the first call recorded is the one applied, deterministically.
        control = make_compact_control(keep_recent=0)
        r1 = control.tool.execute("c1", {"summary": "first call"}, _no_abort, None)
        r2 = control.tool.execute("c2", {"summary": "second call"}, _no_abort, None)
        bundle = tool_results_message(
            [
                ToolResultBlock("c1", "compact", tuple(r1.content), r1.is_error),
                ToolResultBlock("c2", "compact", tuple(r2.content), r2.is_error),
            ],
            target="worker",
            sidecar={"details": {"c1": r1.details, "c2": r2.details}},
        )
        state = State("t")
        items = [
            (0, state.send("task", "user", "worker", "t")),
            (1, state.send("message", "user", "worker", "a")),
            (2, bundle),
        ]
        decision = control.strategy(items, "worker")
        assert decision is not None
        self.assertIn("first call", text_of(decision.replacement.content))
        self.assertEqual(decision.replacement.role, "user")

    def test_request_is_not_reapplied_once_a_newer_summary_exists(self) -> None:
        # Exactly-once via the high-water mark: once a fold has spliced a summary
        # newer than the request, the request is spent and never fires again,
        # even though it stays in the append-only transcript.
        control = make_compact_control(keep_recent=1)
        state = State("t")
        request = _compact_request_item(3, control, summary="folded")
        active = [
            (0, state.send("task", "user", "worker", "t")),
            (1, state.send("message", "user", "worker", "old one")),
            (2, state.send("message", "user", "worker", "old two")),
            request,
        ]
        self.assertIsNotNone(control.strategy(active, "worker"))

        # The applied fold splices a summary at a higher transcript index (4)
        # than the request (3); the request must now be treated as consumed.
        summary = make_message(
            "user",
            "running summary",
            sender="runtime",
            target="worker",
            kind="summary",
        )
        applied = [active[0], (4, summary), request]
        self.assertIsNone(control.strategy(applied, "worker"))

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
