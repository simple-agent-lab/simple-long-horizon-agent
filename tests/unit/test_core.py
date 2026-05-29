from __future__ import annotations

import unittest
from typing import Any

import simple_agent_lab
from simple_agent_lab import (
    Agent,
    AgentEndEvent,
    CompressionDecision,
    ContextCompressionEvent,
    ContextPolicy,
    EventKind,
    Message,
    ModelRequestEvent,
    State,
    TextBlock,
    ToolCallBlock,
    assistant_message,
    build_context_view,
    make_message,
    message_text,
    run,
    tool_result_message,
)
from simple_agent_lab.tools import (
    AbortFlag,
    AgentTool,
    ToolResult,
    ToolUpdateFn,
    task_tool,
    text_result,
    tool_result_text,
)


class CoreTest(unittest.TestCase):
    def test_records_request_and_response_events(self) -> None:
        def writer(visible: list[Message]) -> Message:
            del visible
            return assistant_message(
                "done",
                sender="writer",
                target="user",
                kind="final",
            )

        state = State("write one sentence")
        state.send("task", "user", "writer", state.task)
        for _ in run(
            Agent("writer", writer, role="Write one sentence."),
            state,
        ):
            pass

        self.assertIs(state.events[0].kind, EventKind.MESSAGE)
        self.assertIs(state.events[-1].kind, EventKind.AGENT_END)
        final = next(
            message for message in reversed(state.messages) if message.kind == "final"
        )
        self.assertEqual(message_text(final), "done")
        self.assertIn("model_request", [event.kind for event in state.events])
        self.assertIn("model_response", [event.kind for event in state.events])
        self.assertEqual(state.events[-1].kind, "agent_end")

    def test_dispatches_agent_tool_result_back_to_agent(self) -> None:
        def coordinator(visible: list[Message]) -> Message:
            if any(message.kind == "tool_result" for message in visible):
                result = next(
                    message
                    for message in reversed(visible)
                    if message.kind == "tool_result"
                )
                return assistant_message(
                    f"final: {message_text(result)}",
                    sender="coordinator",
                    target="user",
                    kind="final",
                )
            return assistant_message(
                [
                    TextBlock("asking tool"),
                    ToolCallBlock("call_1", "echo", {"text": "child ok"}),
                ],
                sender="coordinator",
                target="coordinator",
                kind="thought",
            )

        def echo_tool(
            call_id: str,
            args: dict[str, Any],
            abort: AbortFlag,
            on_update: ToolUpdateFn | None,
        ) -> ToolResult:
            del call_id, abort, on_update
            return text_result(str(args["text"]))

        state = State("delegate")
        state.send("task", "user", "coordinator", state.task)
        echo = AgentTool(
            name="echo",
            description="Echo text.",
            parameters={"type": "object"},
            execute=echo_tool,
            execution_mode="sequential",
        )
        for _ in run(
            Agent(
                "coordinator",
                coordinator,
                role="Coordinate tool use.",
                tools=(echo,),
            ),
            state,
            max_turns=2,
        ):
            pass

        self.assertEqual(
            message_text(
                next(
                    message
                    for message in reversed(state.messages)
                    if message.kind == "tool_result"
                )
            ),
            "child ok",
        )
        self.assertEqual(
            message_text(
                next(
                    message
                    for message in reversed(state.messages)
                    if message.kind == "final"
                )
            ),
            "final: child ok",
        )
        self.assertTrue(
            any(event.kind == "tool_execution_start" for event in state.events)
        )
        self.assertTrue(
            any(event.kind == "tool_execution_end" for event in state.events)
        )

    def test_agent_run_drives_loop_and_exposes_state(self) -> None:
        def writer(visible: list[Message]) -> Message:
            del visible
            return assistant_message(
                "done", sender="writer", target="user", kind="final"
            )

        agent = Agent("writer", writer)
        state, events = agent.run("write something")
        for _ in events:
            pass

        final = next(message for message in state.messages if message.kind == "final")
        self.assertEqual(message_text(final), "done")

    def test_max_turns_exhausted_reports_truncation_in_agent_end(self) -> None:
        # If the agent never emits `final`, the run was truncated by the
        # turn budget. The trace must say so — otherwise downstream analysis
        # cannot tell "agent decided to stop" from "ran out of turns".
        def chatty(visible: list[Message]) -> Message:
            del visible
            return assistant_message(
                "still thinking",
                sender="chatty",
                target="user",
                kind="thought",
            )

        state = State("ramble")
        state.send("task", "user", "chatty", state.task)
        for _ in run(Agent("chatty", chatty), state, max_turns=2):
            pass

        end = next(
            event
            for event in reversed(state.events)
            if isinstance(event, AgentEndEvent)
        )
        self.assertEqual(end.reason, "max_turns")

    def test_task_tool_dispatches_to_named_subagent(self) -> None:
        def make_prefix_generate(name: str, prefix: str):
            def generate(visible: list[Message]) -> Message:
                task_msg = next(
                    message for message in visible if message.kind == "task"
                )
                return assistant_message(
                    f"{prefix}:{message_text(task_msg)}",
                    sender=name,
                    target="user",
                    kind="final",
                )

            return generate

        echoer = Agent(
            "echoer", make_prefix_generate("echoer", "echo"), role="Echo back the task."
        )
        shouter = Agent(
            "shouter", make_prefix_generate("shouter", "SHOUT"), role="Shout the task."
        )
        tool = task_tool([echoer, shouter])

        self.assertEqual(tool.name, "task")
        self.assertEqual(tool.parameters["required"], ["subagent_type", "task"])
        self.assertIn("context", tool.parameters["properties"])
        self.assertEqual(
            tool.parameters["properties"]["subagent_type"]["enum"],
            ["echoer", "shouter"],
        )
        self.assertIn("echoer: Echo back the task.", tool.description)
        self.assertIn("shouter: Shout the task.", tool.description)

        result = tool.execute(
            "call_1",
            {"subagent_type": "shouter", "task": "hi"},
            lambda: False,
            None,
        )
        self.assertFalse(result.is_error)
        self.assertEqual(
            "\n".join(b.text for b in result.content if hasattr(b, "text")),
            "SHOUT:hi",
        )

    def test_task_tool_passes_default_and_call_context_to_subagent(self) -> None:
        def reader(visible: list[Message]) -> Message:
            context = [
                message_text(message)
                for message in visible
                if message.kind == "context"
            ]
            task_msg = next(message for message in visible if message.kind == "task")
            return assistant_message(
                "\n".join([*context, message_text(task_msg)]),
                sender="reader",
                target="user",
                kind="final",
            )

        tool = task_tool(
            [Agent("reader", reader, role="Read delegated task context.")],
            default_context="Repository: Simple Agent Lab.",
        )

        result = tool.execute(
            "call_1",
            {
                "subagent_type": "reader",
                "task": "Write the summary.",
                "context": "Caller: include the feedback signal.",
            },
            lambda: False,
            None,
        )

        self.assertFalse(result.is_error)
        self.assertEqual(
            tool_result_text(result),
            "\n".join(
                [
                    "Repository: Simple Agent Lab.",
                    "Caller: include the feedback signal.",
                    "Write the summary.",
                ]
            ),
        )

    def test_task_tool_reports_unknown_subagent_as_tool_error(self) -> None:
        def noop(visible: list[Message]) -> Message:
            del visible
            return assistant_message("ok", sender="only", target="user", kind="final")

        tool = task_tool([Agent("only", noop)])
        result = tool.execute(
            "call_1",
            {"subagent_type": "missing", "task": "hi"},
            lambda: False,
            None,
        )
        self.assertTrue(result.is_error)
        text = "\n".join(b.text for b in result.content if hasattr(b, "text"))
        self.assertIn("Unknown subagent_type", text)
        self.assertIn("'only'", text)

    def test_public_api_surface_is_a_known_set(self) -> None:
        # Pin the public `__all__` to a known set so adding or dropping a
        # symbol is an explicit, reviewable diff rather than silent drift.
        # Catches both regressions (re-exposing removed APIs like StepFn /
        # NextFn / until_final / make_llm_step) and accidental new exports.
        expected = {
            "AbortFlag",
            "Agent",
            "AgentEndEvent",
            "AgentName",
            "AgentStartEvent",
            "AgentTool",
            "AssistantMessage",
            "CompressionDecision",
            "CompressionStrategy",
            "ContentBlock",
            "ContextCompressionEvent",
            "ContextPolicy",
            "ContextView",
            "Event",
            "EventKind",
            "ImageBlock",
            "Message",
            "MessageChannel",
            "MessageContent",
            "MessageEvent",
            "MessageKind",
            "ModelRequestEvent",
            "ModelResponseEvent",
            "ModelTurn",
            "Role",
            "RunTrace",
            "Span",
            "State",
            "SummarizeStrategy",
            "SystemMessage",
            "TextBlock",
            "ThinkingBlock",
            "TokenUsage",
            "Tool",
            "ToolCallBlock",
            "ToolExecutionEndEvent",
            "ToolExecutionMode",
            "ToolExecutionStartEvent",
            "ToolExecutionUpdateEvent",
            "ToolResult",
            "ToolResultBlock",
            "ToolUpdateFn",
            "ToolCompactStrategy",
            "TurnEndEvent",
            "TurnStartEvent",
            "UserMessage",
            "append_openai_training_record",
            "assistant_message",
            "build_context_view",
            "estimate_context_tokens",
            "estimate_message_chars",
            "estimate_message_tokens",
            "event_record",
            "is_tool_result_message",
            "make_llm_agent",
            "make_message",
            "message_text",
            "model_turns_from_events",
            "openai_training_record",
            "print_trace",
            "run",
            "run_trace_from_state",
            "spans_from_events",
            "system_message",
            "task_tool",
            "text_of",
            "text_result",
            "tool_result_message",
            "tool_result_text",
            "tool_results_message",
            "tool_results_of",
            "user_message",
        }
        self.assertEqual(set(simple_agent_lab.__all__), expected)
        # `__all__` should also have no duplicates.
        self.assertEqual(
            len(simple_agent_lab.__all__), len(set(simple_agent_lab.__all__))
        )

    def test_context_view_uses_single_agent_transcript(self) -> None:
        state = State("route test")
        state.send("task", "user", "writer", "visible task")
        state.send("message", "planner", "planner", "planner note")
        state.send("message", "planner", "all", "broadcast")
        state.send("summary", "runtime", "writer", "old summary")
        state.send("message", "planner", "writer", "debug note", channel="debug")

        # Routing fields (sender/target/channel) never filter the view; only
        # `skip_kinds` does. Skipping "summary" hides exactly that one message.
        policy = ContextPolicy(skip_kinds=("summary",))
        view = build_context_view(
            "writer", state.active_context_messages(), policy=policy
        )

        self.assertEqual(
            [message_text(message) for message in view.messages],
            ["visible task", "planner note", "broadcast", "debug note"],
        )
        self.assertEqual(view.stats.total_messages, 5)
        self.assertEqual(view.stats.visible_messages, 4)

    def test_run_records_context_view_summary(self) -> None:
        def writer(visible: list[Message]) -> Message:
            del visible
            return assistant_message(
                "done",
                sender="writer",
                target="user",
                kind="final",
            )

        state = State("project context view")
        state.send("task", "user", "writer", state.task)
        state.send("message", "user", "writer", "first note")
        state.send("message", "user", "writer", "second note")
        for _ in run(
            Agent("writer", writer, role="Write."),
            state,
        ):
            pass

        request = next(
            event
            for event in reversed(state.events)
            if isinstance(event, ModelRequestEvent)
        )
        context = request.context_view
        self.assertEqual(context["agent"], "writer")
        self.assertGreater(context["estimated_tokens"], 0)
        self.assertEqual(context["visible_messages"], context["total_messages"])

    def test_run_compresses_context_before_model_request(self) -> None:
        captured: dict[str, Any] = {}

        def writer(visible: list[Message]) -> Message:
            captured["writer_visible_texts"] = [
                message_text(message) for message in visible
            ]
            return assistant_message(
                "done",
                sender="writer",
                target="user",
                kind="final",
            )

        def compressor(visible: list[Message]) -> Message:
            captured["compressor_prompt"] = message_text(visible[0])
            return assistant_message(
                "compressed old context",
                sender="compressor",
                target="runtime",
                kind="final",
            )

        state = State("compress context")
        state.send("task", "user", "writer", state.task)
        state.send("message", "user", "writer", "old " + ("x" * 120))
        state.send("message", "user", "writer", "recent note")
        compression_policy = ContextPolicy(
            strategies=(
                simple_agent_lab.SummarizeStrategy(
                    compressor=Agent("compressor", compressor),
                    threshold_tokens=1,
                    keep_recent=1,
                ),
            ),
        )

        for _ in run(
            Agent(
                "writer",
                writer,
                role="Write.",
                context_policy=compression_policy,
            ),
            state,
        ):
            pass

        summaries = [message for message in state.messages if message.kind == "summary"]
        self.assertEqual(len(summaries), 1)
        self.assertEqual(message_text(summaries[0]), "compressed old context")
        compression = next(
            event
            for event in state.events
            if isinstance(event, ContextCompressionEvent)
        )
        self.assertEqual(compression.compressed_message_indices, [1])
        self.assertEqual(
            state.snapshot.active_context_message_indices,
            compression.active_message_indices + [len(state.messages) - 1],
        )
        self.assertIn("old", captured["compressor_prompt"])
        self.assertEqual(
            captured["writer_visible_texts"],
            ["compress context", "compressed old context", "recent note"],
        )
        next_view = build_context_view("writer", state.active_context_messages())
        self.assertNotIn(
            "old " + ("x" * 116),
            [message_text(message) for message in next_view.messages],
        )
        rebuilt = state.rebuild_snapshot()
        self.assertEqual(
            rebuilt.active_context_message_indices,
            state.snapshot.active_context_message_indices,
        )

    def test_tool_compact_folds_old_tool_exchanges(self) -> None:
        # ToolCompactStrategy is the no-LLM first stage: when the context
        # exceeds `threshold_tokens`, every tool exchange except the most
        # recent `keep_recent_exchanges` is replaced by one short marker
        # listing the tool name and a result preview.
        def writer(visible: list[Message]) -> Message:
            del visible
            return assistant_message(
                "done", sender="writer", target="user", kind="final"
            )

        state = State("tool compact")
        state.send("task", "user", "writer", state.task)
        for index, payload in enumerate(
            ("alpha result", "beta result", "gamma result")
        ):
            state.record(
                assistant_message(
                    [
                        TextBlock(f"call-{index}"),
                        ToolCallBlock(f"c{index}", "echo", {"i": index}),
                    ],
                    sender="writer",
                    target="user",
                    kind="thought",
                )
            )
            state.record(
                tool_result_message(
                    payload,
                    tool_call_id=f"c{index}",
                    tool_name="echo",
                    target="writer",
                )
            )

        policy = ContextPolicy(
            strategies=(
                simple_agent_lab.ToolCompactStrategy(
                    threshold_tokens=1,
                    keep_recent_exchanges=1,
                ),
            ),
        )
        for _ in run(
            Agent("writer", writer, context_policy=policy),
            state,
            max_turns=1,
        ):
            pass

        compression = next(
            event
            for event in state.events
            if isinstance(event, ContextCompressionEvent)
        )
        # Compacted the first two exchanges (assistant + tool_result each = 4 msgs).
        self.assertEqual(len(compression.compressed_message_indices), 4)
        replacement = state.messages[compression.summary_message_index]
        self.assertEqual(replacement.kind, "summary")
        replacement_text = message_text(replacement)
        self.assertIn("Compacted 2 older tool exchange(s)", replacement_text)
        self.assertIn("alpha result", replacement_text)
        self.assertIn("beta result", replacement_text)
        # The most recent exchange stays verbatim — its messages survive
        # in active and the summary sits between the old block and the
        # kept tail.
        self.assertNotIn("gamma result", replacement_text)
        post_summary_indices = compression.active_message_indices[
            compression.active_message_indices.index(compression.summary_message_index)
            + 1 :
        ]
        recent_texts = [
            message_text(state.messages[index]) for index in post_summary_indices
        ]
        self.assertTrue(any("gamma result" in text for text in recent_texts))

    def test_default_tiered_policy_runs_compact_then_summarize(self) -> None:
        # The two default strategies compose into a tiered policy:
        # ToolCompactStrategy folds older tool exchanges cheaply, then
        # SummarizeStrategy runs an LLM pass if the context is still over
        # budget. Both compression events should fire in order.
        summarizer_calls: list[int] = []

        def writer(visible: list[Message]) -> Message:
            del visible
            return assistant_message(
                "done", sender="writer", target="user", kind="final"
            )

        def fake_summarizer(visible: list[Message]) -> Message:
            summarizer_calls.append(len(visible))
            return assistant_message(
                "llm summary text",
                sender="compressor",
                target="runtime",
                kind="final",
            )

        state = State("tiered policy")
        state.send("task", "user", "writer", state.task)
        for index, payload in enumerate(("alpha", "beta", "gamma")):
            state.record(
                assistant_message(
                    [
                        TextBlock(f"call-{index}"),
                        ToolCallBlock(f"c{index}", "echo", {"i": index}),
                    ],
                    sender="writer",
                    target="user",
                    kind="thought",
                )
            )
            state.record(
                tool_result_message(
                    payload,
                    tool_call_id=f"c{index}",
                    tool_name="echo",
                    target="writer",
                )
            )

        policy = ContextPolicy(
            strategies=(
                simple_agent_lab.ToolCompactStrategy(
                    threshold_tokens=1,
                    keep_recent_exchanges=1,
                ),
                simple_agent_lab.SummarizeStrategy(
                    compressor=Agent("compressor", fake_summarizer),
                    threshold_tokens=1,
                    keep_recent=0,
                ),
            ),
        )

        for _ in run(
            Agent("writer", writer, context_policy=policy),
            state,
            max_turns=1,
        ):
            pass

        compressions = [
            event
            for event in state.events
            if isinstance(event, ContextCompressionEvent)
        ]
        self.assertEqual(len(compressions), 2)
        self.assertEqual(len(summarizer_calls), 1)
        compact_replacement = state.messages[compressions[0].summary_message_index]
        self.assertIn("Compacted", message_text(compact_replacement))
        llm_replacement = state.messages[compressions[1].summary_message_index]
        self.assertEqual(message_text(llm_replacement), "llm summary text")

    def test_framework_auto_fixes_split_tool_pair(self) -> None:
        # If a strategy puts only one side of a tool_call/tool_result pair
        # in `compress_indices`, the framework un-compresses that side so
        # the model never sees an orphan. This frees strategy authors
        # from re-implementing tool-pair bookkeeping in every strategy.
        state = State("tool pair fixup")
        state.send("task", "user", "writer", state.task)
        call_msg = assistant_message(
            [
                TextBlock("calling"),
                ToolCallBlock("call_x", "echo", {"text": "hi"}),
            ],
            sender="writer",
            target="user",
            kind="thought",
        )
        state.record(call_msg)
        result_msg = tool_result_message(
            "ok",
            tool_call_id="call_x",
            tool_name="echo",
            target="writer",
        )
        state.record(result_msg)

        call_index = state.messages.index(call_msg)
        captured: dict[str, Any] = {}

        class CompressOnlyCall:
            def __call__(self, active, agent_name):
                captured["saw_active_count"] = len(active)
                return CompressionDecision(
                    compress_indices=(call_index,),
                    replacement=make_message(
                        "system",
                        "[call gone]",
                        sender="runtime",
                        target=agent_name,
                        kind="summary",
                    ),
                )

        def writer(visible: list[Message]) -> Message:
            return assistant_message(
                f"saw {len(visible)} msgs",
                sender="writer",
                target="user",
                kind="final",
            )

        policy = ContextPolicy(strategies=(CompressOnlyCall(),))
        for _ in run(Agent("writer", writer, context_policy=policy), state):
            pass

        # No compression event recorded: the only compress index was
        # orphan-fixed away, leaving compress_set empty.
        compressions = [
            event
            for event in state.events
            if isinstance(event, ContextCompressionEvent)
        ]
        self.assertEqual(compressions, [])


if __name__ == "__main__":
    unittest.main()
