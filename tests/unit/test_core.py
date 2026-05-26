from __future__ import annotations

import unittest
from typing import Any

import simple_agent_lab
from simple_agent_lab import (
    Agent,
    AgentEndEvent,
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
    message_text,
    run,
    tool_result_message,
    tool_results_of,
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
            "ContentBlock",
            "ContextCompressionEvent",
            "ContextPolicy",
            "ContextStats",
            "ContextView",
            "Event",
            "EventKind",
            "GenerateFn",
            "ImageBlock",
            "Message",
            "MessageChannel",
            "MessageContent",
            "MessageEvent",
            "MessageKind",
            "ModelRequestEvent",
            "ModelResponseEvent",
            "Role",
            "RuntimeEvent",
            "State",
            "StateSnapshot",
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
            "TurnEndEvent",
            "TurnStartEvent",
            "UserMessage",
            "append_openai_training_record",
            "assistant_message",
            "build_context_view",
            "clip_message",
            "dispatch_tool_calls",
            "estimate_context_tokens",
            "estimate_message_chars",
            "estimate_message_tokens",
            "event_record",
            "is_tool_result_message",
            "make_llm_agent",
            "make_message",
            "message_text",
            "message_tool_calls",
            "model_turns_from_events",
            "openai_training_record",
            "print_trace",
            "run",
            "run_trace_from_state",
            "system_message",
            "task_tool",
            "text_of",
            "text_result",
            "thinking_blocks_of",
            "tool_calls_of",
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
        state.send("note", "planner", "planner", "planner note")
        state.send("note", "planner", "all", "broadcast")
        state.send("trace", "runtime", "writer", "internal trace")
        state.send("note", "planner", "writer", "debug note", channel="debug")

        view = build_context_view("writer", state.active_context_messages())

        self.assertEqual(
            [message_text(message) for message in view.messages],
            ["visible task", "planner note", "broadcast", "debug note"],
        )
        self.assertEqual(view.stats.total_messages, 5)
        self.assertEqual(view.stats.visible_messages, 4)

    def test_context_view_budget_keeps_pinned_task_and_recent_tail(self) -> None:
        state = State("budget test")
        state.send("task", "user", "writer", "pinned task")
        state.send("note", "user", "writer", "old " + ("x" * 120))
        state.send("note", "user", "writer", "recent answer")

        view = build_context_view(
            "writer",
            state.active_context_messages(),
            policy=ContextPolicy(max_chars=80, reserve_recent=1),
        )

        self.assertEqual(
            [message_text(message) for message in view.messages],
            ["pinned task", "recent answer"],
        )
        self.assertEqual(view.stats.dropped_messages, 1)
        self.assertIn("budget dropped 1 message(s)", view.notes)

    def test_context_view_keeps_tool_call_and_result_together(self) -> None:
        state = State("tool pair")
        state.send("task", "user", "coordinator", "use tool")
        state.send("note", "user", "coordinator", "old " + ("x" * 120))
        state.record(
            assistant_message(
                [
                    TextBlock("asking tool"),
                    ToolCallBlock("call_1", "echo", {"text": "ok"}),
                ],
                sender="coordinator",
                target="user",
                kind="thought",
            )
        )
        state.record(
            tool_result_message(
                "tool ok",
                tool_call_id="call_1",
                tool_name="echo",
                target="coordinator",
            )
        )

        view = build_context_view(
            "coordinator",
            state.active_context_messages(),
            policy=ContextPolicy(max_chars=120, reserve_recent=1),
        )

        self.assertEqual(
            [message.kind for message in view.messages],
            [
                "task",
                "thought",
                "tool_result",
            ],
        )
        last_results = tool_results_of(view.messages[-1].content)
        self.assertEqual(len(last_results), 1)
        self.assertEqual(last_results[0].tool_call_id, "call_1")

    def test_run_records_context_view_summary(self) -> None:
        def writer(visible: list[Message]) -> Message:
            del visible
            return assistant_message(
                "done",
                sender="writer",
                target="user",
                kind="final",
            )

        state = State("summarize context")
        state.send("task", "user", "writer", state.task)
        state.send("note", "user", "writer", "old " + ("x" * 120))
        for _ in run(
            Agent(
                "writer",
                writer,
                role="Write.",
                context_policy=ContextPolicy(max_chars=80, reserve_recent=0),
            ),
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
        self.assertEqual(context["dropped_messages"], 1)
        self.assertGreater(context["estimated_tokens"], 0)

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
        state.send("note", "user", "writer", "old " + ("x" * 120))
        state.send("note", "user", "writer", "recent note")
        compression_policy = ContextPolicy(
            compress_at_tokens=1,
            compress_keep_recent=1,
            compressor=Agent("compressor", compressor),
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


if __name__ == "__main__":
    unittest.main()
