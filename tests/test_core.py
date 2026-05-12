from __future__ import annotations

import unittest
from typing import Any

import simple_agent_lab
from simple_agent_lab import (
    Agent,
    AgentRuntime,
    ContextPolicy,
    Message,
    State,
    TextBlock,
    ToolCallBlock,
    assistant_message,
    build_agent_context_view,
    last_event,
    last_message,
    message_text,
    run_to_completion,
    sequence,
)
from simple_agent_lab.tools import AbortFlag, AgentTool, ToolResult, ToolUpdateFn, text_result


class CoreTest(unittest.TestCase):
    def test_records_request_and_response_events(self) -> None:
        def writer(agent: Agent, visible: list[Message], state: State) -> Message:
            return assistant_message(
                "done",
                sender=agent.name,
                target="user",
                kind="final",
            )

        state = State("write one sentence")
        state.send("task", "user", "writer", state.task)
        run_to_completion(
            [Agent("writer", "Write one sentence.", writer)],
            state,
            sequence("writer"),
        )

        self.assertEqual(message_text(last_message(state, kind="final")), "done")
        self.assertIn("model_request", [event.kind for event in state.events])
        self.assertIn("model_response", [event.kind for event in state.events])
        self.assertEqual(state.events[-1].kind, "agent_end")

    def test_dispatches_agent_tool_result_back_to_agent(self) -> None:
        def coordinator(agent: Agent, visible: list[Message], state: State) -> Message:
            if any(message.role == "tool_result" for message in visible):
                result = last_message(visible, kind="tool_result")
                return assistant_message(
                    f"final: {message_text(result)}",
                    sender=agent.name,
                    target="user",
                    kind="final",
                )
            return assistant_message(
                [
                    TextBlock("asking tool"),
                    ToolCallBlock("call_1", "echo", {"text": "child ok"}),
                ],
                sender=agent.name,
                target="user",
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

        def until_final(state: State) -> str | None:
            if any(message.kind == "final" for message in state.messages):
                return None
            turns = sum(event.kind == "turn_end" for event in state.events)
            return "coordinator" if turns < 2 else None

        state = State("delegate")
        state.send("task", "user", "coordinator", state.task)
        run_to_completion(
            [Agent("coordinator", "Coordinate tool use.", coordinator)],
            state,
            until_final,
            tools=[
                AgentTool(
                    name="echo",
                    description="Echo text.",
                    parameters={"type": "object"},
                    execute=echo_tool,
                    execution_mode="sequential",
                )
            ],
        )

        self.assertEqual(
            message_text(last_message(state, kind="tool_result")), "child ok"
        )
        self.assertEqual(
            message_text(last_message(state, kind="final")), "final: child ok"
        )
        self.assertTrue(any(event.kind == "tool_execution_start" for event in state.events))
        self.assertTrue(any(event.kind == "tool_execution_end" for event in state.events))

    def test_runtime_has_no_input_queue_api(self) -> None:
        def noop(agent: Agent, visible: list[Message], state: State) -> Message:
            return assistant_message("", sender=agent.name, target="user", kind="final")

        runtime = AgentRuntime([Agent("agent", "Finish.", noop)])

        self.assertFalse(hasattr(runtime, "steer"))
        self.assertFalse(hasattr(runtime, "follow_up"))
        self.assertFalse(hasattr(runtime, "continue_"))
        self.assertTrue(hasattr(runtime, "resume"))

    def test_public_api_drops_unused_step_hooks(self) -> None:
        self.assertFalse(hasattr(simple_agent_lab, "default_convert_to_llm"))
        self.assertFalse(hasattr(simple_agent_lab, "BeforeStepResult"))
        self.assertFalse(hasattr(simple_agent_lab, "AfterStepResult"))

    def test_context_view_filters_routes_and_channels(self) -> None:
        def noop(agent: Agent, visible: list[Message], state: State) -> Message:
            del visible, state
            return assistant_message("", sender=agent.name, target="user", kind="final")

        state = State("route test")
        state.send("task", "user", "writer", "visible task")
        state.send("note", "planner", "planner", "planner private")
        state.send("note", "planner", "all", "broadcast")
        state.send("trace", "runtime", "writer", "hidden trace")
        state.send("note", "planner", "writer", "debug note", channel="debug")

        view = build_agent_context_view(
            Agent("writer", "Write.", noop),
            state,
            policy=ContextPolicy(channels=("main",)),
        )

        self.assertEqual(
            [message_text(message) for message in view.messages],
            ["visible task", "broadcast"],
        )
        self.assertEqual(view.stats.total_messages, 5)
        self.assertEqual(view.stats.visible_messages, 2)

    def test_context_view_budget_keeps_pinned_task_and_recent_tail(self) -> None:
        def noop(agent: Agent, visible: list[Message], state: State) -> Message:
            del visible, state
            return assistant_message("", sender=agent.name, target="user", kind="final")

        state = State("budget test")
        state.send("task", "user", "writer", "pinned task")
        state.send("note", "user", "writer", "old " + ("x" * 120))
        state.send("note", "user", "writer", "recent answer")

        view = build_agent_context_view(
            Agent("writer", "Write.", noop),
            state,
            policy=ContextPolicy(max_chars=80, reserve_recent=1),
        )

        self.assertEqual(
            [message_text(message) for message in view.messages],
            ["pinned task", "recent answer"],
        )
        self.assertEqual(view.stats.dropped_messages, 1)
        self.assertIn("budget dropped 1 message(s)", view.notes)

    def test_context_view_keeps_tool_call_and_result_together(self) -> None:
        def noop(agent: Agent, visible: list[Message], state: State) -> Message:
            del visible, state
            return assistant_message("", sender=agent.name, target="user", kind="final")

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
        state.send(
            "tool_result",
            "echo",
            "coordinator",
            "tool ok",
            role="tool_result",
            tool_call_id="call_1",
            tool_name="echo",
        )

        view = build_agent_context_view(
            Agent("coordinator", "Coordinate.", noop),
            state,
            policy=ContextPolicy(max_chars=120, reserve_recent=1),
        )

        self.assertEqual([message.kind for message in view.messages], [
            "task",
            "thought",
            "tool_result",
        ])
        self.assertEqual(view.messages[-1].tool_call_id, "call_1")

    def test_run_records_context_view_summary(self) -> None:
        def writer(agent: Agent, visible: list[Message], state: State) -> Message:
            del visible, state
            return assistant_message(
                "done",
                sender=agent.name,
                target="user",
                kind="final",
            )

        state = State("summarize context")
        state.send("task", "user", "writer", state.task)
        state.send("note", "user", "writer", "old " + ("x" * 120))
        run_to_completion(
            [Agent("writer", "Write.", writer)],
            state,
            sequence("writer"),
            context_policy=ContextPolicy(max_chars=80, reserve_recent=0),
        )

        request = last_event(state, kind="model_request")
        context = request.data["context_view"]
        self.assertEqual(context["agent"], "writer")
        self.assertEqual(context["dropped_messages"], 1)
        self.assertGreater(context["estimated_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
