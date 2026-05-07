from __future__ import annotations

import unittest
from typing import Any

import simple_agent_lab
from simple_agent_lab import (
    Agent,
    AgentRuntime,
    Message,
    State,
    ToolCallBlock,
    assistant_message,
    last_message,
    run_to_completion,
    sequence,
)
from simple_agent_lab.tools import AbortFlag, AgentTool, ToolResult, ToolUpdateFn, text_result


class BalancedCoreTest(unittest.TestCase):
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

        self.assertEqual(last_message(state, kind="final").content, "done")
        self.assertIn("model_request", [event.kind for event in state.events])
        self.assertIn("model_response", [event.kind for event in state.events])
        self.assertEqual(state.events[-1].kind, "agent_end")

    def test_dispatches_agent_tool_result_back_to_agent(self) -> None:
        def coordinator(agent: Agent, visible: list[Message], state: State) -> Message:
            if any(message.role == "tool_result" for message in visible):
                result = last_message(visible, kind="tool_result")
                return assistant_message(
                    f"final: {result.content}",
                    sender=agent.name,
                    target="user",
                    kind="final",
                )
            return assistant_message(
                "asking tool",
                sender=agent.name,
                target="user",
                kind="thought",
                tool_calls=(ToolCallBlock("call_1", "echo", {"text": "child ok"}),),
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

        self.assertEqual(last_message(state, kind="tool_result").content, "child ok")
        self.assertEqual(last_message(state, kind="final").content, "final: child ok")
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


if __name__ == "__main__":
    unittest.main()
