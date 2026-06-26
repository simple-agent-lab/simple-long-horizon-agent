from __future__ import annotations

import unittest
from typing import cast

from simple_agent_lab import (
    Agent,
    AgentTool,
    Message,
    ToolResult,
    assistant_message,
    tool_result_text,
)
from simple_agent_lab.tools.spawn_agents import (
    SPAWN_AGENTS_TOOL_NAME,
    spawn_agents_tool,
)


def _echo_agent(name: str) -> Agent:
    """A sub-agent that immediately finals with '<name>: <task text>'."""

    def generate(visible: list[Message]) -> Message:
        task = next(
            (m for m in visible if m.kind == "task"),
            None,
        )
        body = task.content[0].text if task and task.content else ""
        return assistant_message(
            f"{name}: {body}",
            sender=name,
            target="user",
            kind="final",
        )

    return Agent(name=name, generate=generate, role=f"role of {name}")


class SpawnAgentsToolTest(unittest.TestCase):
    def test_schema_advertises_a_batch_of_specs(self) -> None:
        tool = spawn_agents_tool([_echo_agent("explorer"), _echo_agent("coder")])
        self.assertEqual(tool.name, SPAWN_AGENTS_TOOL_NAME)
        self.assertEqual(tool.parameters["required"], ["agents"])
        items = tool.parameters["properties"]["agents"]["items"]
        self.assertEqual(items["required"], ["subagent_type", "prompt"])
        self.assertEqual(
            items["properties"]["subagent_type"]["enum"], ["coder", "explorer"]
        )

    def test_runs_multiple_agents_and_labels_each_result(self) -> None:
        tool = spawn_agents_tool([_echo_agent("explorer"), _echo_agent("coder")])
        result = _execute(
            tool,
            {
                "agents": [
                    {"subagent_type": "explorer", "prompt": "look around"},
                    {"subagent_type": "coder", "prompt": "write code"},
                ]
            },
        )

        self.assertFalse(result.is_error, tool_result_text(result))
        text = tool_result_text(result)
        self.assertIn("[agent 0 | explorer]", text)
        self.assertIn("explorer: look around", text)
        self.assertIn("[agent 1 | coder]", text)
        self.assertIn("coder: write code", text)
        self.assertEqual(result.details["count"], 2)

    def test_same_agent_twice_runs_both(self) -> None:
        tool = spawn_agents_tool([_echo_agent("explorer")])
        result = _execute(
            tool,
            {
                "agents": [
                    {"subagent_type": "explorer", "prompt": "branch A"},
                    {"subagent_type": "explorer", "prompt": "branch B"},
                ]
            },
        )
        text = tool_result_text(result)
        self.assertIn("explorer: branch A", text)
        self.assertIn("explorer: branch B", text)

    def test_empty_batch_is_an_error(self) -> None:
        tool = spawn_agents_tool([_echo_agent("explorer")])
        result = _execute(tool, {"agents": []})
        self.assertTrue(result.is_error)
        self.assertIn("non-empty list", tool_result_text(result))

    def test_unknown_subagent_is_an_error(self) -> None:
        tool = spawn_agents_tool([_echo_agent("explorer")])
        result = _execute(tool, {"agents": [{"subagent_type": "ghost", "prompt": "x"}]})
        self.assertTrue(result.is_error)
        self.assertIn("unknown", tool_result_text(result))

    def test_missing_prompt_is_an_error(self) -> None:
        tool = spawn_agents_tool([_echo_agent("explorer")])
        result = _execute(
            tool, {"agents": [{"subagent_type": "explorer", "prompt": "  "}]}
        )
        self.assertTrue(result.is_error)
        self.assertIn("prompt is required", tool_result_text(result))

    def test_exceeding_max_agents_is_an_error(self) -> None:
        tool = spawn_agents_tool([_echo_agent("explorer")], max_agents=1)
        result = _execute(
            tool,
            {
                "agents": [
                    {"subagent_type": "explorer", "prompt": "a"},
                    {"subagent_type": "explorer", "prompt": "b"},
                ]
            },
        )
        self.assertTrue(result.is_error)
        self.assertIn("Too many agents", tool_result_text(result))

    def test_requires_at_least_one_sub_agent(self) -> None:
        with self.assertRaises(ValueError):
            spawn_agents_tool([])


def _execute(tool: AgentTool, args: dict[str, object]) -> ToolResult:
    execute = cast(object, tool.execute)
    if not callable(execute):
        raise AssertionError("spawn_agents tool has no execute function")
    return execute("call_1", args, lambda: False, None)


if __name__ == "__main__":
    unittest.main()
