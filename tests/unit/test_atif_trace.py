from __future__ import annotations

import unittest

from simple_agent_lab.messages import (
    AssistantMessage,
    TextBlock,
    ThinkingBlock,
    TokenUsage,
    ToolCallBlock,
    ToolResultBlock,
    UserMessage,
)
from simple_agent_lab.protocols import MessageEvent, ModelResponseEvent
from simple_agent_lab.trace.atif import atif_trajectory_from_run


class AtifTraceTest(unittest.TestCase):
    def test_atif_export_maps_messages_tools_and_token_totals(self) -> None:
        assistant = AssistantMessage(
            content=(
                ThinkingBlock(text="Need inspect."),
                TextBlock(text="I will inspect the file."),
                ToolCallBlock(
                    id="call-1",
                    name="bash",
                    arguments={"command": "ls"},
                ),
            ),
            sender="agent",
            target="agent",
            kind="thought",
            usage=TokenUsage(
                input_tokens=10,
                output_tokens=5,
                cache_read_tokens=3,
                cache_write_tokens=2,
            ),
            model="model-x",
        )
        tool_result = UserMessage(
            content=(
                ToolResultBlock(
                    tool_call_id="call-1",
                    tool_name="bash",
                    content=(TextBlock(text="README.md"),),
                ),
            ),
            sender="tool",
            target="agent",
            kind="tool_result",
        )
        events = [
            MessageEvent(message=assistant),
            ModelResponseEvent(
                agent="agent",
                output_kind="thought",
                target="agent",
                tool_call_count=1,
                usage=assistant.usage,
                model="model-x",
            ),
            MessageEvent(message=tool_result),
        ]

        trajectory = atif_trajectory_from_run(
            trace_id="trace-1",
            task="Do the task",
            events=events,
            messages=[assistant, tool_result],
            agent_name="simple-agent-lab",
            agent_version="0.1.0",
            model_name="model-x",
            producer="unit-test",
            extra={"agent_flavor": "bash_task"},
        )

        self.assertEqual(trajectory["schema_version"], "ATIF-v1.7")
        self.assertEqual(trajectory["session_id"], "trace-1")
        self.assertEqual(trajectory["steps"][0]["source"], "user")
        self.assertEqual(trajectory["steps"][0]["message"], "Do the task")

        agent_step = trajectory["steps"][1]
        self.assertEqual(agent_step["source"], "agent")
        self.assertEqual(agent_step["model_name"], "model-x")
        self.assertEqual(agent_step["message"], "I will inspect the file.")
        self.assertEqual(agent_step["reasoning_content"], "Need inspect.")
        self.assertEqual(
            agent_step["tool_calls"],
            [
                {
                    "tool_call_id": "call-1",
                    "function_name": "bash",
                    "arguments": {"command": "ls"},
                }
            ],
        )
        self.assertEqual(
            agent_step["observation"],
            {
                "results": [
                    {
                        "source_call_id": "call-1",
                        "content": "README.md",
                        "extra": {"is_error": False, "tool_name": "bash"},
                    }
                ]
            },
        )
        self.assertEqual(
            agent_step["metrics"],
            {
                "prompt_tokens": 15,
                "completion_tokens": 5,
                "cached_tokens": 3,
                "cost_usd": 0.0,
                "extra": {"cache_write_tokens": 2},
            },
        )

        self.assertEqual(
            trajectory["final_metrics"],
            {
                "total_prompt_tokens": 15,
                "total_completion_tokens": 5,
                "total_cached_tokens": 3,
                "total_cost_usd": 0.0,
                "total_steps": 2,
                "extra": {
                    "cache_write_tokens": 2,
                    "llm_call_count": 1,
                    "unpriced_models": ["model-x"],
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
