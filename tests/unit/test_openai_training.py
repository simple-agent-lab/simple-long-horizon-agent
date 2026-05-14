"""Unit tests for the OpenAI fine-tuning JSONL export."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from simple_agent_lab import (
    AgentTool,
    State,
    TextBlock,
    ToolCallBlock,
    assistant_message,
    text_result,
    tool_results_message,
    user_message,
)
from simple_agent_lab.messages import ImageBlock, ToolResultBlock
from simple_agent_lab.trace import append_openai_training_record, openai_training_record


_BASH_TOOL = AgentTool(
    name="bash",
    description="Run a bash command.",
    parameters={"type": "object", "properties": {"command": {"type": "string"}}},
    execute=None,
)


def _state_with_bash_roundtrip() -> State:
    state = State("task")
    state.send("task", "user", "agent", "hello world")
    state.record(
        assistant_message(
            [
                TextBlock("calling bash"),
                ToolCallBlock("c1", "bash", {"command": "echo hi"}),
            ],
            sender="agent",
            target="user",
            kind="thought",
        )
    )
    state.record(
        tool_results_message(
            [
                ToolResultBlock(
                    tool_call_id="c1",
                    tool_name="bash",
                    content=(TextBlock("hi"),),
                ),
            ],
            target="agent",
        )
    )
    state.record(
        assistant_message("Done!", sender="agent", target="user", kind="final")
    )
    return state


class OpenAITrainingRecordTest(unittest.TestCase):
    def test_full_tool_roundtrip_serializes_in_openai_chat_shape(self) -> None:
        state = _state_with_bash_roundtrip()
        record = openai_training_record(
            state,
            tools=[_BASH_TOOL],
            system_prompt="be brief",
        )

        # The wire shape matches an OpenAI fine-tuning JSONL line.
        self.assertIn("messages", record)
        self.assertIn("tools", record)

        roles = [m["role"] for m in record["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "assistant"])

        # System prompt lands first.
        self.assertEqual(record["messages"][0]["content"], "be brief")

        # Assistant turn carries the tool_call in the OpenAI shape.
        assistant_turn = record["messages"][2]
        self.assertIsNotNone(assistant_turn["tool_calls"])
        self.assertEqual(assistant_turn["tool_calls"][0]["id"], "c1")
        self.assertEqual(assistant_turn["tool_calls"][0]["function"]["name"], "bash")
        # arguments are JSON-stringified per OpenAI's wire schema.
        self.assertEqual(
            json.loads(assistant_turn["tool_calls"][0]["function"]["arguments"]),
            {"command": "echo hi"},
        )

        # Tool-result becomes a role=tool entry keyed by tool_call_id.
        tool_turn = record["messages"][3]
        self.assertEqual(tool_turn["tool_call_id"], "c1")
        self.assertEqual(tool_turn["content"], "hi")

        # Tools section uses the chat function-tool schema.
        self.assertEqual(record["tools"][0]["type"], "function")
        self.assertEqual(record["tools"][0]["function"]["name"], "bash")

    def test_no_tools_emits_no_tools_field(self) -> None:
        state = State("plain")
        state.send("task", "user", "agent", "hi")
        state.record(
            assistant_message("hello", sender="agent", target="user", kind="final")
        )
        record = openai_training_record(state)
        self.assertNotIn("tools", record)

    def test_append_writes_jsonl_and_appends_on_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            append_openai_training_record(
                _state_with_bash_roundtrip(), path, tools=[_BASH_TOOL]
            )
            append_openai_training_record(
                _state_with_bash_roundtrip(), path, tools=[_BASH_TOOL]
            )
            lines = path.read_text().splitlines()
        self.assertEqual(len(lines), 2)
        for line in lines:
            parsed = json.loads(line)
            self.assertEqual(parsed["tools"][0]["function"]["name"], "bash")

    def test_image_in_tool_result_splits_into_adjacent_user_message(self) -> None:
        # Matches the openai-chat adapter's wire fan-out: the tool-result
        # entry stays text-only and the image rides in an adjacent user
        # message. The training data should mirror the wire shape.
        state = State("vision")
        state.send("task", "user", "agent", "screenshot")
        state.record(
            assistant_message(
                [ToolCallBlock("c1", "shot", {})],
                sender="agent",
                target="user",
                kind="thought",
            )
        )
        state.record(
            tool_results_message(
                [
                    ToolResultBlock(
                        tool_call_id="c1",
                        tool_name="shot",
                        content=(
                            TextBlock("captured"),
                            ImageBlock(data="aGk=", mime_type="image/png"),
                        ),
                    )
                ],
                target="agent",
            )
        )
        record = openai_training_record(state)

        roles = [m["role"] for m in record["messages"]]
        # user task, assistant tool_call, role=tool (text only), user (image)
        self.assertEqual(roles, ["user", "assistant", "tool", "user"])
        # tool entry stays a plain string.
        self.assertEqual(record["messages"][2]["content"], "captured")
        # The image-bearing user message carries an image_url part.
        user_with_image = record["messages"][3]
        parts = user_with_image["content"]
        self.assertIn("image_url", [p["type"] for p in parts])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
