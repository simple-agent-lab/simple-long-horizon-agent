"""Focused tests for the bash + task delegation agent preset.

The fake LLM adapter is bash-aware but not task-aware (see
``simple_agent_lab.llm.adapters.fake``), so these tests drive the
parent's ``generate`` callable directly with a small stub. That keeps
the focus on the wiring this preset introduces:

* parent agent exposes both ``bash`` and ``task`` tools,
* parent's system prompt is the bash-agent prompt plus a small addendum,
* task dispatch picks ``subagent_type="explorer"`` and the explorer's
  bash final message rides back as the parent's tool result,
* both parent and explorer share the requested ``cwd``.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from simple_agent_lab import (
    Message,
    ToolCallBlock,
    assistant_message,
    message_text,
    tool_results_of,
)
from simple_agent_lab.agents.starter import (
    BASH_AGENT_SYSTEM_PROMPT,
    BASH_TASK_AGENT_SYSTEM_PROMPT,
    BASH_TASK_EXPLORER_ADDENDUM,
    EXPLORER_AGENT_DEFAULT_NAME,
    make_bash_task_agent,
)
from simple_agent_lab.llm import Provider


ROOT = Path(__file__).resolve().parents[2]
FAKE_PROVIDER = Provider(id="fake", api="fake", model="fake-model")


class BashTaskAgentTest(unittest.TestCase):
    def test_parent_exposes_bash_and_task_tools(self) -> None:
        agent = make_bash_task_agent(FAKE_PROVIDER, cwd=ROOT)

        tool_names = sorted(tool.name for tool in agent.tools)
        self.assertEqual(tool_names, ["bash", "task"])

    def test_task_tool_lists_only_the_explorer_subagent(self) -> None:
        agent = make_bash_task_agent(FAKE_PROVIDER, cwd=ROOT)

        task_tool_def = next(tool for tool in agent.tools if tool.name == "task")
        self.assertEqual(
            list(task_tool_def.parameters["properties"]["subagent_type"]["enum"]),
            [EXPLORER_AGENT_DEFAULT_NAME],
        )
        self.assertIn(EXPLORER_AGENT_DEFAULT_NAME, task_tool_def.description)

    def test_system_prompt_extends_bash_prompt_with_short_addendum(self) -> None:
        self.assertTrue(
            BASH_TASK_AGENT_SYSTEM_PROMPT.startswith(BASH_AGENT_SYSTEM_PROMPT),
            "Composed prompt should reuse the bash-agent prompt as its base.",
        )
        self.assertIn(BASH_TASK_EXPLORER_ADDENDUM, BASH_TASK_AGENT_SYSTEM_PROMPT)
        addendum_lines = BASH_TASK_EXPLORER_ADDENDUM.count(". ") + 1
        self.assertLessEqual(
            addendum_lines,
            5,
            f"Addendum should stay short; got ~{addendum_lines} sentences.",
        )

    def test_delegating_to_explorer_returns_sub_agent_final_as_tool_result(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "marker.txt").write_text("explorer-saw-this\n")

            agent = make_bash_task_agent(FAKE_PROVIDER, cwd=workspace)
            agent.generate = _make_parent_generate(
                delegated_task="Use bash to run command: `cat marker.txt`",
            )

            state, events = agent.run("delegate the read", max_turns=4)
            for _ in events:
                pass

        tool_result_msg = next(
            message
            for message in reversed(state.messages)
            if message.kind == "tool_result"
        )
        blocks = tool_results_of(tool_result_msg.content)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].tool_name, "task")
        self.assertIn("explorer-saw-this", message_text(tool_result_msg))

        final = next(
            message for message in reversed(state.messages) if message.kind == "final"
        )
        self.assertEqual(final.sender, "bash_task_agent")
        self.assertIn("done", message_text(final))


def _make_parent_generate(*, delegated_task: str):
    """Build a tiny deterministic parent generate function.

    Turn 1: emit a single ``task`` tool call asking the explorer to run
    a small bash command. Turn 2 (after the tool result comes back):
    emit a one-line final.
    """

    call_count = 0

    def generate(visible: list[Message]) -> Message:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return assistant_message(
                (
                    ToolCallBlock(
                        id="task_1",
                        name="task",
                        arguments={
                            "subagent_type": EXPLORER_AGENT_DEFAULT_NAME,
                            "task": delegated_task,
                        },
                    ),
                ),
                sender="bash_task_agent",
                target="user",
                kind="step",
            )
        return assistant_message(
            "delegation done",
            sender="bash_task_agent",
            target="user",
            kind="final",
        )

    return generate


if __name__ == "__main__":
    unittest.main()
