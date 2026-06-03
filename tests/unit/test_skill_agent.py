from __future__ import annotations

import unittest
from pathlib import Path

from simple_agent_lab import message_text
from simple_agent_lab.agents.skill import (
    SKILL_AGENT_DEFAULT_NAME,
    make_skill_agent,
)
from simple_agent_lab.llm import Provider

ROOT = Path(__file__).resolve().parents[1]
FAKE_PROVIDER = Provider(id="fake", api="fake", model="fake-model")


class SkillAgentTest(unittest.TestCase):
    def test_carries_bash_and_read_tools(self) -> None:
        agent = make_skill_agent(provider=FAKE_PROVIDER, cwd=ROOT)
        tool_names = [tool.name for tool in agent.tools]
        self.assertIn("bash", tool_names)
        self.assertIn("read", tool_names)

    def test_default_name(self) -> None:
        agent = make_skill_agent(provider=FAKE_PROVIDER, cwd=ROOT)
        self.assertEqual(agent.name, SKILL_AGENT_DEFAULT_NAME)

    def test_run_runs_bash_then_finalizes(self) -> None:
        agent = make_skill_agent(provider=FAKE_PROVIDER, cwd=ROOT)
        state, events = agent.run(
            "Use bash to run command: `printf 'skill ok\\n'`",
            max_turns=3,
        )
        for _ in events:
            pass

        tool_result_msg = next(
            message
            for message in reversed(state.messages)
            if message.kind == "tool_result"
        )
        final = next(
            message for message in reversed(state.messages) if message.kind == "final"
        )
        self.assertIn("skill ok", message_text(tool_result_msg))
        self.assertEqual(final.sender, SKILL_AGENT_DEFAULT_NAME)

    def test_overrides_apply(self) -> None:
        agent = make_skill_agent(
            provider=FAKE_PROVIDER,
            cwd=ROOT,
            name="custom_skill",
            system_prompt="custom prompt",
        )
        self.assertEqual(agent.name, "custom_skill")


if __name__ == "__main__":
    unittest.main()
