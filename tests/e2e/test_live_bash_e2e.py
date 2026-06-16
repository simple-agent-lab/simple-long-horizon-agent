from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.agents.starter import make_bash_agent
from simple_agent_lab.llm import Provider
from simple_agent_lab.llm.env import load_dotenv, provider_from_env


ROOT = Path(__file__).resolve().parents[2]


load_dotenv(ROOT / ".env")


def build_live_openai_provider() -> Provider:
    # Single source of truth: `simple_agent_lab.llm.env`. `missing_exc=SkipTest`
    # makes credential-less CI skip rather than fail; `reexport_auth` cleans the
    # token for the adapter, which reads os.environ directly.
    return provider_from_env(
        label="live e2e",
        missing_exc=unittest.SkipTest,
        reexport_auth=True,
    )


class LiveBashAgentE2ETest(unittest.TestCase):
    def test_live_model_writes_file_with_bash(self) -> None:
        """The model must actually invoke bash to create a file on disk.

        We give a natural-language task (no exact command), point cwd at a
        throwaway temp dir, and assert the file exists on disk with the
        expected content. That's the real proof bash was used end-to-end.
        """
        marker = "SIMPLE_AGENT_LAB_E2E_42"
        filename = "hello.txt"

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            agent = make_bash_agent(
                provider=build_live_openai_provider(),
                cwd=workdir,
            )
            state, events = agent.run(
                f"Use bash to create a file named {filename} in the current "
                f"working directory whose contents are exactly the text "
                f"'{marker}' (no trailing newline required). Then return a "
                f"short final answer confirming you did it.",
                max_turns=5,
            )
            for _ in events:
                pass

            target = workdir / filename
            self.assertTrue(
                target.exists(),
                f"agent never created {filename} (workdir contents: "
                f"{sorted(p.name for p in workdir.iterdir())})",
            )
            contents = target.read_text(encoding="utf-8").strip()
            self.assertEqual(contents, marker)

        self.assertTrue(
            any(event.kind == "tool_execution_start" for event in state.events),
            "agent did not invoke the bash tool at all",
        )
        final = next(
            (
                message
                for message in reversed(state.messages)
                if message.kind == "final"
            ),
            None,
        )
        if final is None:
            self.fail("agent did not produce a final message")
        self.assertEqual(final.sender, "bash_agent")


if __name__ == "__main__":
    unittest.main()
