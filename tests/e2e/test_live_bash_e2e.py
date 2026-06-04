from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.agents.starter import make_bash_agent
from simple_agent_lab.llm import Provider


ROOT = Path(__file__).resolve().parents[2]
OPENAI_MODEL_ENV = "OPENAI_MODEL"
OPENAI_AUTH_ENV = "OPENAI_AUTH_TOKEN"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv(ROOT / ".env")


def build_live_openai_provider() -> Provider:
    model = (os.environ.get(OPENAI_MODEL_ENV) or "").strip()
    auth_token = (os.environ.get(OPENAI_AUTH_ENV) or "").strip()
    base_url = (os.environ.get(OPENAI_BASE_URL_ENV) or "").strip() or None

    missing = [
        name
        for name, value in (
            (OPENAI_MODEL_ENV, model),
            (OPENAI_AUTH_ENV, auth_token),
        )
        if not value
    ]
    if missing:
        raise unittest.SkipTest(
            "missing env var(s) for live e2e: " + ", ".join(missing)
        )

    os.environ[OPENAI_AUTH_ENV] = auth_token
    return Provider(
        id="openai-chat-live-e2e",
        api="openai-chat",
        model=model,
        base_url=base_url,
        api_key_env=OPENAI_AUTH_ENV,
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
