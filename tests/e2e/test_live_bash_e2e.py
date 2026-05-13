from __future__ import annotations

import os
import unittest
from pathlib import Path

from simple_agent_lab import message_text, tool_results_of
from simple_agent_lab.agents.bash import (
    BASH_AGENT_DEFAULT_NAME,
    bash_agent_until_final,
    make_bash_agent_runtime,
)
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
    def test_live_model_calls_bash_and_finalizes(self) -> None:
        marker = "SIMPLE_AGENT_LAB_LIVE_E2E_OK"
        runtime = make_bash_agent_runtime(
            provider=build_live_openai_provider(),
            cwd=ROOT,
        )
        for _ in runtime.prompt(
            f"Use bash to run command: `printf '{marker}\\n'`",
            target=BASH_AGENT_DEFAULT_NAME,
            next_agent=bash_agent_until_final,
        ):
            pass

        tool_result_msg = next(
            message
            for message in reversed(runtime.state.messages)
            if message.kind == "tool_result"
        )
        final = next(
            message
            for message in reversed(runtime.state.messages)
            if message.kind == "final"
        )

        blocks = tool_results_of(tool_result_msg.content)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].tool_name, "bash")
        self.assertFalse(blocks[0].is_error)
        self.assertIn(marker, message_text(tool_result_msg))
        self.assertEqual(final.sender, "bash_agent")
        self.assertTrue(message_text(final).strip())
        self.assertTrue(
            any(event.kind == "tool_execution_start" for event in runtime.state.events)
        )


if __name__ == "__main__":
    unittest.main()
