from __future__ import annotations

import os
import unittest
from pathlib import Path

from simple_agent_lab import message_text, tool_results_of
from simple_agent_lab.agents.bash import run_bash_agent_demo
from simple_agent_lab.llm import Provider

ROOT = Path(__file__).resolve().parents[1]
LIVE_ENV = "SIMPLE_AGENT_LAB_LIVE_E2E"
OPENAI_AUTH_ENV = "OPENAI_AUTH_TOKEN"


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


def live_e2e_enabled() -> bool:
    return os.environ.get(key=LIVE_ENV) == "1"


def build_live_openai_provider() -> Provider:
    model = (os.environ.get("OPENAI_MODEL") or "").strip()
    auth_token = (os.environ.get(OPENAI_AUTH_ENV) or "").strip()
    base_url = (os.environ.get("OPENAI_BASE_URL") or "").strip() or None

    missing = [
        name
        for name, value in (
            ("OPENAI_MODEL", model),
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


@unittest.skipUnless(
    live_e2e_enabled(),
    f"set {LIVE_ENV}=1 plus OPENAI_MODEL/{OPENAI_AUTH_ENV} to run live e2e",
)
class LiveBashAgentE2ETest(unittest.TestCase):
    def test_live_model_calls_bash_and_finalizes(self) -> None:
        marker = "SIMPLE_AGENT_LAB_LIVE_E2E_OK"
        runtime = run_bash_agent_demo(
            command=f"printf '{marker}\\n'",
            cwd=ROOT,
            provider=build_live_openai_provider(),
        )

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
