from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.agents.bash import make_bash_agent
from simple_agent_lab.llm import Provider
from simple_agent_lab.trajectory import run_trace_from_state, trace_record


ROOT = Path(__file__).resolve().parents[2]
OPENAI_MODEL_ENV = "OPENAI_MODEL"
OPENAI_AUTH_ENV = "OPENAI_AUTH_TOKEN"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
E2E_TRACE_PATH_ENV = "E2E_TRACE_PATH"


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


def build_live_openai_responses_provider() -> Provider:
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
        id="openai-responses-live-e2e",
        api="openai-responses",
        model=model,
        base_url=base_url,
        api_key_env=OPENAI_AUTH_ENV,
        default_temperature=None,
    )


def write_trace_if_requested(
    *,
    provider: Provider,
    state,
    filename: str,
    file_exists: bool,
    file_text: str | None,
) -> None:
    raw_path = (os.environ.get(E2E_TRACE_PATH_ENV) or "").strip()
    if not raw_path:
        return

    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    trace = run_trace_from_state(
        state=state,
        trace_id="live.openai_responses.tool_trace",
        producer="tests:e2e.openai_responses",
        meta={
            "provider": {
                "id": provider.id,
                "api": provider.api,
                "model": provider.model,
                "base_url_set": bool(provider.base_url),
                "api_key_env": provider.api_key_env,
            },
            "file_check": {
                "filename": filename,
                "exists": file_exists,
                "text": file_text,
            },
        },
    )
    path.write_text(
        json.dumps(trace_record(trace), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class LiveOpenAIResponsesE2ETest(unittest.TestCase):
    def test_live_responses_multi_turn_bash_tool_calls(self) -> None:
        """The Responses adapter must support tool call -> result -> tool call."""

        marker = "simple-agent-lab-responses-tool-ok"
        filename = "responses-tool.txt"

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            provider = build_live_openai_responses_provider()
            agent = make_bash_agent(
                provider=provider,
                system_prompt=(
                    "You are a connectivity probe for tool-use. Use exactly one "
                    "bash tool call per assistant turn. For this task, first call "
                    "bash to create the requested file. After you receive that "
                    "tool result, call bash a second time to read the file back. "
                    "Only after the second tool result, return a short final answer."
                ),
                cwd=workdir,
            )
            state, events = agent.run(
                f"Create a file named {filename} whose contents are exactly "
                f"'{marker}'. Then, in a separate later bash tool call after "
                f"you see the first tool result, read {filename} back and "
                "confirm the content in your final answer.",
                max_turns=6,
            )
            for _ in events:
                pass

            target = workdir / filename
            file_exists = target.exists()
            file_text = (
                target.read_text(encoding="utf-8").strip() if file_exists else None
            )
            write_trace_if_requested(
                provider=provider,
                state=state,
                filename=filename,
                file_exists=file_exists,
                file_text=file_text,
            )
            self.assertTrue(
                file_exists,
                f"agent never created {filename} (workdir contents: "
                f"{sorted(p.name for p in workdir.iterdir())})",
            )
            self.assertEqual(file_text, marker)

        tool_start_events = [
            event for event in state.events if event.kind == "tool_execution_start"
        ]
        self.assertGreaterEqual(
            len(tool_start_events),
            2,
            "agent did not execute bash at least twice",
        )

        tool_call_turns = [
            event
            for event in state.events
            if event.kind == "model_response" and event.tool_call_count > 0
        ]
        self.assertGreaterEqual(
            len(tool_call_turns),
            2,
            "agent did not perform tool calls across multiple model turns",
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
