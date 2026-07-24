from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.agents.starter import make_bash_agent
from simple_agent_lab.llm import Provider
from simple_agent_lab.llm.env import load_dotenv, provider_from_env
from simple_agent_lab.trace import event_stream, run_trace_from_state


ROOT = Path(__file__).resolve().parents[2]
E2E_TRACE_PATH_ENV = "E2E_TRACE_PATH"


load_dotenv(ROOT / ".env")


def build_live_openai_responses_provider() -> Provider:
    # Single source of truth: `simple_agent_lab.llm.env`. Force the Responses
    # adapter and drop temperature (that endpoint rejects it); SkipTest lets
    # credential-less CI skip rather than fail.
    return provider_from_env(
        api_kind="openai-responses",
        default_temperature=None,
        label="live e2e",
        missing_exc=unittest.SkipTest,
        reexport_auth=True,
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
    header, lines, _pool = event_stream(trace)
    path.write_text(
        "".join(json.dumps(rec, ensure_ascii=False) + "\n" for rec in (header, *lines)),
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
