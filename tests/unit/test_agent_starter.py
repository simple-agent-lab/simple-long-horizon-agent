"""Tests for the consolidated general agent starter.

Covers the `Toolset` abstraction (`toolsets.py`) and the `AgentSession`
runner plus presets (`starter.py`). MCP-backed cases reuse the SDK's
in-memory transport (same pattern as `test_mcp.py`) and are skipped when the
optional `mcp` extra is absent.
"""

from __future__ import annotations

import logging
import unittest
from contextlib import ExitStack, asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Sequence

from simple_agent_lab.llm import Provider
from simple_agent_lab.messages import TextBlock
from simple_agent_lab.skills import SkillRoot
from simple_agent_lab.tools import AgentTool, ToolResult, text_result
from simple_agent_lab.tools.bash import make_bash_tool
from simple_agent_lab.tools.read import make_read_tool

from simple_agent_lab.agents.starter import (
    BASH_AGENT_SYSTEM_PROMPT,
    BASH_TASK_AGENT_SYSTEM_PROMPT,
    EXPLORER_AGENT_DEFAULT_NAME,
    AgentSession,
    SkillConfig,
    bash_session,
    bash_task_session,
    make_bash_agent,
    make_bash_task_agent,
    mcp_session,
    skill_session,
)
from simple_agent_lab.agents.toolsets import MCPToolset, Toolset


ROOT = Path(__file__).resolve().parents[2]
FAKE_PROVIDER = Provider(id="fake", api="fake", model="fake-model")

try:
    from mcp import ClientSession
    from mcp.server.fastmcp import FastMCP
    from mcp.shared.memory import (
        create_connected_server_and_client_session as connect_session,
    )

    from simple_agent_lab.mcp import MCPConnection, MCPServerConfig

    HAS_MCP = True
except ImportError:  # pragma: no cover - exercised only without the extra
    HAS_MCP = False

_SKIP_REASON = "mcp extra not installed (install with: uv sync --extra mcp)"

# Quiet the in-memory MCP server's per-request INFO logs so test output stays
# readable (mirrors test_mcp.py).
logging.getLogger("mcp").setLevel(logging.WARNING)


def _static_tool(name: str) -> AgentTool:
    def execute(call_id, args, abort, on_update) -> ToolResult:  # noqa: ANN001
        del call_id, args, abort, on_update
        return text_result(f"{name} ran")

    return AgentTool(
        name=name,
        description=f"test tool {name}",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )


class FakeToolset:
    """A `Toolset` with no real resource — records open/close for assertions."""

    def __init__(self, tools: Sequence[AgentTool]) -> None:
        self._tools = tuple(tools)
        self.opened = False
        self.closed = False

    def __enter__(self) -> "FakeToolset":
        self.opened = True
        return self

    def __exit__(self, *exc: object) -> None:
        self.closed = True

    def tools(self) -> Sequence[AgentTool]:
        if not self.opened:
            raise RuntimeError("FakeToolset.tools() before open")
        return self._tools


class ToolsetProtocolTest(unittest.TestCase):
    def test_fake_toolset_satisfies_protocol(self) -> None:
        toolset = FakeToolset([_static_tool("a")])
        self.assertIsInstance(toolset, Toolset)

    def test_exit_stack_opens_and_closes_toolset(self) -> None:
        toolset = FakeToolset([_static_tool("a")])
        with ExitStack() as stack:
            opened = stack.enter_context(toolset)
            self.assertTrue(opened.opened)
            self.assertEqual([t.name for t in opened.tools()], ["a"])
        self.assertTrue(toolset.closed)


FIXTURE_SKILLS = ROOT / "tests" / "fixtures" / "skills"


class AgentSessionTest(unittest.TestCase):
    def test_agent_before_enter_raises(self) -> None:
        session = AgentSession(
            provider=FAKE_PROVIDER, name="t", static_tools=[_static_tool("a")]
        )
        with self.assertRaises(RuntimeError):
            _ = session.agent

    def test_builds_agent_with_static_tool(self) -> None:
        with AgentSession(
            provider=FAKE_PROVIDER, name="t", static_tools=[_static_tool("a")]
        ) as session:
            self.assertEqual([t.name for t in session.agent.tools], ["a"])

    def test_toolset_tools_merged_after_static_tools(self) -> None:
        toolset = FakeToolset([_static_tool("b")])
        with AgentSession(
            provider=FAKE_PROVIDER,
            name="t",
            static_tools=[_static_tool("a")],
            toolsets=[toolset],
        ) as session:
            self.assertEqual([t.name for t in session.agent.tools], ["a", "b"])
            self.assertTrue(toolset.opened)
        self.assertTrue(toolset.closed)

    def test_exit_closes_toolset_even_when_body_raises(self) -> None:
        toolset = FakeToolset([_static_tool("b")])
        with self.assertRaises(ValueError):
            with AgentSession(provider=FAKE_PROVIDER, name="t", toolsets=[toolset]):
                raise ValueError("boom")
        self.assertTrue(toolset.closed)

    def test_plain_session_records_no_skills_menu(self) -> None:
        with AgentSession(
            provider=FAKE_PROVIDER,
            name="t",
            static_tools=[make_bash_tool(cwd=str(FIXTURE_SKILLS))],
        ) as session:
            state, events = session.run(
                "Use bash to run command: `printf 'ok\\n'`", max_turns=3
            )
            for _ in events:
                pass
        self.assertFalse(any(m.sender == "skills" for m in state.messages))

    def test_skills_enabled_routes_through_run_with_skills(self) -> None:
        with AgentSession(
            provider=FAKE_PROVIDER,
            name="t",
            role="skilled",
            static_tools=[
                make_bash_tool(cwd=str(FIXTURE_SKILLS)),
                make_read_tool(cwd=str(FIXTURE_SKILLS)),
            ],
            skills=SkillConfig(
                enabled=True,
                roots=[SkillRoot(str(FIXTURE_SKILLS), "repo")],
                cwd=str(FIXTURE_SKILLS),
            ),
        ) as session:
            state, events = session.run("do something", max_turns=3)
            for _ in events:
                pass
        menu = next((m for m in state.messages if m.sender == "skills"), None)
        self.assertIsNotNone(menu)
        self.assertIn("echo-fixture", menu.content[0].text)


class PresetTest(unittest.TestCase):
    def test_bash_session_exposes_only_bash(self) -> None:
        with bash_session(FAKE_PROVIDER, cwd=str(ROOT)) as session:
            self.assertEqual([t.name for t in session.agent.tools], ["bash"])
            self.assertEqual(session.agent.system_prompt, BASH_AGENT_SYSTEM_PROMPT)

    def test_bash_task_session_exposes_bash_and_task(self) -> None:
        with bash_task_session(FAKE_PROVIDER, cwd=str(ROOT)) as session:
            names = sorted(t.name for t in session.agent.tools)
            self.assertEqual(names, ["bash", "task"])
            task = next(t for t in session.agent.tools if t.name == "task")
            self.assertEqual(
                list(task.parameters["properties"]["subagent_type"]["enum"]),
                [EXPLORER_AGENT_DEFAULT_NAME],
            )
            self.assertTrue(
                BASH_TASK_AGENT_SYSTEM_PROMPT.startswith(BASH_AGENT_SYSTEM_PROMPT)
            )

    def test_skill_session_carries_bash_read_and_skill_config(self) -> None:
        session = skill_session(FAKE_PROVIDER, cwd=str(FIXTURE_SKILLS))
        with session as entered:
            self.assertEqual(
                sorted(t.name for t in entered.agent.tools), ["bash", "read"]
            )

    def test_make_bash_agent_returns_plain_agent(self) -> None:
        agent = make_bash_agent(provider=FAKE_PROVIDER, cwd=str(ROOT))
        self.assertEqual([t.name for t in agent.tools], ["bash"])
        self.assertEqual(agent.name, "bash_agent")

    def test_make_bash_task_agent_returns_plain_agent(self) -> None:
        agent = make_bash_task_agent(FAKE_PROVIDER, cwd=str(ROOT))
        self.assertEqual(sorted(t.name for t in agent.tools), ["bash", "task"])
        self.assertEqual(agent.name, "bash_task_agent")


@unittest.skipUnless(HAS_MCP, _SKIP_REASON)
class MCPToolsetTest(unittest.TestCase):
    @staticmethod
    def _demo_server() -> "FastMCP":
        server = FastMCP("demo")

        @server.tool(description="Echo text back as plain text.")
        def echo(text: str) -> str:
            return f"echo: {text}"

        return server

    def _in_memory_connect(self):
        server = self._demo_server()

        def connect(config: "MCPServerConfig") -> "MCPConnection":
            @asynccontextmanager
            async def factory() -> AsyncIterator["ClientSession"]:
                async with connect_session(server._mcp_server) as session:
                    yield session

            return MCPConnection(factory, name=config.name).open()

        return connect

    def test_tools_before_enter_raises(self) -> None:
        toolset = MCPToolset(MCPServerConfig.stdio("demo", "noop"))
        with self.assertRaises(RuntimeError):
            toolset.tools()

    def test_open_yields_prefixed_tools_and_closes(self) -> None:
        config = MCPServerConfig.stdio("demo", "noop")
        toolset = MCPToolset(config, connect=self._in_memory_connect())
        with toolset as opened:
            names = {t.name for t in opened.tools()}
            self.assertIn("demo_echo", names)
            echo = next(t for t in opened.tools() if t.name == "demo_echo")
            result = echo.execute("c1", {"text": "hi"}, lambda: False, None)
            self.assertEqual(result.content, (TextBlock("echo: hi"),))
        # After exit the connection is dropped.
        with self.assertRaises(RuntimeError):
            toolset.tools()

    def test_mcp_session_exposes_server_tools(self) -> None:
        config = MCPServerConfig.stdio("demo", "noop")
        with mcp_session(
            FAKE_PROVIDER,
            servers=[config],
            connect=self._in_memory_connect(),
        ) as session:
            self.assertIn("demo_echo", {t.name for t in session.agent.tools})


if __name__ == "__main__":
    unittest.main()
