"""Tests for the consolidated general agent starter.

Covers the `Toolset` abstraction (`toolsets.py`) and the `AgentSession`
runner plus presets (`starter.py`). MCP-backed cases reuse the SDK's
in-memory transport (same pattern as `test_mcp.py`) and are skipped when the
optional `mcp` extra is absent.
"""

from __future__ import annotations

import logging
import unittest
from contextlib import ExitStack, asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterator, Sequence

from simple_agent_lab.agents.starter import (
    BASH_AGENT_SYSTEM_PROMPT,
    BASH_TASK_ADDENDUM,
    GENERAL_PURPOSE_AGENT_DEFAULT_NAME,
    MCP_ADDENDUM,
    SKILLS_ADDENDUM,
    AgentSession,
    SkillConfig,
    agent_session,
    compose_agent_system_prompt,
    make_agent,
    make_bash_agent,
    make_bash_task_agent,
    make_skill_agent,
    mcp_session,
)
from simple_agent_lab.agents.toolsets import MCPToolset, Toolset
from simple_agent_lab.hooks import HookPoint
from simple_agent_lab.llm import Provider
from simple_agent_lab.messages import TextBlock
from simple_agent_lab.skills import SkillRoot
from simple_agent_lab.tools import AgentTool, ToolResult, text_result
from simple_agent_lab.tools.bash import make_bash_tool
from simple_agent_lab.tools.read import make_read_tool


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_SKILLS = ROOT / "tests" / "fixtures" / "skills"
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


def _tool_names(agent: Any, *, sorted_: bool = False) -> list[str]:
    names = [tool.name for tool in agent.tools]
    return sorted(names) if sorted_ else names


def _skill_config() -> SkillConfig:
    return SkillConfig(
        enabled=True,
        roots=[SkillRoot(str(FIXTURE_SKILLS), "repo")],
        cwd=str(FIXTURE_SKILLS),
    )


def _run(subject: Any, task: str = "do something") -> Any:
    state, events = subject.run(task, max_turns=3)
    tuple(events)
    return state


def _skill_menu(state: Any) -> Any:
    return next(
        (message for message in state.messages if message.sender == "skills"), None
    )


@contextmanager
def _built_agent(factory: Callable[..., Any], **kwargs: Any) -> Iterator[Any]:
    if factory is make_agent:
        yield factory(FAKE_PROVIDER, **kwargs)
    else:
        with factory(FAKE_PROVIDER, **kwargs) as session:
            yield session.agent


def _demo_server() -> "FastMCP":
    server = FastMCP("demo")

    @server.tool(description="Echo text back as plain text.")
    def echo(text: str) -> str:
        return f"echo: {text}"

    return server


def _in_memory_connect() -> Callable[["MCPServerConfig"], "MCPConnection"]:
    server = _demo_server()

    def connect(config: "MCPServerConfig") -> "MCPConnection":
        @asynccontextmanager
        async def factory() -> AsyncIterator["ClientSession"]:
            async with connect_session(server._mcp_server) as session:
                yield session

        return MCPConnection(factory, name=config.name).open()

    return connect


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


class ComposePromptTest(unittest.TestCase):
    def test_composes_enabled_fragments_in_order(self) -> None:
        cases = [
            (
                "bash only",
                {},
                [BASH_AGENT_SYSTEM_PROMPT],
            ),
            (
                "all",
                {"general_purpose": True, "skills": True, "mcp": True},
                [
                    BASH_AGENT_SYSTEM_PROMPT,
                    BASH_TASK_ADDENDUM,
                    SKILLS_ADDENDUM,
                    MCP_ADDENDUM,
                ],
            ),
            (
                "skills",
                {"skills": True},
                [BASH_AGENT_SYSTEM_PROMPT, SKILLS_ADDENDUM],
            ),
        ]
        for label, enabled, expected in cases:
            with self.subTest(label):
                flags = {
                    "bash": True,
                    "general_purpose": False,
                    "skills": False,
                    "mcp": False,
                }
                flags.update(enabled)
                self.assertEqual(
                    compose_agent_system_prompt(**flags),
                    "\n\n".join(expected),
                )


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
            self.assertEqual(_tool_names(session.agent), ["a"])

    def test_toolset_tools_merged_after_static_tools(self) -> None:
        toolset = FakeToolset([_static_tool("b")])
        with AgentSession(
            provider=FAKE_PROVIDER,
            name="t",
            static_tools=[_static_tool("a")],
            toolsets=[toolset],
        ) as session:
            self.assertEqual(_tool_names(session.agent), ["a", "b"])
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
            state = _run(session, "Use bash to run command: `printf 'ok\\n'`")
        self.assertFalse(any(m.sender == "skills" for m in state.messages))

    def test_skills_enabled_installs_state_initializer(self) -> None:
        with AgentSession(
            provider=FAKE_PROVIDER,
            name="t",
            role="skilled",
            static_tools=[
                make_bash_tool(cwd=str(FIXTURE_SKILLS)),
                make_read_tool(cwd=str(FIXTURE_SKILLS)),
            ],
            skills=_skill_config(),
        ) as session:
            state = _run(session)
        menu = _skill_menu(state)
        self.assertIsNotNone(menu)
        self.assertIn("echo-fixture", menu.content[0].text)


class BackCompatFactoryTest(unittest.TestCase):
    def test_wrappers_return_expected_plain_agents(self) -> None:
        cases = [
            (make_bash_agent, ["bash"], "bash_agent"),
            (make_bash_task_agent, ["bash", "task"], "bash_task_agent"),
        ]
        for factory, tools, name in cases:
            with self.subTest(factory.__name__):
                agent = factory(FAKE_PROVIDER, cwd=str(ROOT))
                self.assertEqual(_tool_names(agent, sorted_=True), tools)
                self.assertEqual(agent.name, name)


@unittest.skipUnless(HAS_MCP, _SKIP_REASON)
class MCPToolsetTest(unittest.TestCase):
    def test_tools_before_enter_raises(self) -> None:
        toolset = MCPToolset(MCPServerConfig.stdio("demo", "noop"))
        with self.assertRaises(RuntimeError):
            toolset.tools()

    def test_open_yields_prefixed_tools_and_closes(self) -> None:
        config = MCPServerConfig.stdio("demo", "noop")
        toolset = MCPToolset(config, connect=_in_memory_connect())
        with toolset as opened:
            names = {t.name for t in opened.tools()}
            self.assertIn("demo_echo", names)
            echo = next(t for t in opened.tools() if t.name == "demo_echo")
            result = echo.execute("c1", {"text": "hi"}, lambda: False, None)
            self.assertEqual(result.content, (TextBlock("echo: hi"),))
        # After exit the connection is dropped.
        with self.assertRaises(RuntimeError):
            toolset.tools()


class AgentFactoryTest(unittest.TestCase):
    factories = (make_agent, agent_session)

    def test_bash_only_default(self) -> None:
        for factory in self.factories:
            with (
                self.subTest(factory.__name__),
                _built_agent(factory, cwd=str(ROOT)) as agent,
            ):
                self.assertEqual(_tool_names(agent), ["bash"])
                self.assertEqual(agent.system_prompt, BASH_AGENT_SYSTEM_PROMPT)

    def test_read_and_general_purpose_compose_tools_and_prompt(self) -> None:
        for factory in self.factories:
            with (
                self.subTest(factory.__name__),
                _built_agent(
                    factory, cwd=str(ROOT), read=True, general_purpose=True
                ) as agent,
            ):
                self.assertEqual(
                    _tool_names(agent, sorted_=True), ["bash", "read", "task"]
                )
                task = next(tool for tool in agent.tools if tool.name == "task")
                self.assertEqual(
                    list(task.parameters["properties"]["subagent_type"]["enum"]),
                    [GENERAL_PURPOSE_AGENT_DEFAULT_NAME],
                )
                self.assertIn(BASH_TASK_ADDENDUM, agent.system_prompt)

    def test_extra_tools_and_prompt_override(self) -> None:
        cases = [
            ({"tools": [_static_tool("x")]}, ["bash", "x"], BASH_AGENT_SYSTEM_PROMPT),
            (
                {"general_purpose": True, "system_prompt": "custom"},
                ["bash", "task"],
                "custom",
            ),
        ]
        for factory in self.factories:
            for kwargs, tools, prompt in cases:
                with (
                    self.subTest(factory=factory.__name__, prompt=prompt),
                    _built_agent(factory, cwd=str(ROOT), **kwargs) as agent,
                ):
                    self.assertEqual(_tool_names(agent), tools)
                    self.assertEqual(agent.system_prompt, prompt)

    def test_session_skills_configs_imply_read_and_initialize_state(self) -> None:
        for skills in (True, _skill_config()):
            with (
                self.subTest(type=type(skills).__name__),
                _built_agent(
                    agent_session, cwd=str(FIXTURE_SKILLS), skills=skills
                ) as agent,
            ):
                self.assertIn(SKILLS_ADDENDUM, agent.system_prompt)
                self.assertEqual(_tool_names(agent, sorted_=True), ["bash", "read"])
                if isinstance(skills, SkillConfig):
                    state = _run(agent)
                    menu = _skill_menu(state)
                    self.assertIsNotNone(menu)
                    self.assertIn("echo-fixture", menu.content[0].text)

    def test_resource_free_options(self) -> None:
        agent = make_agent(FAKE_PROVIDER, cwd=str(ROOT), bash=False, read=True)
        self.assertEqual(_tool_names(agent), ["read"])

        hooks = {HookPoint.SESSION_END: [lambda ctx: None]}
        for factory in self.factories:
            with (
                self.subTest(factory.__name__),
                _built_agent(
                    factory, cwd=str(ROOT), general_purpose=True, hooks=hooks
                ) as built,
            ):
                self.assertIs(built.hooks, hooks)


class MakeSkillAgentTest(unittest.TestCase):
    def test_builds_bare_skills_aware_agent(self) -> None:
        # Returns a plain Agent (no session, no runner wrapper) — skills ride
        # on the core `init_state` hook, not a separate run path.
        agent = make_skill_agent(FAKE_PROVIDER, cwd=str(FIXTURE_SKILLS))
        self.assertIn(SKILLS_ADDENDUM, agent.system_prompt)
        self.assertEqual(_tool_names(agent, sorted_=True), ["bash", "read"])
        self.assertIsNotNone(agent.init_state)

    def test_bare_run_advertises_skills_menu(self) -> None:
        agent = make_skill_agent(
            FAKE_PROVIDER,
            cwd=str(FIXTURE_SKILLS),
            roots=[SkillRoot(str(FIXTURE_SKILLS), "repo")],
        )
        # The key property: a plain `agent.run` is skills-aware, no wrapper.
        state = _run(agent)
        menu = _skill_menu(state)
        self.assertIsNotNone(menu)
        self.assertIn("echo-fixture", menu.content[0].text)

    def test_optional_general_purpose_and_hooks(self) -> None:
        agent = make_skill_agent(
            FAKE_PROVIDER,
            cwd=str(FIXTURE_SKILLS),
            general_purpose=True,
        )
        self.assertIn("task", _tool_names(agent))

        hooks = {HookPoint.SESSION_END: [lambda ctx: None]}
        agent = make_skill_agent(FAKE_PROVIDER, cwd=str(FIXTURE_SKILLS), hooks=hooks)
        self.assertIs(agent.hooks, hooks)


@unittest.skipUnless(HAS_MCP, _SKIP_REASON)
class MCPAgentSessionTest(unittest.TestCase):
    def test_mcp_servers_expose_tools_and_compose_prompt(self) -> None:
        config = MCPServerConfig.stdio("demo", "noop")
        with agent_session(
            FAKE_PROVIDER,
            cwd=str(ROOT),
            mcp_servers=[config],
            connect=_in_memory_connect(),
        ) as session:
            names = {t.name for t in session.agent.tools}
            self.assertIn("bash", names)
            self.assertIn("demo_echo", names)
            self.assertIn(MCP_ADDENDUM, session.agent.system_prompt)

    def test_hooks_attach_to_session_agent(self) -> None:
        hooks = {HookPoint.SESSION_END: [lambda ctx: None]}
        config = MCPServerConfig.stdio("demo", "noop")
        with agent_session(
            FAKE_PROVIDER,
            cwd=str(ROOT),
            mcp_servers=[config],
            connect=_in_memory_connect(),
            hooks=hooks,
        ) as session:
            self.assertIs(session.agent.hooks, hooks)

    def test_skills_and_mcp_together(self) -> None:
        # The combination the old presets could not express.
        config = MCPServerConfig.stdio("demo", "noop")
        with agent_session(
            FAKE_PROVIDER,
            cwd=str(FIXTURE_SKILLS),
            read=True,
            skills=_skill_config(),
            mcp_servers=[config],
            connect=_in_memory_connect(),
        ) as session:
            names = {t.name for t in session.agent.tools}
            self.assertIn("demo_echo", names)
            self.assertIn("read", names)
            state = _run(session)
        self.assertTrue(any(m.sender == "skills" for m in state.messages))

    def test_mcp_session_wrapper_exposes_tools(self) -> None:
        config = MCPServerConfig.stdio("demo", "noop")
        with mcp_session(
            FAKE_PROVIDER,
            [config],
            cwd=str(ROOT),
            connect=_in_memory_connect(),
        ) as session:
            self.assertIn("demo_echo", {t.name for t in session.agent.tools})
