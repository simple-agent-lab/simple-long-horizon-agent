"""The general agent starter: one runner + presets for every agent kind.

This module replaces the per-kind subfolders. A single :class:`AgentSession`
(the runner) opens any :class:`~simple_agent_lab.agents.toolsets.Toolset`
resources, assembles the full tool list, builds an ``Agent`` via the existing
:func:`~simple_agent_lab.llm_agent.make_llm_agent`, and dispatches ``run`` to
either the plain loop or the skills loop. Four thin preset constructors
(:func:`bash_session`, :func:`bash_task_session`, :func:`skill_session`,
:func:`mcp_session`) configure that session for each kind.

The kinds differ only by their tool list and an optional skills flag — there is
no per-kind class. ``bash_task``'s explorer sub-agent is an ordinary ``task``
tool entry, and ``mcp`` servers are ``MCPToolset`` entries the session opens
and closes. Back-compat ``make_bash_agent`` / ``make_bash_task_agent`` wrappers
remain for callers that already drive a plain ``Agent`` themselves.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, Sequence

from simple_agent_lab.context_view import ContextPolicy
from simple_agent_lab.core import Agent
from simple_agent_lab.llm import Provider as LLMProvider
from simple_agent_lab.llm_agent import make_llm_agent
from simple_agent_lab.protocols import Event
from simple_agent_lab.skills import SkillMetadata, SkillRoot, run_with_skills
from simple_agent_lab.state import State
from simple_agent_lab.tools import AbortFlag, AgentTool
from simple_agent_lab.tools.bash import make_bash_tool
from simple_agent_lab.tools.read import make_read_tool
from simple_agent_lab.tools.task import task_tool

from .toolsets import MCPToolset, Toolset

if TYPE_CHECKING:
    from simple_agent_lab.mcp import MCPConnection, MCPServerConfig


@dataclass(frozen=True)
class SkillConfig:
    """How a session advertises and injects agent skills.

    Mirrors the keyword surface of
    :func:`~simple_agent_lab.skills.run_with_skills`: ``skills`` (pre-discovered
    metadata) takes precedence over ``roots`` (discovery roots); ``preload``
    names have their bodies injected up front; ``cwd`` scopes default
    discovery. When ``enabled`` is false the session runs the plain loop even
    though a config is present.
    """

    enabled: bool = True
    roots: Sequence[SkillRoot] | None = None
    skills: Sequence[SkillMetadata] | None = None
    preload: Sequence[str] = ()
    cwd: str = "."


class AgentSession:
    """A context-managed agent runner that owns its toolsets' lifetimes.

    Enter the session to open every toolset (via one ``ExitStack``), gather
    their tools alongside the static tools, and build the ``Agent``. Call
    :meth:`run` while inside the ``with`` block; exit to close the toolsets.
    """

    def __init__(
        self,
        *,
        provider: LLMProvider,
        name: str,
        role: str = "",
        system_prompt: str = "",
        target: str = "user",
        static_tools: Sequence[AgentTool] = (),
        toolsets: Sequence[Toolset] = (),
        skills: SkillConfig | None = None,
        context_policy: ContextPolicy | None = None,
        request_extra: Mapping[str, Any] | None = None,
        max_turns: int = 10,
    ) -> None:
        self._provider = provider
        self._name = name
        self._role = role
        self._system_prompt = system_prompt
        self._target = target
        self._static_tools = tuple(static_tools)
        self._toolsets = tuple(toolsets)
        self._skills = skills
        self._context_policy = context_policy
        self._request_extra = request_extra
        self._max_turns = max_turns

        self._stack: ExitStack | None = None
        self._agent: Agent | None = None

    @property
    def agent(self) -> Agent:
        """The built ``Agent`` — only available inside the ``with`` block."""

        if self._agent is None:
            raise RuntimeError(
                "AgentSession.agent is only available inside the context; "
                "enter the session first (use `with session as s: ...`)"
            )
        return self._agent

    def __enter__(self) -> "AgentSession":
        stack = ExitStack()
        tools: list[AgentTool] = list(self._static_tools)
        try:
            for toolset in self._toolsets:
                opened = stack.enter_context(toolset)
                tools.extend(opened.tools())
            self._agent = make_llm_agent(
                name=self._name,
                provider=self._provider,
                role=self._role,
                tools=tools,
                system_prompt=self._system_prompt,
                target=self._target,
                context_policy=self._context_policy,
                request_extra=self._request_extra,
            )
        except BaseException:
            # A toolset that opened before the failure must still be closed.
            stack.close()
            raise
        self._stack = stack
        return self

    def __exit__(self, *exc: object) -> None:
        self._agent = None
        stack, self._stack = self._stack, None
        if stack is not None:
            stack.close()

    def run(
        self,
        task: str,
        *,
        max_turns: int | None = None,
        abort: AbortFlag = lambda: False,
    ) -> tuple[State, Iterator[Event]]:
        """Run the built agent on ``task``; dispatch skills vs. the plain loop.

        Returns ``(state, events)`` exactly like ``Agent.run`` — ``events`` is a
        lazy generator the caller iterates to advance the loop, so it must be
        consumed while the session (and any open toolset) is still entered.
        """

        agent = self.agent
        turns = self._max_turns if max_turns is None else max_turns
        if self._skills is not None and self._skills.enabled:
            return run_with_skills(
                agent,
                task,
                skills=self._skills.skills,
                roots=self._skills.roots,
                preload=self._skills.preload,
                cwd=self._skills.cwd,
                max_turns=turns,
                abort=abort,
            )
        return agent.run(task, max_turns=turns, abort=abort)


# --------------------------------------------------------------------------
# Prompt + identity defaults (moved verbatim from the retired bash/ and
# bash_task/ subfolders so the presets and back-compat wrappers share them).
# --------------------------------------------------------------------------

BASH_AGENT_SYSTEM_PROMPT = (
    "You are a tiny bash-use agent. Use the bash tool to satisfy the task "
    "— parallel tool calls are fine when the steps are independent, and "
    "set `attach` on the call when you want to surface an image file. "
    "After the tool_result, return a short final answer."
)
BASH_AGENT_DEFAULT_ROLE = (
    "Use bash for local commands, then summarize what you observed."
)
BASH_AGENT_DEFAULT_NAME = "bash_agent"

BASH_TASK_AGENT_DEFAULT_NAME = "bash_task_agent"
BASH_TASK_AGENT_DEFAULT_ROLE = (
    "Drive the task with bash for focused commands and delegate heavy "
    "exploration or multi-file reads to the bash worker via the task tool."
)
# Small addendum bolted onto the bash-agent system prompt so the parent
# inherits all of its phrasing and only learns the extra `task` affordance.
BASH_TASK_EXPLORER_ADDENDUM = (
    "You also have a `task` tool that delegates a sub-task to a bash "
    'worker (`subagent_type="explorer"`); strongly prefer it for any '
    "investigation step (locating relevant code, mapping how a feature "
    "is used, reading a long file, tracing a failing test). Use `bash` "
    "directly only for the actual edits and the focused tests you run "
    "to verify them — not for exploration."
)
BASH_TASK_AGENT_SYSTEM_PROMPT = (
    BASH_AGENT_SYSTEM_PROMPT + "\n\n" + BASH_TASK_EXPLORER_ADDENDUM
)

EXPLORER_AGENT_DEFAULT_NAME = "explorer"
EXPLORER_AGENT_DEFAULT_ROLE = (
    "Run focused bash exploration in the workspace and return a short "
    "summary of what was found."
)
EXPLORER_AGENT_SYSTEM_PROMPT = (
    "You are a bash-using exploration sub-agent. Your job is to satisfy "
    "the delegated task by running bash commands (parallel reads when the "
    "steps are independent), then return a short final summary that the "
    "parent can use to plan its next edit. Prefer narrow tools (`grep -n`, "
    "`head`/`tail`, `sed -n 'A,Bp'`, `find`) over dumping whole files. "
    "Do NOT attempt edits — only investigate. Stop as soon as you have "
    "the evidence the task asks for."
)
DEFAULT_TASK_MAX_TURNS = 70

SKILL_AGENT_DEFAULT_NAME = "skill_agent"
SKILL_AGENT_DEFAULT_ROLE = (
    "You are a capable software agent with a bash tool and a read tool. Use the "
    "available skills when they fit the task: read a skill's SKILL.md, then run "
    "its scripts via bash. Work from evidence and verify your result."
)

MCP_AGENT_DEFAULT_NAME = "mcp_agent"
MCP_AGENT_DEFAULT_ROLE = (
    "Use the tools provided by the connected MCP server(s) to satisfy the "
    "task, then return a short final answer."
)

DEFAULT_AGENT_NAME = "agent"

SKILLS_ADDENDUM = (
    "You also have access to a library of skills. When a skill fits the task, "
    "read its SKILL.md first, then run its scripts via bash. Prefer a matching "
    "skill over improvising, and work from the evidence it produces."
)

MCP_ADDENDUM = (
    "You also have tools provided by one or more connected MCP servers (their "
    "names are prefixed with the server name). Use them when they fit the task, "
    "then return a short final answer."
)


def compose_agent_system_prompt(
    *, bash: bool, explorer: bool, skills: bool, mcp: bool
) -> str:
    """Build a system prompt by appending capability fragments to the base.

    The base is always the bash-agent prompt; ``explorer``/``skills``/``mcp``
    each append their fragment in that fixed order. Callers that pass an
    explicit ``system_prompt`` bypass this entirely. When ``bash`` is false the
    base still reads as the bash prompt, so a bash-less agent should supply its
    own ``system_prompt`` (see the design spec's edge-case note).
    """

    del bash  # base is unconditional today; kept for signature symmetry
    parts = [BASH_AGENT_SYSTEM_PROMPT]
    if explorer:
        parts.append(BASH_TASK_EXPLORER_ADDENDUM)
    if skills:
        parts.append(SKILLS_ADDENDUM)
    if mcp:
        parts.append(MCP_ADDENDUM)
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# Presets: each kind is a thin AgentSession configuration.
# --------------------------------------------------------------------------


def bash_session(
    provider: LLMProvider,
    *,
    cwd: str | Path | None = None,
    name: str = BASH_AGENT_DEFAULT_NAME,
    role: str = BASH_AGENT_DEFAULT_ROLE,
    system_prompt: str = BASH_AGENT_SYSTEM_PROMPT,
    request_extra: Mapping[str, Any] | None = None,
    max_turns: int = 10,
) -> AgentSession:
    """A session whose agent carries only the bash tool."""

    return AgentSession(
        provider=provider,
        name=name,
        role=role,
        system_prompt=system_prompt,
        target="user",
        static_tools=[make_bash_tool(cwd=cwd)],
        request_extra=request_extra,
        max_turns=max_turns,
    )


def bash_task_session(
    provider: LLMProvider,
    *,
    cwd: str | Path | None = None,
    name: str = BASH_TASK_AGENT_DEFAULT_NAME,
    role: str = BASH_TASK_AGENT_DEFAULT_ROLE,
    system_prompt: str = BASH_TASK_AGENT_SYSTEM_PROMPT,
    explorer_name: str = EXPLORER_AGENT_DEFAULT_NAME,
    explorer_role: str = EXPLORER_AGENT_DEFAULT_ROLE,
    explorer_system_prompt: str = EXPLORER_AGENT_SYSTEM_PROMPT,
    task_max_turns: int = DEFAULT_TASK_MAX_TURNS,
    request_extra: Mapping[str, Any] | None = None,
    max_turns: int = 10,
) -> AgentSession:
    """A session with bash + a `task` tool delegating to a bash explorer.

    Parent and explorer share ``cwd`` so a delegated investigation sees the
    same workspace the parent's edits affect.
    """

    explorer = make_bash_agent(
        provider=provider,
        cwd=cwd,
        name=explorer_name,
        role=explorer_role,
        system_prompt=explorer_system_prompt,
        request_extra=request_extra,
    )
    return AgentSession(
        provider=provider,
        name=name,
        role=role,
        system_prompt=system_prompt,
        target="user",
        static_tools=[
            make_bash_tool(cwd=cwd),
            task_tool([explorer], max_turns=task_max_turns),
        ],
        request_extra=request_extra,
        max_turns=max_turns,
    )


def skill_session(
    provider: LLMProvider,
    *,
    cwd: str | Path | None = None,
    roots: Sequence[SkillRoot] | None = None,
    skills: Sequence[SkillMetadata] | None = None,
    preload: Sequence[str] = (),
    name: str = SKILL_AGENT_DEFAULT_NAME,
    role: str = SKILL_AGENT_DEFAULT_ROLE,
    system_prompt: str | None = None,
    request_extra: Mapping[str, Any] | None = None,
    max_turns: int = 10,
) -> AgentSession:
    """A session with bash + read whose `run` advertises and injects skills.

    Skills are driven by the task text (a leading ``/mention`` or
    ``/no-skills``) plus any ``preload`` names — exactly like the interactive
    ``run_with_skills`` edge.
    """

    resolved_cwd = str(cwd) if cwd is not None else "."
    return AgentSession(
        provider=provider,
        name=name,
        role=role,
        system_prompt=role if system_prompt is None else system_prompt,
        target="user",
        static_tools=[make_bash_tool(cwd=cwd), make_read_tool(cwd=cwd)],
        skills=SkillConfig(
            enabled=True,
            roots=roots,
            skills=skills,
            preload=preload,
            cwd=resolved_cwd,
        ),
        request_extra=request_extra,
        max_turns=max_turns,
    )


def mcp_session(
    provider: LLMProvider,
    *,
    servers: Sequence["MCPServerConfig"],
    name: str = MCP_AGENT_DEFAULT_NAME,
    role: str = MCP_AGENT_DEFAULT_ROLE,
    system_prompt: str = "",
    static_tools: Sequence[AgentTool] = (),
    call_timeout: float = 60.0,
    connect: Callable[["MCPServerConfig"], "MCPConnection"] | None = None,
    request_extra: Mapping[str, Any] | None = None,
    max_turns: int = 10,
) -> AgentSession:
    """A session whose tools come from one or more live MCP servers.

    Each server becomes an :class:`MCPToolset` the session opens on enter and
    closes on exit. ``static_tools`` can add local tools (e.g. bash) alongside
    the MCP tools. ``connect`` is a test seam for the in-memory transport.
    """

    toolsets = [
        MCPToolset(server, call_timeout=call_timeout, connect=connect)
        for server in servers
    ]
    return AgentSession(
        provider=provider,
        name=name,
        role=role,
        system_prompt=system_prompt,
        target="user",
        static_tools=static_tools,
        toolsets=toolsets,
        request_extra=request_extra,
        max_turns=max_turns,
    )


# --------------------------------------------------------------------------
# Back-compat: callers that drive a plain `Agent` themselves (evals, the TUI
# gateway) keep these factories. Resource-free kinds need no session.
# --------------------------------------------------------------------------


def make_bash_agent(
    provider: LLMProvider,
    *,
    cwd: str | Path | None = None,
    name: str = BASH_AGENT_DEFAULT_NAME,
    role: str = BASH_AGENT_DEFAULT_ROLE,
    system_prompt: str = BASH_AGENT_SYSTEM_PROMPT,
    request_extra: Mapping[str, Any] | None = None,
) -> Agent:
    """Build a bash-using `Agent` with the bash tool already bound."""

    return make_llm_agent(
        name=name,
        provider=provider,
        role=role,
        tools=[make_bash_tool(cwd=cwd)],
        system_prompt=system_prompt,
        target="user",
        request_extra=request_extra,
    )


def make_bash_task_agent(
    provider: LLMProvider,
    *,
    cwd: str | Path | None = None,
    name: str = BASH_TASK_AGENT_DEFAULT_NAME,
    role: str = BASH_TASK_AGENT_DEFAULT_ROLE,
    system_prompt: str = BASH_TASK_AGENT_SYSTEM_PROMPT,
    explorer_name: str = EXPLORER_AGENT_DEFAULT_NAME,
    explorer_role: str = EXPLORER_AGENT_DEFAULT_ROLE,
    explorer_system_prompt: str = EXPLORER_AGENT_SYSTEM_PROMPT,
    task_max_turns: int = DEFAULT_TASK_MAX_TURNS,
    request_extra: Mapping[str, Any] | None = None,
) -> Agent:
    """Build a parent `Agent` with bash + task(explorer) tools."""

    explorer = make_bash_agent(
        provider=provider,
        cwd=cwd,
        name=explorer_name,
        role=explorer_role,
        system_prompt=explorer_system_prompt,
        request_extra=request_extra,
    )
    return make_llm_agent(
        name=name,
        provider=provider,
        role=role,
        tools=[
            make_bash_tool(cwd=cwd),
            task_tool([explorer], max_turns=task_max_turns),
        ],
        system_prompt=system_prompt,
        target="user",
        request_extra=request_extra,
    )
