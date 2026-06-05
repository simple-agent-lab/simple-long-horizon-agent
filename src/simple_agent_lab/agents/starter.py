"""The general agent starter: one runner + one composable front door.

This module replaces the per-kind subfolders. A single :class:`AgentSession`
(the runner) opens any :class:`~simple_agent_lab.agents.toolsets.Toolset`
resources, assembles the full tool list, builds an ``Agent`` via the existing
:func:`~simple_agent_lab.llm_agent.make_llm_agent`, and dispatches ``run`` to
either the plain loop or the skills loop. The composable
:func:`agent_session` front door configures that session by combining
capabilities (bash, read, explorer, skills, MCP) additively.

The capabilities differ only by their tool list and an optional skills flag —
there is no per-kind class. The explorer sub-agent is an ordinary ``task``
tool entry, and ``mcp`` servers are ``MCPToolset`` entries the session opens
and closes.

:func:`make_agent` is the definition-layer twin of :func:`agent_session`: same
resource-free capability flags, but it returns a bare ``Agent`` you run
yourself (no session). The ``make_bash_agent`` / ``make_bash_task_agent`` /
:func:`make_skill_agent` factories are thin wrappers over it — including
skills, because a skill is not a live resource but a *seed* (it records a menu
before the task). ``make_skill_agent`` installs that seed via the core
``Agent.seed`` hook, so a bare ``agent.run(task)`` is skills-aware with no
session and no separate run path. :func:`mcp_session` is a thin wrapper over
:func:`agent_session` for the MCP case, which *does* own a live resource and so
still needs the :class:`AgentSession` ``with`` block for deterministic
open/close.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, Sequence

from simple_agent_lab.context_view import ContextPolicy
from simple_agent_lab.core import Agent, SeedFn
from simple_agent_lab.llm import Provider as LLMProvider
from simple_agent_lab.llm_agent import make_llm_agent
from simple_agent_lab.messages import ContentInput
from simple_agent_lab.protocols import Event
from simple_agent_lab.skills import SkillMetadata, SkillRoot, seed_state_with_skills
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


def _skills_seed(config: SkillConfig) -> SeedFn:
    """Adapt a `SkillConfig` into a core `Agent.seed` callable.

    The bridge between the agents layer and the core seed hook: it binds the
    config's discovery keywords and forwards to
    :func:`~simple_agent_lab.skills.seed_state_with_skills`, so the resulting
    agent advertises skills on every ``run`` without a session. Skills are
    text-only, so a multimodal task is rejected (the directive parser needs a
    string).
    """

    def seed(agent: Agent, task: ContentInput) -> State:
        if not isinstance(task, str):
            raise TypeError("skills require a text task, not multimodal content")
        return seed_state_with_skills(
            agent,
            task,
            skills=config.skills,
            roots=config.roots,
            preload=config.preload,
            cwd=config.cwd,
        )

    return seed


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
        # Skills are a seed, not a resource: install them on the built agent so
        # `run` is a single `agent.run` for every capability mix. MCP toolsets
        # (opened below) are the only thing that needs the `with` lifetime.
        seed = (
            _skills_seed(self._skills)
            if self._skills is not None and self._skills.enabled
            else None
        )
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
                seed=seed,
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
        """Run the built agent on ``task``.

        A single ``agent.run``: any skills behavior comes from the seed
        installed in :meth:`__enter__`, so this method no longer branches on
        skills. Returns ``(state, events)`` exactly like ``Agent.run`` —
        ``events`` is a lazy generator the caller iterates to advance the loop,
        so it must be consumed while the session (and any open toolset) is
        still entered.
        """

        agent = self.agent
        turns = self._max_turns if max_turns is None else max_turns
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


def _assemble_static_tools(
    provider: LLMProvider,
    *,
    cwd: str | Path | None,
    bash: bool,
    read: bool,
    explorer: bool,
    tools: Sequence[AgentTool],
    request_extra: Mapping[str, Any] | None,
) -> list[AgentTool]:
    """Build the resource-free tool list shared by `make_agent`/`agent_session`.

    Order is deterministic: bash, read, the explorer `task` tool, then any
    caller-supplied `tools`. The explorer is an ordinary bash sub-agent wrapped
    as a `task` tool — exactly how a parent delegates today.
    """

    assembled: list[AgentTool] = []
    if bash:
        assembled.append(make_bash_tool(cwd=cwd))
    if read:
        assembled.append(make_read_tool(cwd=cwd))
    if explorer:
        explorer_agent = make_agent(
            provider,
            cwd=cwd,
            name=EXPLORER_AGENT_DEFAULT_NAME,
            role=EXPLORER_AGENT_DEFAULT_ROLE,
            system_prompt=EXPLORER_AGENT_SYSTEM_PROMPT,
            request_extra=request_extra,
        )
        assembled.append(task_tool([explorer_agent], max_turns=DEFAULT_TASK_MAX_TURNS))
    assembled.extend(tools)
    return assembled


def make_agent(
    provider: LLMProvider,
    *,
    cwd: str | Path | None = None,
    bash: bool = True,
    read: bool = False,
    explorer: bool = False,
    tools: Sequence[AgentTool] = (),
    name: str = DEFAULT_AGENT_NAME,
    role: str = "",
    system_prompt: str | None = None,
    context_policy: ContextPolicy | None = None,
    request_extra: Mapping[str, Any] | None = None,
    seed: SeedFn | None = None,
) -> Agent:
    """Build a stateless `Agent` by composing resource-free capabilities.

    The definition-layer twin of :func:`agent_session`: it shares the same
    ``bash``/``read``/``explorer``/``tools`` flags and prompt composition but
    returns a bare ``Agent`` you run yourself (``agent.run(...)``). It omits
    ``mcp_servers`` on purpose — those own a live resource and so require the
    :class:`AgentSession` runner. Skills, by contrast, are a ``seed`` (see
    :func:`make_skill_agent`), so they *can* live on a bare agent. When
    ``system_prompt`` is None the prompt is composed from the enabled
    capabilities' fragments; otherwise the explicit value is used verbatim.
    """

    assembled = _assemble_static_tools(
        provider,
        cwd=cwd,
        bash=bash,
        read=read,
        explorer=explorer,
        tools=tools,
        request_extra=request_extra,
    )
    if system_prompt is None:
        system_prompt = compose_agent_system_prompt(
            bash=bash, explorer=explorer, skills=False, mcp=False
        )
    return make_llm_agent(
        name=name,
        provider=provider,
        role=role,
        tools=assembled,
        system_prompt=system_prompt,
        target="user",
        context_policy=context_policy,
        request_extra=request_extra,
        seed=seed,
    )


def make_skill_agent(
    provider: LLMProvider,
    *,
    cwd: str | Path | None = None,
    roots: Sequence[SkillRoot] | None = None,
    skills: Sequence[SkillMetadata] | None = None,
    preload: Sequence[str] = (),
    read: bool = True,
    explorer: bool = False,
    tools: Sequence[AgentTool] = (),
    name: str = DEFAULT_AGENT_NAME,
    role: str = "",
    system_prompt: str | None = None,
    context_policy: ContextPolicy | None = None,
    request_extra: Mapping[str, Any] | None = None,
) -> Agent:
    """Build a bare, skills-aware `Agent` — the skills twin of `make_bash_agent`.

    Returns a plain ``Agent`` (no session, no runner wrapper), but installs a
    skills ``seed`` so ``agent.run(task)`` advertises the skills menu and
    injects mentioned/preloaded bodies on *every* run. That works because a
    skill is not a live resource that must be opened and closed — it is a way
    to seed the conversation, which the core ``Agent.seed`` hook supports
    directly. Skills imply the ``read`` tool (a skill reads its SKILL.md before
    running scripts), so ``read`` defaults to ``True`` here.

    For skills combined with a live resource like MCP, use
    ``agent_session(skills=..., mcp_servers=[...])`` — that needs the session's
    ``with`` block for the MCP connection, and reuses the same seed internally.
    """

    resolved_cwd = str(cwd) if cwd is not None else "."
    config = SkillConfig(
        enabled=True, roots=roots, skills=skills, preload=preload, cwd=resolved_cwd
    )
    if system_prompt is None:
        system_prompt = compose_agent_system_prompt(
            bash=True, explorer=explorer, skills=True, mcp=False
        )
    return make_agent(
        provider,
        cwd=cwd,
        bash=True,
        read=read,
        explorer=explorer,
        tools=tools,
        name=name,
        role=role,
        system_prompt=system_prompt,
        context_policy=context_policy,
        request_extra=request_extra,
        seed=_skills_seed(config),
    )


def agent_session(
    provider: LLMProvider,
    *,
    cwd: str | Path | None = None,
    bash: bool = True,
    read: bool = False,
    explorer: bool = False,
    skills: "SkillConfig | bool" = False,
    mcp_servers: Sequence["MCPServerConfig"] = (),
    tools: Sequence[AgentTool] = (),
    name: str = DEFAULT_AGENT_NAME,
    role: str = "",
    system_prompt: str | None = None,
    call_timeout: float = 60.0,
    connect: Callable[["MCPServerConfig"], "MCPConnection"] | None = None,
    context_policy: ContextPolicy | None = None,
    request_extra: Mapping[str, Any] | None = None,
    max_turns: int = 10,
) -> AgentSession:
    """Build one `AgentSession` by composing capabilities additively.

    Each capability is independent and combinable: ``bash``/``read`` add their
    local tools, ``explorer`` adds a `task` tool delegating to a bash explorer,
    ``tools`` appends arbitrary `AgentTool`s, ``mcp_servers`` each become an
    `MCPToolset` the session opens/closes, and ``skills`` (``True`` for defaults
    or a `SkillConfig`) routes `run` through the skills loop. Enabling skills
    implies the read tool (a skill reads its SKILL.md before running its
    scripts), so ``read`` need not be passed alongside ``skills``. When
    ``system_prompt`` is None the prompt is composed from per-capability
    fragments; otherwise the explicit value is used verbatim. ``connect`` is the
    MCP test seam (in-memory transport).
    """

    if skills is True:
        skill_config: SkillConfig | None = SkillConfig(
            enabled=True, cwd=str(cwd) if cwd is not None else "."
        )
    elif not skills:
        skill_config = None
    else:
        skill_config = skills  # an explicit SkillConfig
    skills_enabled = skill_config is not None and skill_config.enabled

    # Skills drive themselves through the read tool (read a SKILL.md, then run
    # its scripts via bash), so enabling skills implies read even when the
    # caller left `read` at its default.
    static_tools = _assemble_static_tools(
        provider,
        cwd=cwd,
        bash=bash,
        read=read or skills_enabled,
        explorer=explorer,
        tools=tools,
        request_extra=request_extra,
    )

    toolsets: list[Toolset] = [
        MCPToolset(server, call_timeout=call_timeout, connect=connect)
        for server in mcp_servers
    ]

    if system_prompt is None:
        system_prompt = compose_agent_system_prompt(
            bash=bash,
            explorer=explorer,
            skills=skills_enabled,
            mcp=bool(mcp_servers),
        )

    return AgentSession(
        provider=provider,
        name=name,
        role=role,
        system_prompt=system_prompt,
        target="user",
        static_tools=static_tools,
        toolsets=toolsets,
        skills=skill_config,
        context_policy=context_policy,
        request_extra=request_extra,
        max_turns=max_turns,
    )


# --------------------------------------------------------------------------
# Named session sugar: a thin wrapper that presets one capability on the
# composable `agent_session`. It adds no behavior of its own — composition
# stays intact, so e.g. `mcp_session(provider, [...], skills=True)` still
# works. It forwards every other keyword straight through. (Skills alone need
# no session — use `make_skill_agent` for that.)
# --------------------------------------------------------------------------


def mcp_session(
    provider: LLMProvider,
    mcp_servers: Sequence["MCPServerConfig"],
    **kwargs: Any,
) -> AgentSession:
    """`agent_session` wired to one or more MCP servers."""

    return agent_session(provider, mcp_servers=mcp_servers, **kwargs)


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
    """Build a bash-using `Agent` — `make_agent` with the bash-agent defaults."""

    return make_agent(
        provider,
        cwd=cwd,
        bash=True,
        name=name,
        role=role,
        system_prompt=system_prompt,
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
    """Build a parent `Agent` with bash + task(explorer) tools.

    Unlike ``make_agent(explorer=True)`` (which uses the fixed explorer
    defaults), this keeps the explorer's name/role/prompt and the task turn
    budget configurable, so it builds the explorer explicitly and passes it as
    a `task` tool via ``tools=``.
    """

    explorer = make_agent(
        provider,
        cwd=cwd,
        name=explorer_name,
        role=explorer_role,
        system_prompt=explorer_system_prompt,
        request_extra=request_extra,
    )
    return make_agent(
        provider,
        cwd=cwd,
        bash=True,
        tools=[task_tool([explorer], max_turns=task_max_turns)],
        name=name,
        role=role,
        system_prompt=system_prompt,
        request_extra=request_extra,
    )
