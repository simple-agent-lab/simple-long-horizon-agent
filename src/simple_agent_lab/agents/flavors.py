"""Build shipped agent flavors from shared flavor names.

This module owns the mapping from a flavor string (``bash``, ``bash_task``,
``bash_task_read``, ``bash_skills``, ``loop``, ``pdr``) to concrete agent
capabilities. Runners pass in name/role/prompt/cwd; the agent layer decides
which tools, prompt addenda, sessions, or workflow choreography implement that
flavor.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from simple_agent_lab.agent_flavors import SIMPLE_AGENT_FLAVORS, WORKFLOW_AGENT_FLAVORS
from simple_agent_lab.compression import SummarizeStrategy
from simple_agent_lab.context_view import ContextPolicy
from simple_agent_lab.core import Agent
from simple_agent_lab.hooks import HookContext, HookMap, HookPoint
from simple_agent_lab.llm import Provider
from simple_agent_lab.llm_agent import make_llm_agent
from simple_agent_lab.messages import Message, assistant_message, text_of
from simple_agent_lab.model_metadata import default_context_window_book
from simple_agent_lab.skills import system_prompt_with_skills
from simple_agent_lab.state import State
from simple_agent_lab.tools import AgentTool
from simple_agent_lab.tools.bash import make_bash_tool
from simple_agent_lab.tools.read import make_read_tool
from simple_agent_lab.workflow import (
    VERIFY_BEFORE_DONE_ADDENDUM,
    VERIFY_CONTINUATION,
    GoalBudgets,
    WorkflowResult,
    compose_workflow_trace_state,
    executed_completion_check,
    make_distiller_agent,
    run_goal_loop,
    run_pdr,
    update_goal_tool,
    workflow_steps_breakdown,
    write_workflow_subagent_traces,
)

from .starter import (
    BASH_TASK_EXPLORER_ADDENDUM,
    make_agent,
)

PDR_ROUNDS_ENV = "SWE_PDR_ROUNDS"
PDR_WIDTH_ENV = "SWE_PDR_WIDTH"
PDR_ATTEMPT_TURNS_ENV = "SWE_PDR_ATTEMPT_TURNS"
LOOP_MAX_TURNS_ENV = "SWE_LOOP_MAX_TURNS"
WORKER_MAX_TURNS_ENV = "SWE_WORKER_MAX_TURNS"
AGENT_COMPRESSION_THRESHOLD_ENV = "SAL_AGENT_COMPRESSION_THRESHOLD_TOKENS"
AGENT_COMPRESSION_WINDOW_RATIO_ENV = "SAL_AGENT_COMPRESSION_WINDOW_RATIO"
AGENT_COMPRESSION_KEEP_RECENT_ENV = "SAL_AGENT_COMPRESSION_KEEP_RECENT"
DEFAULT_AGENT_COMPRESSION_WINDOW_RATIO = 0.8
DEFAULT_AGENT_COMPRESSION_FALLBACK_THRESHOLD_TOKENS = 80_000
DEFAULT_AGENT_COMPRESSION_KEEP_RECENT = 4

WorkflowRunner = Callable[[str], WorkflowResult]
PrepareWorkflowWorkspace = Callable[[Path], None]
ArtifactPut = Callable[[str, bytes], None]


def build_flavor_agent(
    *,
    flavor: str,
    provider: Provider,
    cwd: Path,
    name: str = "agent",
    role: str = "",
    system_prompt: str = "",
    request_extra: Mapping[str, Any] | None = None,
    hooks: HookMap | None = None,
    context_policy: ContextPolicy | None = None,
    enable_default_compression: bool = True,
    tools: Sequence[AgentTool] = (),
    bash_exec_prefix: tuple[str, ...] = (),
    prepare_workspace: PrepareWorkflowWorkspace | None = None,
    trace_put: ArtifactPut | None = None,
) -> Agent:
    """Build an Agent for one shipped flavor, simple or workflow.

    Dispatches on the flavor vocabulary: workflow flavors (``loop``, ``pdr``)
    return a facade Agent that runs the whole arm in its single ``generate``;
    every other flavor returns a resource-free simple Agent. The workflow-only
    knobs (``prepare_workspace``/``trace_put``) are ignored by simple flavors,
    and the simple-only knobs (``hooks``/``tools``) are ignored by workflow
    flavors.
    """

    if flavor.strip().lower() in WORKFLOW_AGENT_FLAVORS:
        return _build_workflow_flavor_agent(
            flavor=flavor,
            provider=provider,
            cwd=cwd,
            name=name,
            role=role,
            system_prompt=system_prompt,
            request_extra=request_extra,
            prepare_workspace=prepare_workspace,
            trace_put=trace_put,
            context_policy=context_policy,
            enable_default_compression=enable_default_compression,
            bash_exec_prefix=bash_exec_prefix,
        )
    return _build_simple_flavor_agent(
        flavor=flavor,
        provider=provider,
        cwd=cwd,
        name=name,
        role=role,
        system_prompt=system_prompt,
        request_extra=request_extra,
        hooks=hooks,
        context_policy=context_policy,
        enable_default_compression=enable_default_compression,
        tools=tools,
        bash_exec_prefix=bash_exec_prefix,
    )


def _build_simple_flavor_agent(
    *,
    flavor: str,
    provider: Provider,
    cwd: Path,
    name: str = "agent",
    role: str = "",
    system_prompt: str = "",
    request_extra: Mapping[str, Any] | None = None,
    hooks: HookMap | None = None,
    context_policy: ContextPolicy | None = None,
    enable_default_compression: bool = True,
    tools: Sequence[AgentTool] = (),
    bash_exec_prefix: tuple[str, ...] = (),
) -> Agent:
    """Build a resource-free Agent for one shipped simple flavor."""

    hooks = hooks or {}
    policy = _resolve_context_policy(
        provider,
        request_extra=request_extra,
        context_policy=context_policy,
        enable_default_compression=enable_default_compression,
    )
    if flavor == "bash":
        return make_agent(
            provider,
            cwd=cwd,
            bash=True,
            tools=tools,
            name=name,
            role=role,
            system_prompt=system_prompt,
            context_policy=policy,
            request_extra=request_extra,
            hooks=hooks,
            bash_exec_prefix=bash_exec_prefix,
        )
    if flavor == "bash_task":
        return make_agent(
            provider,
            cwd=cwd,
            bash=True,
            explorer=True,
            tools=tools,
            name=name,
            role=role,
            system_prompt=_with_explorer_addendum(system_prompt),
            context_policy=policy,
            request_extra=request_extra,
            hooks=hooks,
            bash_exec_prefix=bash_exec_prefix,
        )
    if flavor == "bash_task_read":
        # bash + the dedicated read tool + a task(explorer) sub-agent, so the
        # parent can read files directly while still delegating wider exploration.
        return make_agent(
            provider,
            cwd=cwd,
            bash=True,
            read=True,
            explorer=True,
            tools=tools,
            name=name,
            role=role,
            system_prompt=_with_explorer_addendum(system_prompt),
            context_policy=policy,
            request_extra=request_extra,
            hooks=hooks,
            bash_exec_prefix=bash_exec_prefix,
        )
    if flavor == "bash_skills":
        # Skills are prompt/state behavior plus bash/read tools. The benchmark
        # path folds the discovered menu into the system prompt because there is
        # no separate interactive session message to carry it.
        skill_tools = [
            make_bash_tool(cwd=cwd, exec_prefix=bash_exec_prefix),
            make_read_tool(cwd=cwd),
            *tools,
        ]
        return make_llm_agent(
            name=name,
            provider=provider,
            role=role,
            tools=skill_tools,
            system_prompt=system_prompt_with_skills(system_prompt, cwd=cwd),
            target="user",
            context_policy=policy,
            request_extra=request_extra,
            hooks=hooks,
        )
    raise SystemExit(
        f"Unsupported agent flavor {flavor!r}; expected one of {SIMPLE_AGENT_FLAVORS}."
    )


def _build_workflow_flavor_agent(
    *,
    flavor: str,
    provider: Provider,
    cwd: Path,
    name: str = "agent",
    role: str = "",
    system_prompt: str = "",
    request_extra: Mapping[str, Any] | None = None,
    prepare_workspace: PrepareWorkflowWorkspace | None = None,
    trace_put: ArtifactPut | None = None,
    context_policy: ContextPolicy | None = None,
    enable_default_compression: bool = True,
    bash_exec_prefix: tuple[str, ...] = (),
) -> Agent:
    """Build the facade Agent for one shipped workflow flavor.

    The facade delays creating workers until its single `generate` call, so a
    build_agent probe constructs no worktrees or sub-agents until the run
    actually starts.

    The facade owns all workflow bookkeeping so callers (eval suites) do not
    each re-implement it: it writes the sub-agent traces via ``trace_put`` and
    stashes the per-step breakdown on ``state.data["workflow"]`` at session end,
    where the generic runner folds it into the result.
    """

    selected = flavor.strip().lower()
    if selected not in WORKFLOW_AGENT_FLAVORS:
        raise SystemExit(
            f"Unsupported workflow flavor {selected!r}; "
            f"expected one of {WORKFLOW_AGENT_FLAVORS}."
        )
    workdir = Path(cwd)
    policy = _resolve_context_policy(
        provider,
        request_extra=request_extra,
        context_policy=context_policy,
        enable_default_compression=enable_default_compression,
    )
    last_overview: list[dict[str, Any]] | None = None
    last_final_output = ""
    last_breakdown: dict[str, Any] | None = None

    def generate(visible: list[Message]) -> Message:
        nonlocal last_overview, last_final_output, last_breakdown
        runner = make_workflow_runner_for_flavor(
            selected,
            provider,
            workdir,
            request_extra=request_extra,
            name=name,
            role=role,
            system_prompt=system_prompt,
            context_policy=policy,
            prepare_workspace=prepare_workspace,
            bash_exec_prefix=bash_exec_prefix,
        )
        result = runner(_task_text(visible))
        last_overview = (
            write_workflow_subagent_traces(result, selected, trace_put)
            if trace_put is not None
            else []
        )
        last_final_output = result.output or ""
        last_breakdown = workflow_steps_breakdown(result, selected)
        return assistant_message(
            result.output or "", sender=name, target="user", kind="final"
        )

    def stash_breakdown(ctx: HookContext) -> None:
        # Hand the per-step breakdown to the eval runner without coupling the
        # agent layer to it: the runner reads `state.data["workflow"]` and folds
        # it into the result product.
        if last_breakdown is not None:
            ctx.state.data["workflow"] = last_breakdown
        return None

    def compose_trace(agent: Agent, state: State) -> State | None:
        return compose_workflow_trace_state(
            state,
            overview=last_overview,
            final_output=last_final_output,
            agent_name=agent.name,
        )

    return Agent(
        name=name,
        generate=generate,
        role=role,
        hooks={HookPoint.SESSION_END: [stash_breakdown]},
        compose_trace_state=compose_trace,
    )


def make_workflow_runner_for_flavor(
    flavor: str,
    provider: Provider,
    workdir: Path,
    *,
    request_extra: Mapping[str, Any] | None = None,
    name: str = "agent",
    role: str = "",
    system_prompt: str = "",
    context_policy: ContextPolicy | None = None,
    prepare_workspace: PrepareWorkflowWorkspace | None = None,
    bash_exec_prefix: tuple[str, ...] = (),
) -> WorkflowRunner:
    """Build the `task -> WorkflowResult` runner for a workflow flavor."""

    selected = flavor.strip().lower()
    worker_max_turns = _env_int(WORKER_MAX_TURNS_ENV, 40)

    if selected == "loop":
        loop_turns = _env_int(LOOP_MAX_TURNS_ENV, 6)
        # The judge-gated loop is the general workflow optimization; this flavor
        # supplies a bash+read solver and the per-task budget.
        agent = _solver_agent(
            provider,
            workdir,
            request_extra,
            name=name,
            role=role,
            system_prompt=system_prompt,
            context_policy=context_policy,
            extra_tools=[update_goal_tool()],
            extra_prompt=VERIFY_BEFORE_DONE_ADDENDUM,
            bash_exec_prefix=bash_exec_prefix,
        )
        completion_check = executed_completion_check(
            cwd=workdir, exec_prefix=bash_exec_prefix
        )

        def run_loop(task: str) -> WorkflowResult:
            result = run_goal_loop(
                agent,
                task,
                # Model-declared completion is gated by actually re-running the
                # model's declared verify command in the workspace.
                check=completion_check,
                budgets=GoalBudgets(max_turns=loop_turns),
                inner_max_turns=worker_max_turns,
                # Feed the task verbatim on turn 1, so the loop flavor differs
                # only by its continuations, not by a reframed task.
                initial_prompt=lambda objective: objective,
                continuation_prompt=lambda objective: VERIFY_CONTINUATION,
            )
            return WorkflowResult(output=result.output, steps=result.steps)

        return run_loop

    if selected == "pdr":
        rounds = _env_int(PDR_ROUNDS_ENV, 2)
        width = _env_int(PDR_WIDTH_ENV, 3)
        # Opt-in cost guard. Shorter throwaway attempts can cut cost, but may
        # reduce the quality of the distilled brief, so default to full budget.
        attempt_turns = _env_int(PDR_ATTEMPT_TURNS_ENV, worker_max_turns)
        distiller = make_distiller_agent(provider, request_extra=request_extra)

        def run_pdr_arm(task: str) -> WorkflowResult:
            baseline = _baseline_commit(workdir)
            if prepare_workspace is not None:
                prepare_workspace(workdir)
            root = Path(tempfile.mkdtemp(prefix="sal-pdr-"))
            worktrees = [
                _add_worktree(workdir, baseline, root, i) for i in range(width)
            ]
            try:
                attempts = [
                    _solver_agent(
                        provider,
                        worktree,
                        request_extra,
                        name=f"attempt_{i}",
                        role=role,
                        system_prompt=system_prompt,
                        context_policy=context_policy,
                        reset_to=baseline,
                        extra_prompt=_workspace_note(worktree),
                        bash_exec_prefix=bash_exec_prefix,
                    )
                    for i, worktree in enumerate(worktrees)
                ]
                # The finalizer writes the real answer in the canonical workspace
                # (no reset): its edits are the product read by the caller.
                finalizer = _solver_agent(
                    provider,
                    workdir,
                    request_extra,
                    name="finalizer",
                    role=role,
                    system_prompt=system_prompt,
                    context_policy=context_policy,
                    bash_exec_prefix=bash_exec_prefix,
                )
                return run_pdr(
                    attempts,
                    distiller,
                    task,
                    rounds=rounds,
                    finalizer=finalizer,
                    worker_max_turns=attempt_turns,
                    finalizer_max_turns=worker_max_turns,
                )
            finally:
                _remove_worktrees(workdir, worktrees, root)

        return run_pdr_arm

    raise SystemExit(
        f"Unsupported workflow flavor {selected!r}; "
        f"expected one of {WORKFLOW_AGENT_FLAVORS}."
    )


def _with_explorer_addendum(system_prompt: str) -> str:
    if BASH_TASK_EXPLORER_ADDENDUM in system_prompt:
        return system_prompt
    return "\n\n".join(
        part for part in (system_prompt, BASH_TASK_EXPLORER_ADDENDUM) if part
    )


def _resolve_context_policy(
    provider: Provider,
    *,
    request_extra: Mapping[str, Any] | None,
    context_policy: ContextPolicy | None,
    enable_default_compression: bool,
) -> ContextPolicy | None:
    if context_policy is not None:
        return context_policy
    if not enable_default_compression:
        return None
    threshold = _compression_threshold(provider)
    keep_recent = _env_int(
        AGENT_COMPRESSION_KEEP_RECENT_ENV,
        DEFAULT_AGENT_COMPRESSION_KEEP_RECENT,
        minimum=0,
    )
    if threshold <= 0:
        return None
    compressor = make_llm_agent(
        name="context_compressor",
        provider=provider,
        role="Summarize older agent context before it exceeds the model window.",
        system_prompt=(
            "You compress long agent transcripts. Produce a concise, faithful "
            "summary that preserves durable facts, constraints, tool results, "
            "decisions, open questions, and file paths. Do not invent facts."
        ),
        tools=(),
        request_extra=request_extra,
    )
    return ContextPolicy(
        strategy=SummarizeStrategy(
            compressor=compressor,
            threshold_tokens=threshold,
            keep_recent=keep_recent,
        )
    )


def _compression_threshold(provider: Provider) -> int:
    override = os.environ.get(AGENT_COMPRESSION_THRESHOLD_ENV)
    if override is not None and override.strip():
        return _env_int(
            AGENT_COMPRESSION_THRESHOLD_ENV,
            DEFAULT_AGENT_COMPRESSION_FALLBACK_THRESHOLD_TOKENS,
            minimum=0,
        )
    window = provider.context_window or default_context_window_book().window_for(
        provider.model
    )
    if not window:
        return DEFAULT_AGENT_COMPRESSION_FALLBACK_THRESHOLD_TOKENS
    ratio = _env_float(
        AGENT_COMPRESSION_WINDOW_RATIO_ENV,
        DEFAULT_AGENT_COMPRESSION_WINDOW_RATIO,
    )
    return max(1, int(window * ratio))


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _git(
    args: list[str], cwd: Path, *, input_text: str | None = None, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        input=input_text,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=check,
    )


def _baseline_commit(workdir: Path) -> str:
    """The commit the workflow starts from."""

    return _git(["rev-parse", "HEAD"], workdir).stdout.strip()


def _add_worktree(workdir: Path, baseline: str, root: Path, index: int) -> Path:
    """Create a detached worktree at `baseline`, outside the canonical workspace."""

    path = root / f"wt-{index}"
    _git(["worktree", "add", "--detach", str(path), baseline], workdir)
    return path


def _reset_worktree(worktree: Path, baseline: str) -> None:
    """Return a worktree to a clean `baseline` checkout."""

    _git(["reset", "--hard", baseline], worktree, check=False)
    _git(["clean", "-fdx"], worktree, check=False)


def _remove_worktrees(workdir: Path, worktrees: Sequence[Path], root: Path) -> None:
    for worktree in worktrees:
        _git(["worktree", "remove", "--force", str(worktree)], workdir, check=False)
    _git(["worktree", "prune"], workdir, check=False)
    subprocess.run(["rm", "-rf", str(root)], check=False)


def _workspace_note(worktree: Path) -> str:
    """A system-prompt line pinning a worker to its own checkout."""

    return (
        f"IMPORTANT: your entire workspace is `{worktree}`. Every bash command "
        "you run is rooted there; use paths relative to it and never `cd` to a "
        "different directory (ignore any other workspace path in the task)."
    )


def _reset_init_state(worktree: Path, baseline: str):
    """An `init_state` hook that resets `worktree` to `baseline` before each run."""

    def init_state(agent: Agent, task: Any) -> State:
        _reset_worktree(worktree, baseline)
        return agent._default_init_state(task)

    return init_state


def _solver_agent(
    provider: Provider,
    cwd: Path,
    request_extra: Mapping[str, Any] | None,
    *,
    name: str,
    role: str,
    system_prompt: str,
    context_policy: ContextPolicy | None,
    reset_to: str | None = None,
    extra_tools: Sequence[Any] = (),
    extra_prompt: str = "",
    bash_exec_prefix: tuple[str, ...] = (),
) -> Agent:
    """A bash+read worker rooted at `cwd`."""

    prompt = system_prompt
    if extra_prompt:
        prompt = f"{prompt}\n\n{extra_prompt}" if prompt else extra_prompt
    init_state = _reset_init_state(cwd, reset_to) if reset_to else None
    return make_agent(
        provider,
        cwd=cwd,
        bash=True,
        read=True,
        tools=list(extra_tools),
        name=name,
        role=role,
        system_prompt=prompt,
        context_policy=context_policy,
        request_extra=request_extra,
        init_state=init_state,
        bash_exec_prefix=bash_exec_prefix,
    )


def _task_text(visible: list[Message]) -> str:
    for message in visible:
        if message.kind == "task":
            return text_of(message.content)
    return text_of(visible[0].content) if visible else ""
