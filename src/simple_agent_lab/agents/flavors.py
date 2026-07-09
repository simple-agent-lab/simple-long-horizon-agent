"""Build shipped agent flavors from shared flavor names.

This module owns the mapping from a flavor string (``bash``, ``bash_task``,
``bash_task_read``, ``bash_skills``, ``loop``, ``goal``, ``pdr``) to concrete
agent capabilities. Runners pass in name/role/prompt/cwd; the agent layer
decides which tools, prompt addenda, sessions, or workflow choreography
implement that flavor.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import simple_agent_lab.config as config
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
from simple_agent_lab.tools import AbortFlag, AgentTool
from simple_agent_lab.tools.bash import make_bash_tool
from simple_agent_lab.tools.read import make_read_tool
from simple_agent_lab.workflow import (
    VERIFY_BEFORE_DONE_ADDENDUM,
    VERIFY_CONTINUATION,
    GoalBudgets,
    ThreadGoalResult,
    ThreadGoalStore,
    WorkflowResult,
    compose_workflow_trace_state,
    executed_completion_check,
    make_get_goal_tool,
    make_update_goal_tool,
    make_distiller_agent,
    never_abort,
    run_goal_loop,
    run_pdr,
    run_thread_goal_loop,
    update_goal_tool,
    workflow_steps_breakdown,
    write_workflow_subagent_traces,
)

from .starter import (
    BASH_TASK_ADDENDUM,
    make_agent,
)

# The threshold fallback used only when the provider's context window is
# unknown, so it stays here with the logic that needs it rather than as an
# env knob in `simple_agent_lab.config`. Kept high on purpose: an unregistered
# window should not trigger aggressive summarization far below a modern
# large-window model's real capacity — better to under-compress than to fold
# away working context when the window is merely unknown. Register the model in
# `model_metadata.DEFAULT_CONTEXT_WINDOWS` (or a window book) for an exact value.
DEFAULT_AGENT_COMPRESSION_FALLBACK_THRESHOLD_TOKENS = 400_000

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
    solver_read: bool = True,
    solver_task: bool = False,
) -> Agent:
    """Build an Agent for one shipped flavor, simple or workflow.

    Dispatches on the flavor vocabulary: workflow flavors (``loop``, ``goal``,
    ``pdr``) return a facade Agent that runs the whole arm in its single
    ``generate``; every other flavor returns a resource-free simple Agent. The
    workflow-only knobs (``prepare_workspace``/``trace_put``) are ignored by
    simple flavors, and the simple-only knobs (``hooks``/``tools``) are ignored
    by workflow flavors.
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
            solver_read=solver_read,
            solver_task=solver_task,
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
        solver_task=solver_task,
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
    solver_task: bool = False,
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
            general_purpose=solver_task,
            tools=tools,
            name=name,
            role=role,
            system_prompt=_with_task_addendum(system_prompt)
            if solver_task
            else system_prompt,
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
            general_purpose=True,
            tools=tools,
            name=name,
            role=role,
            system_prompt=_with_task_addendum(system_prompt),
            context_policy=policy,
            request_extra=request_extra,
            hooks=hooks,
            bash_exec_prefix=bash_exec_prefix,
        )
    if flavor == "bash_task_read":
        # bash + the dedicated read tool + a task(general-purpose) sub-agent, so
        # the parent can read files directly while still delegating wider work.
        return make_agent(
            provider,
            cwd=cwd,
            bash=True,
            read=True,
            general_purpose=True,
            tools=tools,
            name=name,
            role=role,
            system_prompt=_with_task_addendum(system_prompt),
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
    solver_read: bool = True,
    solver_task: bool = False,
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
            solver_read=solver_read,
            solver_task=solver_task,
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


def run_goal_flavor(
    provider: Provider,
    workdir: Path,
    request_extra: Mapping[str, Any] | None,
    *,
    name: str,
    role: str,
    system_prompt: str,
    context_policy: ContextPolicy | None,
    bash_exec_prefix: tuple[str, ...] = (),
    solver_read: bool,
    solver_task: bool,
    objective: str,
    state: State | None = None,
    steering_preface: str = "",
    loop_turns: int | None = None,
    inner_max_turns: int | None = None,
    abort: AbortFlag = never_abort,
) -> ThreadGoalResult:
    """Run the Codex-style ``goal`` flavor and return its ``ThreadGoalResult``.

    Single construction site shared by the standalone ``goal`` workflow facade
    (``run_goal_arm``) and the repo-chain runner, so both build the identical
    bash solver + ``get_goal``/``update_goal`` tools and drive the identical
    ``run_thread_goal_loop``. A chain only needs two extra knobs:

    - ``state``: seed the loop with the shared chain state so the goal solver
      inherits every earlier instance's context (resumes on it from segment 1).
    - ``steering_preface``: trusted host framing prepended to each steering
      message (e.g. "this is one long chain of sub-problems; reuse the context").

    With both left at their defaults this reproduces the standalone ``goal`` arm
    exactly, so ``goal`` behaves the same inside and outside a chain.

    ``abort`` is forwarded to the goal loop so a caller can stop the run at a
    turn boundary (the repo chain uses it to break out when the active context
    reaches the context-window handoff threshold, then reset and resume).
    """

    goal_store = ThreadGoalStore()
    agent = _solver_agent(
        provider,
        workdir,
        request_extra,
        name=name,
        role=role,
        system_prompt=system_prompt,
        context_policy=context_policy,
        extra_tools=[
            make_get_goal_tool(goal_store),
            make_update_goal_tool(goal_store),
        ],
        bash_exec_prefix=bash_exec_prefix,
        read=solver_read,
        task=solver_task,
    )
    return run_thread_goal_loop(
        agent,
        objective,
        budgets=GoalBudgets(
            max_turns=(
                loop_turns if loop_turns is not None else config.LOOP_MAX_TURNS.get()
            )
        ),
        inner_max_turns=(
            inner_max_turns
            if inner_max_turns is not None
            else config.WORKER_MAX_TURNS.get()
        ),
        goal_store=goal_store,
        state=state,
        steering_preface=steering_preface,
        abort=abort,
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
    solver_read: bool = True,
    solver_task: bool = False,
) -> WorkflowRunner:
    """Build the `task -> WorkflowResult` runner for a workflow flavor."""

    selected = flavor.strip().lower()
    worker_max_turns = config.WORKER_MAX_TURNS.get()

    if selected == "loop":
        loop_turns = config.LOOP_MAX_TURNS.get()
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
            read=solver_read,
            task=solver_task,
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

    if selected == "goal":

        def run_goal_arm(task: str) -> WorkflowResult:
            result = run_goal_flavor(
                provider,
                workdir,
                request_extra,
                name=name,
                role=role,
                system_prompt=system_prompt,
                context_policy=context_policy,
                bash_exec_prefix=bash_exec_prefix,
                solver_read=solver_read,
                solver_task=solver_task,
                objective=task,
                inner_max_turns=worker_max_turns,
            )
            return WorkflowResult(output=result.output, steps=result.steps)

        return run_goal_arm

    if selected == "pdr":
        rounds = config.PDR_ROUNDS.get()
        width = config.PDR_WIDTH.get()
        # Opt-in cost guard. Shorter throwaway attempts can cut cost, but may
        # reduce the quality of the distilled brief, so default to full budget.
        attempt_turns = config.PDR_ATTEMPT_TURNS.get(default=worker_max_turns)
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
                        read=solver_read,
                        task=solver_task,
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
                    read=solver_read,
                    task=solver_task,
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


def _with_task_addendum(system_prompt: str) -> str:
    if BASH_TASK_ADDENDUM in system_prompt:
        return system_prompt
    return "\n\n".join(part for part in (system_prompt, BASH_TASK_ADDENDUM) if part)


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
    keep_recent = config.COMPRESSION_KEEP_RECENT.get()
    if threshold <= 0:
        return None
    compressor = make_llm_agent(
        name="context_compressor",
        provider=provider,
        role="Compact older agent context into durable working memory.",
        system_prompt=(
            "You compact an agent's own working transcript into durable memory "
            "it will keep reading. Be faithful: preserve facts exactly and never "
            "invent — if unsure, omit. Preserve identifiers VERBATIM (file "
            "paths, symbols, commands, error strings, test names, IDs, numbers); "
            "do not paraphrase them. Capture not just what is true but the state "
            "of the work — what is done, what remains, and which approaches "
            "already failed (with the reason, so they are not retried). Drop "
            "chit-chat and superseded detail. The summary must stand alone: the "
            "agent should be able to continue from it plus the few recent "
            "messages."
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
    override = config.COMPRESSION_THRESHOLD.get(default=None)
    if override is not None:
        return override
    window = provider.context_window or default_context_window_book().window_for(
        provider.model
    )
    if not window:
        return DEFAULT_AGENT_COMPRESSION_FALLBACK_THRESHOLD_TOKENS
    ratio = config.COMPRESSION_WINDOW_RATIO.get()
    return max(1, int(window * ratio))


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
    read: bool = True,
    task: bool = False,
) -> Agent:
    """A solver worker rooted at `cwd`."""

    prompt = system_prompt
    if extra_prompt:
        prompt = f"{prompt}\n\n{extra_prompt}" if prompt else extra_prompt
    if task:
        prompt = _with_task_addendum(prompt)
    init_state = _reset_init_state(cwd, reset_to) if reset_to else None
    return make_agent(
        provider,
        cwd=cwd,
        bash=True,
        read=read,
        general_purpose=task,
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
    for message in reversed(visible):
        if message.kind == "task":
            return text_of(message.content)
    return text_of(visible[0].content) if visible else ""
