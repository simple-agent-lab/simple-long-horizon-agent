"""SWE-bench container half backed by dynamic JavaScript workflows.

The generic eval runner still drives one facade ``Agent``. That facade generates
or loads ``workflow.js``, executes it with ``simple_agent_lab.dynamic_workflows``,
and lets JS ``agent()`` calls run normal SWE-bench bash agents against the same
workspace. Patch extraction remains the standard SWE-bench ``git diff`` path.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from simple_agent_lab.agents.starter import make_bash_task_agent
from simple_agent_lab.core import Agent
from simple_agent_lab.dynamic_workflows import (
    AgentCallOptions,
    DynamicWorkflowRuntime,
    SimpleAgentCallRunner,
    WorkflowRuntimeOptions,
    generate_workflow_script,
    read_workflow_artifacts,
)
from simple_agent_lab.llm import Provider
from simple_agent_lab.llm_agent import make_llm_agent
from simple_agent_lab.messages import Message, assistant_message, text_of
from simple_agent_lab.skills import system_prompt_with_skills
from simple_agent_lab.tools.bash import make_bash_tool
from simple_agent_lab.tools.read import make_read_tool

from .container import (  # noqa: F401  (suite surface re-exports)
    AGENT_FLAVOR_ENV,
    AGENT_NAME,
    AGENT_ROLE,
    agent_spec,
    apply_oracle,
    build_task,
    evaluate,
    prepare,
)
from .container import extract_result as _base_extract_result

SWE_DYNAMIC_WORKFLOW_SCRIPT_ENV = "SWEBENCH_DYNAMIC_WORKFLOW_SCRIPT"
SWE_DYNAMIC_WORKFLOW_MAX_CONCURRENCY_ENV = "SWEBENCH_DYNAMIC_MAX_CONCURRENCY"
SWE_DYNAMIC_WORKFLOW_MAX_AGENTS_ENV = "SWEBENCH_DYNAMIC_MAX_AGENTS"
SWE_DYNAMIC_WORKFLOW_TIMEOUT_ENV = "SWEBENCH_DYNAMIC_TIMEOUT"
SWE_DYNAMIC_WORKFLOW_ARTIFACT_DIR = ".simple-agent-lab/dynamic_workflow"

DEFAULT_DYNAMIC_WORKFLOW_SCRIPT = r"""
phase("investigate");
const investigation = await agent(
  `Inspect the repository and problem. Do not edit files. Use find/grep/sed; do not assume rg is installed. Identify the likely files, behavior, and smallest implementation plan.\n\nTask:\n${args.task}`,
  { name: "investigator", role: "Read-only SWE-bench investigator.", maxTurns: 8, cacheKey: "investigate" }
);

phase("implement");
const implementation = await agent(
  `Implement the fix in the repository. Use find/grep/sed; do not assume rg is installed. Use the investigation notes, edit only necessary source files, and run focused verification when practical. Finish with a short final summary after edits.\n\nTask:\n${args.task}\n\nInvestigation notes:\n${investigation.output}`,
  { name: "implementer", role: "SWE-bench implementer. You may edit files.", maxTurns: 24, cacheKey: "implement" }
);

phase("review");
const review = await agent(
  `Review the current workspace after implementation. Use find/grep/sed; do not assume rg is installed. Inspect the diff and run focused checks if available. Do not make broad unrelated changes. Finish with a short final summary.\n\nTask:\n${args.task}\n\nImplementation notes:\n${implementation.output}`,
  { name: "reviewer", role: "SWE-bench reviewer and verifier.", maxTurns: 10, cacheKey: "review" }
);

phase("finalize");
return [
  "Dynamic SWE-bench workflow completed.",
  "",
  "Investigation:",
  investigation.output,
  "",
  "Implementation:",
  implementation.output,
  "",
  "Review:",
  review.output
].join("\n");
"""


def build_agent(
    *,
    provider: Provider,
    cwd: Path,
    request_extra: Mapping[str, Any] | None = None,
) -> Agent:
    """Facade agent that solves one SWE-bench case through a dynamic workflow."""

    artifacts_dir = Path(cwd) / SWE_DYNAMIC_WORKFLOW_ARTIFACT_DIR

    def build_subagent(
        options: AgentCallOptions, sub_provider: Provider, sub_cwd: Path
    ) -> Agent:
        return _build_swebench_subagent(
            options=options,
            provider=sub_provider,
            cwd=sub_cwd,
            request_extra=request_extra,
        )

    def generate(visible: list[Message]) -> Message:
        task = _task_text(visible)
        script = _load_or_generate_script(
            provider=provider,
            task=task,
            request_extra=request_extra,
        )
        runner = SimpleAgentCallRunner(
            provider=provider,
            cwd=cwd,
            request_extra=request_extra,
            build_agent=build_subagent,
        )
        runtime = DynamicWorkflowRuntime(
            runner=runner,
            options=WorkflowRuntimeOptions(
                # SWE-bench subagents share one worktree; serialize by default so
                # generated workflows cannot race concurrent writes.
                max_concurrency=_env_int(SWE_DYNAMIC_WORKFLOW_MAX_CONCURRENCY_ENV, 1),
                max_agents=_env_int(SWE_DYNAMIC_WORKFLOW_MAX_AGENTS_ENV, 12),
                timeout_seconds=_env_float(SWE_DYNAMIC_WORKFLOW_TIMEOUT_ENV, 1800.0),
            ),
        )
        result = runtime.run(
            script=script,
            task=task,
            artifacts_dir=artifacts_dir,
            args={"task": task},
            budget={},
            name="swebench_dynamic",
        )
        return assistant_message(
            result.output, sender=AGENT_NAME, target="user", kind="final"
        )

    return Agent(name=AGENT_NAME, generate=generate, role=AGENT_ROLE)


def extract_result(
    workspace: Any,
    instance: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(_base_extract_result(workspace, instance, context=context))
    workflow = read_workflow_artifacts(
        Path(workspace) / SWE_DYNAMIC_WORKFLOW_ARTIFACT_DIR
    )
    if workflow:
        result["dynamic_workflow"] = workflow
    return result


def _build_swebench_subagent(
    *,
    options: AgentCallOptions,
    provider: Provider,
    cwd: Path,
    request_extra: Mapping[str, Any] | None,
) -> Agent:
    spec = agent_spec()
    name = options.name or spec.name
    role = options.role or spec.role
    system_prompt = _subagent_system_prompt(spec.system_prompt, options)

    if spec.flavor == "bash_task":
        return make_bash_task_agent(
            provider,
            cwd=cwd,
            name=name,
            role=role,
            system_prompt=system_prompt,
            request_extra=request_extra,
        )
    if spec.flavor == "bash_skills":
        return make_llm_agent(
            name=name,
            provider=provider,
            role=role,
            tools=[make_bash_tool(cwd=cwd), make_read_tool(cwd=cwd)],
            system_prompt=system_prompt_with_skills(system_prompt, cwd=cwd),
            target="user",
            request_extra=request_extra,
            timeout_seconds=options.timeout_seconds,
        )
    if spec.flavor == "bash":
        return make_llm_agent(
            name=name,
            provider=provider,
            role=role,
            tools=[make_bash_tool(cwd=cwd)],
            system_prompt=system_prompt,
            target="user",
            request_extra=request_extra,
            timeout_seconds=options.timeout_seconds,
        )
    raise SystemExit(
        f"Unsupported {AGENT_FLAVOR_ENV} {spec.flavor!r}; "
        "expected 'bash', 'bash_task', or 'bash_skills'."
    )


def _subagent_system_prompt(base: str, options: AgentCallOptions) -> str:
    parts = [base]
    if options.system_prompt:
        parts.append(options.system_prompt)
    elif options.role:
        parts.append(f"Workflow role: {options.role}")
    return "\n\n".join(part for part in parts if part)


def _load_or_generate_script(
    *,
    provider: Provider,
    task: str,
    request_extra: Mapping[str, Any] | None,
) -> str:
    script_path = os.environ.get(SWE_DYNAMIC_WORKFLOW_SCRIPT_ENV, "").strip()
    if script_path:
        return Path(script_path).read_text(encoding="utf-8")
    generated = generate_workflow_script(
        provider=provider,
        task=_workflow_writer_task(task),
        request_extra=request_extra,
        fallback_script=DEFAULT_DYNAMIC_WORKFLOW_SCRIPT,
    )
    return generated.source


def _workflow_writer_task(task: str) -> str:
    return f"""Create a dynamic workflow JavaScript script for this SWE-bench task.

The JS script controls bash-capable subagents that all share one repository
worktree. Use phases and several focused subagents when useful, but keep file
edits sequential: at most one subagent should edit files. Parallel calls are
acceptable only for read-only inspection. The final answer should be a concise
summary; the harness collects the git diff separately. Prompt subagents to use
portable ``find``/``grep``/``sed`` commands because ``rg`` may not be installed
inside benchmark images.

SWE-bench task:
{task}
"""


def _task_text(visible: list[Message]) -> str:
    for message in visible:
        if message.kind == "task":
            return text_of(message.content)
    return text_of(visible[0].content) if visible else ""


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float, *, minimum: float = 1.0) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return max(minimum, float(raw))
    except ValueError:
        return default
