"""ProgramBench container half backed by agent-written JavaScript workflows.

The generic eval runner still drives one facade ``Agent``. The facade generates
or loads ``workflow.js`` and lets each JavaScript ``agent()`` call run a normal
ProgramBench worker against the shared workspace. Every worker reuses the
suite's per-command network-isolation prefix, and result extraction still packs
the workspace for the official ProgramBench evaluator.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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
from simple_agent_lab.messages import Message, assistant_message, text_of

from .container import (  # noqa: F401  (suite surface re-exports)
    AGENT_NAME,
    AGENT_ROLE,
    AGENT_SYSTEM_PROMPT,
    REQUIRE_ISOLATION_ENV,
    build_task,
    make_programbench_agent,
    network_isolation_prefix,
    prepare,
)
from .container import extract_result as _base_extract_result

PROGRAMBENCH_DYNAMIC_WORKFLOW_SCRIPT_ENV = "PROGRAMBENCH_DYNAMIC_WORKFLOW_SCRIPT"
PROGRAMBENCH_DYNAMIC_WORKFLOW_SOURCE_ENV = "PROGRAMBENCH_DYNAMIC_WORKFLOW_SOURCE"
PROGRAMBENCH_DYNAMIC_WORKFLOW_MAX_CONCURRENCY_ENV = (
    "PROGRAMBENCH_DYNAMIC_MAX_CONCURRENCY"
)
PROGRAMBENCH_DYNAMIC_WORKFLOW_MAX_AGENTS_ENV = "PROGRAMBENCH_DYNAMIC_MAX_AGENTS"
PROGRAMBENCH_DYNAMIC_WORKFLOW_MAX_TURNS_ENV = "PROGRAMBENCH_DYNAMIC_MAX_TURNS"
PROGRAMBENCH_DYNAMIC_WORKFLOW_TIMEOUT_ENV = "PROGRAMBENCH_DYNAMIC_TIMEOUT"
PROGRAMBENCH_DYNAMIC_WORKFLOW_NODE_ENV = "PROGRAMBENCH_DYNAMIC_NODE_BINARY"
PROGRAMBENCH_DYNAMIC_WORKFLOW_ARTIFACT_DIR = "dynamic_workflow"

DEFAULT_DYNAMIC_WORKFLOW_SCRIPT = r"""
phase("investigate");
const investigation = await agent(
  `Read every bundled documentation file and probe ./executable extensively. Do not edit files yet. Infer the CLI, input/output behavior, edge cases, and a concrete implementation plan. Never use the network, decompile the binary, trace it, or obtain original source.\n\nTask:\n${args.task}`,
  { name: "investigator", role: "Read-only ProgramBench behavior investigator.", maxTurns: 40, cacheKey: "investigate" }
);

phase("implement");
const implementation = await agent(
  `Implement an original replacement from behavioral observations. Create source plus ./compile.sh, build it, and compare your executable against the provided behavior. Do not obtain or reuse original source or binaries.\n\nTask:\n${args.task}\n\nInvestigation:\n${investigation.output}`,
  { name: "implementer", role: "ProgramBench reimplementation engineer.", maxTurns: 120, cacheKey: "implement" }
);

phase("verify");
const verification = await agent(
  `Review the current workspace, run ./compile.sh, and probe important behavior against the observations. Fix focused correctness gaps you find. Keep the solution original and obey all reverse-engineering restrictions.\n\nTask:\n${args.task}\n\nInvestigation:\n${investigation.output}\n\nImplementation notes:\n${implementation.output}`,
  { name: "verifier", role: "ProgramBench verifier and focused fixer.", maxTurns: 60, cacheKey: "verify" }
);

return [
  "Dynamic ProgramBench workflow completed.",
  "",
  "Investigation:", investigation.output,
  "",
  "Implementation:", implementation.output,
  "",
  "Verification:", verification.output
].join("\n");
"""


def build_agent(
    *,
    provider: Provider,
    cwd: Path,
    request_extra: Mapping[str, Any] | None = None,
) -> Agent:
    """Facade agent that solves one ProgramBench case through a JS workflow."""

    artifacts_dir = _artifact_dir(Path(cwd))
    exec_prefix = network_isolation_prefix()

    def build_subagent(
        options: AgentCallOptions, sub_provider: Provider, sub_cwd: Path
    ) -> Agent:
        unsupported_tools = set(options.tools) - {"bash"}
        if unsupported_tools:
            raise ValueError(
                "ProgramBench dynamic workers only support the isolated bash tool; "
                f"unsupported: {sorted(unsupported_tools)}"
            )
        return make_programbench_agent(
            provider=sub_provider,
            cwd=sub_cwd,
            name=options.name or AGENT_NAME,
            role=options.role or AGENT_ROLE,
            system_prompt=_subagent_system_prompt(options),
            request_extra=request_extra,
            timeout_seconds=options.timeout_seconds,
            exec_prefix=exec_prefix,
        )

    def generate(visible: list[Message]) -> Message:
        task = _task_text(visible)
        timeout_seconds = _env_float(PROGRAMBENCH_DYNAMIC_WORKFLOW_TIMEOUT_ENV, 21600.0)
        deadline = time.monotonic() + timeout_seconds
        script = _load_or_generate_script(
            provider=provider,
            task=task,
            request_extra=request_extra,
            timeout_seconds=min(_remaining_seconds(deadline), 600.0),
        )
        runner = SimpleAgentCallRunner(
            provider=provider,
            cwd=cwd,
            request_extra=request_extra,
            build_agent=build_subagent,
            allow_worktrees=False,
            max_turns_cap=_env_int(PROGRAMBENCH_DYNAMIC_WORKFLOW_MAX_TURNS_ENV, 1000),
        )
        runtime = DynamicWorkflowRuntime(
            runner=runner,
            options=WorkflowRuntimeOptions(
                # All workers share the scored workspace. Serialize by default
                # so generated workflows cannot race file writes.
                max_concurrency=_env_int(
                    PROGRAMBENCH_DYNAMIC_WORKFLOW_MAX_CONCURRENCY_ENV, 1
                ),
                max_agents=_env_int(PROGRAMBENCH_DYNAMIC_WORKFLOW_MAX_AGENTS_ENV, 12),
                timeout_seconds=_remaining_seconds(deadline),
                node_binary=(
                    os.environ.get(PROGRAMBENCH_DYNAMIC_WORKFLOW_NODE_ENV) or "node"
                ),
                # workflow.js is model-generated and Node's v24 permission
                # model does not restrict networking. Put the orchestration
                # process in the same sealed namespace as worker commands.
                process_prefix=exec_prefix,
            ),
        )
        result = runtime.run(
            script=script,
            task=task,
            artifacts_dir=artifacts_dir,
            args={"task": task},
            budget={},
            name="programbench_dynamic",
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
        _artifact_dir(Path(workspace)), embed_traces_in_calls=False
    )
    if workflow:
        result["dynamic_workflow"] = workflow
    return result


def _subagent_system_prompt(options: AgentCallOptions) -> str:
    parts = [AGENT_SYSTEM_PROMPT]
    parts.append(
        "All workers operate on the one scored workspace. Do not create, switch, "
        "or edit a git worktree elsewhere; make every submission change here."
    )
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
    timeout_seconds: float,
) -> str:
    source = os.environ.get(PROGRAMBENCH_DYNAMIC_WORKFLOW_SOURCE_ENV, "").strip()
    if source:
        return source + "\n"
    script_path = os.environ.get(PROGRAMBENCH_DYNAMIC_WORKFLOW_SCRIPT_ENV, "").strip()
    if script_path:
        return Path(script_path).read_text(encoding="utf-8")
    generated = generate_workflow_script(
        provider=provider,
        task=_workflow_writer_task(task),
        request_extra=request_extra,
        fallback_script=DEFAULT_DYNAMIC_WORKFLOW_SCRIPT,
        timeout_seconds=timeout_seconds,
    )
    return generated.source


def _workflow_writer_task(task: str) -> str:
    return f"""Create a dynamic workflow JavaScript script for this ProgramBench task.

The JavaScript orchestrates bash-capable subagents that share one scored
workspace. Use sequential phases for investigation, implementation, and
verification. Do not request worktrees. Keep edits sequential; parallel calls
are safe only for read-only behavioral investigation. Every subagent command is
network-isolated and must obey the reverse-engineering rules: no source lookup,
no binary reuse, no decompilation, and no tracing of the provided executable.
The final workspace must contain original source and an executable compile.sh.
Return a concise summary; the harness packages the workspace separately.

ProgramBench task:
{task}
"""


def _task_text(visible: list[Message]) -> str:
    for message in visible:
        if message.kind == "task":
            return text_of(message.content)
    return text_of(visible[0].content) if visible else ""


def _artifact_dir(workspace: Path) -> Path:
    """Keep orchestration internals outside the scored/git-tracked workspace."""

    store_root = (os.environ.get("SAL_STORE_ROOT") or "").strip()
    if store_root:
        return Path(store_root) / "out" / PROGRAMBENCH_DYNAMIC_WORKFLOW_ARTIFACT_DIR
    return (
        workspace.parent
        / f".{workspace.name}-simple-agent-lab"
        / PROGRAMBENCH_DYNAMIC_WORKFLOW_ARTIFACT_DIR
    )


def _remaining_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("ProgramBench dynamic workflow timed out")
    return remaining


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
