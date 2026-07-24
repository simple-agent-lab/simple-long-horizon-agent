"""OneMillion-Bench container half backed by dynamic JavaScript workflows.

The generic eval runner still drives one facade ``Agent``. That facade generates
or loads ``workflow.js``, executes it with ``simple_agent_lab.dynamic_workflows``,
and persists the script/journal/subagent trace summary for ``extract_result``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import simple_agent_lab.config as config
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

from .container import (  # noqa: F401  (suite surface re-exports)
    AGENT_NAME,
    AGENT_ROLE,
    RESPONSE_FILENAME,
    agent_spec,
    build_task,
    evaluate,
)
from .container import extract_result as _base_extract_result
from .workflow_container import ANSWER_SYSTEM_PROMPT

DYNAMIC_WORKFLOW_SCRIPT_ENV = config.OMB_DYNAMIC_WORKFLOW_SCRIPT.name
DYNAMIC_WORKFLOW_SOURCE_ENV = config.OMB_DYNAMIC_WORKFLOW_SOURCE.name
DYNAMIC_WORKFLOW_MAX_CONCURRENCY_ENV = config.OMB_DYNAMIC_MAX_CONCURRENCY.name
DYNAMIC_WORKFLOW_MAX_AGENTS_ENV = config.OMB_DYNAMIC_MAX_AGENTS.name
DYNAMIC_WORKFLOW_TIMEOUT_ENV = config.OMB_DYNAMIC_TIMEOUT.name
DYNAMIC_WORKFLOW_ARTIFACT_DIR = "dynamic_workflow"

DEFAULT_DYNAMIC_WORKFLOW_SCRIPT = r"""
phase("draft");
const draft = await agent(
  `Answer the benchmark question directly and completely.\n\n${args.task}`,
  { name: "draft", maxTurns: 1, cacheKey: "draft" }
);

phase("review");
const reviews = await parallel([
  () => agent(
    `Check this answer for missing details, ambiguity, or factual problems.\n\nQuestion:\n${args.task}\n\nAnswer:\n${draft.output}`,
    { name: "critic_accuracy", maxTurns: 1, cacheKey: "critic_accuracy" }
  ),
  () => agent(
    `Check whether this answer is well structured and directly answers the question.\n\nQuestion:\n${args.task}\n\nAnswer:\n${draft.output}`,
    { name: "critic_structure", maxTurns: 1, cacheKey: "critic_structure" }
  )
], { maxConcurrency: 2 });

phase("synthesize");
const final = await agent(
  `Produce the final answer. Use the draft, fix issues raised by the reviews, and return the full answer only.\n\nQuestion:\n${args.task}\n\nDraft:\n${draft.output}\n\nReviews:\n${reviews.map(r => r.output).join("\n\n")}`,
  { name: "synthesizer", maxTurns: 1, cacheKey: "synthesizer" }
);

return final.output;
"""


def build_agent(
    *,
    provider: Provider,
    cwd: Path,
    request_extra: Mapping[str, Any] | None = None,
) -> Agent:
    """Facade agent that answers one OMB case through a dynamic workflow."""

    response_path = Path(cwd) / RESPONSE_FILENAME
    artifacts_dir = Path(cwd) / DYNAMIC_WORKFLOW_ARTIFACT_DIR

    def build_subagent(
        options: AgentCallOptions, sub_provider: Provider, sub_cwd: Path
    ) -> Agent:
        del sub_cwd
        system_prompt = ANSWER_SYSTEM_PROMPT
        if options.system_prompt:
            system_prompt = f"{system_prompt}\n\n{options.system_prompt}"
        elif options.role:
            system_prompt = f"{system_prompt}\n\nRole: {options.role}"
        return make_llm_agent(
            name=options.name,
            provider=sub_provider,
            role=options.role or AGENT_ROLE,
            tools=(),
            system_prompt=system_prompt,
            target="user",
            request_extra=request_extra,
            timeout_seconds=options.timeout_seconds,
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
            allow_worktrees=False,
            max_turns_cap=config.WORKER_MAX_TURNS.get(),
        )
        runtime = DynamicWorkflowRuntime(
            runner=runner,
            options=WorkflowRuntimeOptions(
                max_concurrency=config.OMB_DYNAMIC_MAX_CONCURRENCY.get(),
                max_agents=config.OMB_DYNAMIC_MAX_AGENTS.get(),
                timeout_seconds=config.OMB_DYNAMIC_TIMEOUT.get(),
            ),
        )
        result = runtime.run(
            script=script,
            task=task,
            artifacts_dir=artifacts_dir,
            args={"task": task},
            budget={},
            name="onemillion_dynamic",
        )
        response_path.write_text(result.output, encoding="utf-8")
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
    workflow = read_workflow_artifacts(Path(workspace) / DYNAMIC_WORKFLOW_ARTIFACT_DIR)
    if workflow:
        result["dynamic_workflow"] = workflow
    return result


def _load_or_generate_script(
    *,
    provider: Provider,
    task: str,
    request_extra: Mapping[str, Any] | None,
) -> str:
    script_source = config.OMB_DYNAMIC_WORKFLOW_SOURCE.get()
    if script_source:
        return script_source
    script_path = config.OMB_DYNAMIC_WORKFLOW_SCRIPT.get()
    if script_path:
        return Path(script_path).read_text(encoding="utf-8")
    generated = generate_workflow_script(
        provider=provider,
        task=task,
        request_extra=request_extra,
        fallback_script=DEFAULT_DYNAMIC_WORKFLOW_SCRIPT,
    )
    return generated.source


def _task_text(visible: list[Message]) -> str:
    for message in visible:
        if message.kind == "task":
            return text_of(message.content)
    return text_of(visible[0].content) if visible else ""
