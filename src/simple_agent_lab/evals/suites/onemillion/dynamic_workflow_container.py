"""OneMillion-Bench container half backed by dynamic JavaScript workflows.

The generic eval runner still drives one facade ``Agent``. That facade generates
or loads ``workflow.js``, executes it with ``simple_agent_lab.dynamic_workflows``,
and persists the script/journal/subagent trace summary for ``extract_result``.
"""

from __future__ import annotations

import json
import os
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

DYNAMIC_WORKFLOW_SCRIPT_ENV = "OMB_DYNAMIC_WORKFLOW_SCRIPT"
DYNAMIC_WORKFLOW_MAX_CONCURRENCY_ENV = "OMB_DYNAMIC_MAX_CONCURRENCY"
DYNAMIC_WORKFLOW_MAX_AGENTS_ENV = "OMB_DYNAMIC_MAX_AGENTS"
DYNAMIC_WORKFLOW_TIMEOUT_ENV = "OMB_DYNAMIC_TIMEOUT"
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
        )
        runtime = DynamicWorkflowRuntime(
            runner=runner,
            options=WorkflowRuntimeOptions(
                max_concurrency=_env_int(DYNAMIC_WORKFLOW_MAX_CONCURRENCY_ENV, 16),
                max_agents=_env_int(DYNAMIC_WORKFLOW_MAX_AGENTS_ENV, 1000),
                timeout_seconds=_env_float(DYNAMIC_WORKFLOW_TIMEOUT_ENV, 1800.0),
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
    workflow = _read_workflow_artifacts(Path(workspace) / DYNAMIC_WORKFLOW_ARTIFACT_DIR)
    if workflow:
        result["dynamic_workflow"] = workflow
    return result


def _load_or_generate_script(
    *,
    provider: Provider,
    task: str,
    request_extra: Mapping[str, Any] | None,
) -> str:
    script_path = os.environ.get(DYNAMIC_WORKFLOW_SCRIPT_ENV, "").strip()
    if script_path:
        return Path(script_path).read_text(encoding="utf-8")
    generated = generate_workflow_script(
        provider=provider,
        task=task,
        request_extra=request_extra,
        fallback_script=DEFAULT_DYNAMIC_WORKFLOW_SCRIPT,
    )
    return generated.source


def _read_workflow_artifacts(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {}
    result = _read_json(root / "workflow_result.json")
    journal = _read_jsonl(root / "workflow_journal.jsonl")
    script = _read_text(root / "workflow.js")
    traces = _read_subagent_traces(root)
    agent_calls = list((result or {}).get("agent_calls") or [])
    for call in agent_calls:
        if not isinstance(call, dict):
            continue
        trace = traces.get(str(call.get("call_id") or ""))
        if trace:
            call["trace"] = trace
    return {
        "workflow_js": script,
        "result": result,
        "journal": journal,
        "agent_calls": agent_calls,
        "subagent_traces": traces,
    }


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return records
    for line in lines:
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def _read_subagent_traces(root: Path) -> dict[str, dict[str, Any]]:
    traces: dict[str, dict[str, Any]] = {}
    subagents = root / "subagents"
    if not subagents.exists():
        return traces
    for trace_path in subagents.glob("*/trajectory.jsonl"):
        records = _read_jsonl(trace_path)
        if records:
            traces[trace_path.parent.name] = records[0]
    return traces


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
