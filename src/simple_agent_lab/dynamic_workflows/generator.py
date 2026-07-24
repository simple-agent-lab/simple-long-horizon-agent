"""Generate JavaScript workflow scripts with an ordinary agent."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from simple_agent_lab.llm import Provider
from simple_agent_lab.llm_agent import make_llm_agent
from simple_agent_lab.workflow.base import final_output


@dataclass(frozen=True)
class GeneratedWorkflow:
    source: str
    prompt: str
    raw_output: str
    generated_by: str


WORKFLOW_WRITER_SYSTEM_PROMPT = """You write executable JavaScript workflow scripts.

The script will run in a restricted workflow runtime with these globals:
- args: structured workflow input.
- budget: budget hints.
- phase(name): mark progress.
- log(message): write progress.
- await agent(prompt, opts): run a subagent and return { output, status, ... }.
- await parallel([() => agent(...), ...], { maxConcurrency }): run tasks and wait.
- await pipeline(items, stage1, stage2, ...): transform items through async stages.
- await workflow(name, args): call a saved workflow by name.

The script itself cannot read files, run shell commands, load modules, or access
environment variables. Subagents do actual model/tool work. Return JavaScript
only: no prose outside the code.
"""


def generate_workflow_script(
    *,
    provider: Provider,
    task: str,
    request_extra: Mapping[str, Any] | None = None,
    fallback_script: str | None = None,
    max_turns: int = 1,
    timeout_seconds: float | None = 600.0,
) -> GeneratedWorkflow:
    """Ask a workflow-writer agent to produce executable JavaScript.

    Tests and fake-provider smokes may pass ``fallback_script`` because the
    deterministic fake model is not a real script writer.
    """

    prompt = _writer_prompt(task)
    if provider.api == "fake" and fallback_script:
        return GeneratedWorkflow(
            source=fallback_script.strip() + "\n",
            prompt=prompt,
            raw_output=fallback_script,
            generated_by="fallback_for_fake_provider",
        )
    writer = make_llm_agent(
        name="workflow_writer",
        provider=provider,
        role="Write a task-specific dynamic workflow script.",
        tools=(),
        system_prompt=WORKFLOW_WRITER_SYSTEM_PROMPT,
        target="user",
        request_extra=request_extra,
        timeout_seconds=timeout_seconds,
    )
    state, events = writer.run(prompt, max_turns=max_turns)
    for _ in events:
        pass
    raw = final_output(state, writer.name)
    source = extract_javascript(raw)
    if not source.strip():
        raise RuntimeError("workflow writer did not produce JavaScript")
    return GeneratedWorkflow(
        source=source.strip() + "\n",
        prompt=prompt,
        raw_output=raw,
        generated_by=f"agent:{writer.name}",
    )


def extract_javascript(text: str) -> str:
    """Extract JavaScript from a model response."""

    fenced = re.search(
        r"```(?:javascript|js)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL
    )
    if fenced:
        return fenced.group(1).strip()
    stripped = text.strip()
    if (
        stripped.startswith("return ")
        or "await agent(" in stripped
        or "phase(" in stripped
    ):
        return stripped
    return ""


def _writer_prompt(task: str) -> str:
    return f"""Create a dynamic workflow JavaScript script for this task.

Use several focused subagents when it improves quality. Keep intermediate
results in JavaScript variables, use phases, and return the final answer from
the script.

Task:
{task}
"""
