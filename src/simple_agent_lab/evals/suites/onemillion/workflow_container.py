"""OneMillion-Bench container half that answers via a multi-agent *workflow*.

A drop-in alternative to the sibling ``container.py``: the task seam
(``build_task``), the answer-collection seam (``extract_result``), and the
rubric ``evaluate`` judge are reused **verbatim**; only *generation* changes —
instead of a single tool-free model turn, the answer is produced by one of the
``simple_agent_lab.workflow`` orchestrations (reflection, planner/executor,
parallel, chain, routing) so they can be compared on the same benchmark.

How a workflow plugs into the generic runner
--------------------------------------------
The in-container runner drives exactly **one** ``Agent``
(``agent.run(task)`` → the core ReAct loop) and reads the answer back from
``model_response.txt``. A workflow is *several* agent runs, not one agent, so it
is wrapped behind a **facade ``Agent``** whose ``generate`` runs the whole
workflow to completion and returns its final answer as a single ``final``
message (also persisting it to ``model_response.txt``). The facade returns on
its first turn, so the outer loop stops immediately; all the real multi-agent
work happens *inside* that one ``generate`` call, each sub-agent driven by the
same ``core.run`` loop.

Selection is by the ``OMB_WORKFLOW`` env var (``build_agent`` only receives
``provider`` / ``cwd`` / ``request_extra``, so the choice rides in the
environment). ``OMB_REFLECTION_ROUNDS`` and ``OMB_PARALLEL_WORKERS`` tune the
two parameterized workflows.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

from simple_agent_lab.core import Agent
from simple_agent_lab.llm import LLMRequest, Provider, complete_with_retry, llm_message
from simple_agent_lab.llm_agent import make_llm_agent
from simple_agent_lab.messages import Message, assistant_message, text_of
from simple_agent_lab.trace import run_trace_from_state, trace_record
from simple_agent_lab.workflow import (
    Route,
    StepResult,
    WorkflowResult,
    make_critic_agent,
    make_distiller_agent,
    make_planner_agent,
    make_router_agent,
    run_agent,
    run_chain,
    run_parallel,
    run_pdr,
    run_planner_executor,
    run_reflection,
    run_routing,
)

# Reuse the single-turn container's task / collection seam. `extract_result` and
# `evaluate` are wrapped (to add the per-step traces and the raw judge prompt /
# response); the rest are re-exported so the runner finds them on this module.
from .container import (  # noqa: F401  (build_task is part of the suite surface)
    AGENT_NAME,
    AGENT_ROLE,
    RESPONSE_FILENAME,
    agent_spec,
    build_task,
    judge_provider_from_env,
)
from .container import extract_result as _base_extract_result
from .grading import (
    build_grading_prompt,
    convert_scores,
    parse_grading_response,
    score_summary,
)

WORKFLOW_ENV = "OMB_WORKFLOW"
REFLECTION_ROUNDS_ENV = "OMB_REFLECTION_ROUNDS"
PARALLEL_WORKERS_ENV = "OMB_PARALLEL_WORKERS"
PDR_ROUNDS_ENV = "OMB_PDR_ROUNDS"
PDR_WIDTH_ENV = "OMB_PDR_WIDTH"
# Per-request timeout for every sub-agent. The default LLM request timeout is
# 60s, far too short for slow high-reasoning models on a long case, so the
# workflow agents use a generous default that callers can override.
TIMEOUT_ENV = "OMB_TIMEOUT"
DEFAULT_TIMEOUT_S = 600.0
JUDGE_TIMEOUT_S = 600.0
DEFAULT_WORKFLOW = "single"
WORKFLOW_STEPS_FILENAME = "workflow_steps.json"

# English answer prompt mirroring OMB's generation intent (direct, complete,
# never asking the user back). Used by every role that emits the *final* answer.
ANSWER_SYSTEM_PROMPT = (
    "You are a professional question-answering assistant. Answer the question "
    "directly and completely. Do NOT ask the user for more information or pose "
    "questions back. Using only the information given, provide an accurate, "
    "detailed, well-structured answer."
)

# A workflow runner takes the case's question text and returns a WorkflowResult.
WorkflowRunner = Callable[[str], WorkflowResult]

WORKFLOW_CHOICES = (
    "single",
    "reflection",
    "planner_executor",
    "parallel",
    "chain",
    "routing",
    "pdr",
)


# --------------------------------------------------------------------------- #
# Answer-producing agent: an ordinary LLM agent carrying the OMB answer prompt.
# Every role that emits the *final* answer (generator, executor, workers,
# aggregator, chain stages, routing specialists, the single baseline) uses this
# so the graded text always matches OMB's "direct, complete" intent; only the
# meta roles (critic, planner, router) carry their own workflow prompts.
# --------------------------------------------------------------------------- #
def _answer_agent(
    provider: Provider,
    name: str,
    request_extra: Mapping[str, Any] | None,
    *,
    extra_prompt: str = "",
    timeout_seconds: float | None = None,
) -> Agent:
    system_prompt = ANSWER_SYSTEM_PROMPT
    if extra_prompt:
        system_prompt = f"{ANSWER_SYSTEM_PROMPT}\n\n{extra_prompt}"
    return make_llm_agent(
        name=name,
        provider=provider,
        role=AGENT_ROLE,
        tools=(),
        system_prompt=system_prompt,
        target="user",
        request_extra=request_extra,
        timeout_seconds=timeout_seconds,
    )


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


def make_workflow_runner(
    provider: Provider,
    *,
    request_extra: Mapping[str, Any] | None = None,
    workflow: str | None = None,
) -> WorkflowRunner:
    """Build the `task -> WorkflowResult` runner for the selected workflow."""

    name = (
        (workflow or os.environ.get(WORKFLOW_ENV) or DEFAULT_WORKFLOW).strip().lower()
    )
    timeout = _env_float(TIMEOUT_ENV, DEFAULT_TIMEOUT_S)

    if name == "single":
        agent = _answer_agent(
            provider, AGENT_NAME, request_extra, timeout_seconds=timeout
        )

        def run_single(task: str) -> WorkflowResult:
            step = run_agent(agent, task, max_turns=1)
            return WorkflowResult(output=step.output, steps=[step])

        return run_single

    if name == "reflection":
        generator = _answer_agent(
            provider,
            "generator",
            request_extra,
            timeout_seconds=timeout,
            extra_prompt=(
                "If you receive critic feedback, address every point and return "
                "the COMPLETE revised answer (not just the changes)."
            ),
        )
        critic = make_critic_agent(
            provider, request_extra=request_extra, timeout_seconds=timeout
        )
        rounds = _env_int(REFLECTION_ROUNDS_ENV, 2)
        return lambda task: run_reflection(generator, critic, task, max_rounds=rounds)

    if name == "planner_executor":
        planner = make_planner_agent(
            provider, request_extra=request_extra, timeout_seconds=timeout
        )
        executor = _answer_agent(
            provider,
            "executor",
            request_extra,
            timeout_seconds=timeout,
            extra_prompt=(
                "Follow the given plan closely and produce a complete, accurate "
                "final answer."
            ),
        )
        return lambda task: run_planner_executor(planner, executor, task)

    if name == "parallel":
        n = _env_int(PARALLEL_WORKERS_ENV, 3)
        workers = [
            _answer_agent(
                provider, f"worker_{i}", request_extra, timeout_seconds=timeout
            )
            for i in range(n)
        ]
        aggregator = _answer_agent(
            provider,
            "aggregator",
            request_extra,
            timeout_seconds=timeout,
            extra_prompt=(
                "You will be given several candidate answers to the same "
                "question. Synthesize them into one complete, accurate final "
                "answer: keep what is correct, fix errors, fill gaps — do not "
                "merely concatenate."
            ),
        )
        return lambda task: run_parallel(workers, task, aggregator=aggregator)

    if name == "chain":
        drafter = _answer_agent(
            provider, "drafter", request_extra, timeout_seconds=timeout
        )
        refiner = _answer_agent(
            provider,
            "refiner",
            request_extra,
            timeout_seconds=timeout,
            extra_prompt=(
                "Improve and complete the previous answer; return the full "
                "revised answer."
            ),
        )
        return lambda task: run_chain([drafter, refiner], task)

    if name == "routing":
        # Routing is the least natural fit for single-shot Q&A (there is no
        # routable type the agent sees), but it is wired here for completeness:
        # a router picks between a reasoning-heavy and a knowledge-heavy answerer.
        reasoning = _answer_agent(
            provider,
            "reasoning",
            request_extra,
            timeout_seconds=timeout,
            extra_prompt=(
                "This is a reasoning/quantitative/analytical question. Show clear "
                "reasoning steps, then give a complete answer."
            ),
        )
        knowledge = _answer_agent(
            provider,
            "knowledge",
            request_extra,
            timeout_seconds=timeout,
            extra_prompt=(
                "This is a factual/conceptual question. Give an accurate, "
                "complete answer directly."
            ),
        )
        routes = [
            Route("reasoning", reasoning, "reasoning / calculation / analysis"),
            Route("knowledge", knowledge, "factual / conceptual / definitional"),
        ]
        router = make_router_agent(
            provider, routes, request_extra=request_extra, timeout_seconds=timeout
        )
        return lambda task: run_routing(router, routes, task, default="knowledge")

    if name == "pdr":
        # Parallel-Distill-Refine: each round runs `width` attempts, distills
        # them into a findings brief, and conditions the next round on it.
        rounds = _env_int(PDR_ROUNDS_ENV, 2)
        width = _env_int(PDR_WIDTH_ENV, 3)
        worker = _answer_agent(
            provider, "attempt", request_extra, timeout_seconds=timeout
        )
        distiller = make_distiller_agent(
            provider, request_extra=request_extra, timeout_seconds=timeout
        )
        finalizer = _answer_agent(
            provider,
            "finalizer",
            request_extra,
            timeout_seconds=timeout,
            extra_prompt=(
                "Using the prior findings, write the complete, accurate final answer."
            ),
        )
        return lambda task: run_pdr(
            worker,
            distiller,
            task,
            rounds=rounds,
            width=width,
            finalizer=finalizer,
            worker_max_turns=1,
            finalizer_max_turns=1,
        )

    raise SystemExit(
        f"Unsupported {WORKFLOW_ENV}={name!r}; expected one of {WORKFLOW_CHOICES}."
    )


def build_agent(
    *,
    provider: Provider,
    cwd: Path,
    request_extra: Mapping[str, Any] | None = None,
) -> Agent:
    """A facade `Agent` that answers by running the selected workflow.

    The outer loop calls ``generate`` once; it runs the whole workflow on the
    case's question and returns the final answer as a single ``final`` message,
    persisting it to ``model_response.txt`` (the seam ``extract_result`` reads)
    and the per-step breakdown to ``workflow_steps.json`` for inspection.
    """

    run_workflow = make_workflow_runner(provider, request_extra=request_extra)
    response_path = Path(cwd) / RESPONSE_FILENAME
    steps_path = Path(cwd) / WORKFLOW_STEPS_FILENAME

    def generate(visible: list[Message]) -> Message:
        task = _task_text(visible)
        result = run_workflow(task)
        text = result.output or ""
        if text.strip():
            response_path.write_text(text, encoding="utf-8")
        _write_steps(steps_path, result)
        return assistant_message(text, sender=AGENT_NAME, target="user", kind="final")

    return Agent(name=AGENT_NAME, generate=generate, role=AGENT_ROLE)


def extract_result(
    workspace: Any,
    instance: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect the answer (base seam) plus the per-step workflow breakdown.

    The facade ``build_agent`` writes ``workflow_steps.json`` into the same
    workspace; reading it here folds the breakdown into ``result.json`` so it
    survives after the (ephemeral) workdir is cleaned up.
    """

    result = dict(_base_extract_result(workspace, instance, context=context))
    steps = _read_steps(Path(workspace))
    if steps is not None:
        result["workflow"] = steps
    return result


def evaluate(
    workspace: Any,
    instance: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Grade with the rubric judge, additionally recording the raw judge I/O.

    Mirrors the base ``container.evaluate`` (same judge, prompt, and scoring)
    but also returns ``judge_prompt`` and ``judge_raw_response`` so the judge
    model's full input/output is auditable, not just the parsed verdict.
    """

    gold = dict((context or {}).get("eval") or {})
    rubrics = list(gold.get("rubrics") or [])
    if not rubrics:
        return {"scored": False, "status": "no_rubrics"}

    prompt = str(
        gold.get("prompt") or instance.get("prompt") or instance.get("Prompt") or ""
    )
    human_scores = dict(gold.get("human_scores") or {})
    model_response = _read_response(Path(workspace))

    judge = judge_provider_from_env()
    grading_prompt = build_grading_prompt(prompt, model_response, rubrics, human_scores)
    judge_text = _judge_complete(judge, grading_prompt)

    raw_results = parse_grading_response(judge_text, rubrics)
    final_scores = convert_scores(raw_results, rubrics)
    summary = score_summary(final_scores, rubrics)

    return {
        "scored": True,
        "judge_model": judge.model,
        "model_response_chars": len(model_response),
        "total_score": summary["total_score"],
        "max_score": summary["max_score"],
        "min_score": summary["min_score"],
        "score": summary["accuracy"],
        "rubric_scores": {str(k): v for k, v in final_scores.items()},
        "judge_verdicts": {str(k): v for k, v in raw_results.items()},
        "judge_prompt": grading_prompt,
        "judge_raw_response": judge_text,
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _read_response(workspace: Path) -> str:
    try:
        return (workspace / RESPONSE_FILENAME).read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return ""


def _judge_complete(provider: Provider, grading_prompt: str) -> str:
    """One blocking judge call (no tools); transient throttling is retried."""

    request = LLMRequest(
        provider=provider,
        messages=[llm_message("user", grading_prompt)],
        tools=[],
        system_prompt=None,
        temperature=0.0,
        timeout_seconds=JUDGE_TIMEOUT_S,
    )
    return complete_with_retry(request).text


def _task_text(visible: list[Message]) -> str:
    for message in visible:
        if message.kind == "task":
            return text_of(message.content)
    return text_of(visible[0].content) if visible else ""


def _step_record(step: StepResult, index: int, workflow_name: str) -> dict[str, Any]:
    """One workflow step: its output text plus the sub-agent's full trace.

    The sub-agent ran in its own `State`, so its real model turns (messages,
    per-call usage, events) are not in the outer trajectory; serializing each
    step's `State` here is what records them. For ``single`` this is the real
    generation trace; for multi-agent workflows it is every draft/critique/
    worker call.
    """

    trace = run_trace_from_state(
        state=step.state,
        trace_id=f"{workflow_name}.{index}.{step.role or step.name}",
        producer=f"workflow:{workflow_name}",
        meta={"role": step.role, "name": step.name, "step": index},
    )
    return {
        "name": step.name,
        "role": step.role,
        "output": step.output,
        "trace": trace_record(trace),
    }


def _write_steps(path: Path, result: WorkflowResult) -> None:
    """Persist each step's output + full sub-agent trace for inspection."""

    workflow_name = os.environ.get(WORKFLOW_ENV) or DEFAULT_WORKFLOW
    try:
        path.write_text(
            json.dumps(
                {
                    "workflow": workflow_name,
                    "steps": [
                        _step_record(step, index, workflow_name)
                        for index, step in enumerate(result.steps)
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        pass


def _read_steps(workspace: Path) -> dict[str, Any] | None:
    try:
        raw = (workspace / WORKFLOW_STEPS_FILENAME).read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None
