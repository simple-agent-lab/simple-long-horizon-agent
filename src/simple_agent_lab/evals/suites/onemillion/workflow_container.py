"""OneMillion-Bench multi-agent *generation* (the workflow flavors).

A generation-only helper for the single OneMillion suite: ``container.build_agent``
delegates here when ``AGENT_FLAVOR`` selects a workflow (``reflection`` /
``planner_executor`` / ``parallel`` / ``chain`` / ``routing`` / ``pdr``) instead
of the ``single`` tool-free turn. The task seam, answer collection, and rubric
judge all stay in ``container``; only generation differs.

How a workflow plugs into the generic runner
--------------------------------------------
The in-container runner drives exactly **one** ``Agent``
(``agent.run(task)`` → the core ReAct loop) and reads the answer back from
``model_response.txt``. A workflow is *several* agent runs, not one agent, so
``build_workflow_agent`` wraps it behind a **facade ``Agent``** whose ``generate``
runs the whole workflow to completion and returns its final answer as a single
``final`` message (also persisting it to ``model_response.txt`` and the per-step
breakdown to ``workflow_steps.json``). The facade returns on its first turn, so
the outer loop stops immediately; all the real multi-agent work happens *inside*
that one ``generate`` call, each sub-agent driven by the same ``core.run`` loop.

The flavor name arrives from ``container`` (resolved from ``AGENT_FLAVOR``);
``OMB_REFLECTION_ROUNDS`` / ``OMB_PARALLEL_WORKERS`` / ``OMB_PDR_*`` /
``OMB_TIMEOUT`` tune the parameterized workflows.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import simple_agent_lab.config as config
from simple_agent_lab.core import Agent
from simple_agent_lab.llm import Provider
from simple_agent_lab.llm_agent import make_llm_agent
from simple_agent_lab.messages import Message, assistant_message, text_of
from simple_agent_lab.trace import event_stream, run_trace_from_state
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

# Generation-only helper module for OneMillion: `container.build_agent` delegates
# here for a workflow flavor (reflection / parallel / …). The task seam, answer
# collection, and judge live in `container`; this file owns only the multi-agent
# *generation*. Reuse the shared identity + filenames from `container`.
from .container import (
    AGENT_NAME,
    AGENT_ROLE,
    RESPONSE_FILENAME,
    WORKFLOW_STEPS_FILENAME,
)

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


def make_workflow_runner(
    provider: Provider,
    *,
    request_extra: Mapping[str, Any] | None = None,
    workflow: str | None = None,
) -> WorkflowRunner:
    """Build the `task -> WorkflowResult` runner for the selected workflow."""

    name = (workflow or "single").strip().lower()
    timeout = config.OMB_TIMEOUT.get()

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
        rounds = config.OMB_REFLECTION_ROUNDS.get()
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
        n = config.OMB_PARALLEL_WORKERS.get()
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
        rounds = config.OMB_PDR_ROUNDS.get()
        width = config.OMB_PDR_WIDTH.get()
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
        f"Unsupported OneMillion AGENT_FLAVOR={name!r}; "
        f"expected one of {WORKFLOW_CHOICES}."
    )


def build_workflow_agent(
    *,
    provider: Provider,
    cwd: Path,
    request_extra: Mapping[str, Any] | None = None,
    flavor: str,
) -> Agent:
    """A facade `Agent` that answers by running the workflow named by ``flavor``.

    Called by ``container.build_agent`` for a workflow flavor. The outer loop
    calls ``generate`` once; it runs the whole workflow on the case's question
    and returns the final answer as a single ``final`` message, persisting it to
    ``model_response.txt`` (the seam ``extract_result`` reads) and the per-step
    breakdown to ``workflow_steps.json`` for inspection.
    """

    run_workflow = make_workflow_runner(
        provider, request_extra=request_extra, workflow=flavor
    )
    response_path = Path(cwd) / RESPONSE_FILENAME
    steps_path = Path(cwd) / WORKFLOW_STEPS_FILENAME

    def generate(visible: list[Message]) -> Message:
        task = _task_text(visible)
        result = run_workflow(task)
        text = result.output or ""
        if text.strip():
            response_path.write_text(text, encoding="utf-8")
        _write_steps(steps_path, result, flavor)
        return assistant_message(text, sender=AGENT_NAME, target="user", kind="final")

    return Agent(name=AGENT_NAME, generate=generate, role=AGENT_ROLE)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
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
    header, lines, raw_pool = event_stream(trace)
    return {
        "name": step.name,
        "role": step.role,
        "output": step.output,
        # v5 stream embedded inline for this debug dump: header then event lines,
        # with the provider raw pool alongside.
        "trace": [header, *lines],
        "trace_raw": raw_pool,
    }


def _write_steps(path: Path, result: WorkflowResult, workflow_name: str) -> None:
    """Persist each step's output + full sub-agent trace for inspection."""

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
