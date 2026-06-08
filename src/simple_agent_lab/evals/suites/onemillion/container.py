"""OneMillion-Bench container half (ADR 0017): the functions a suite supplies.

OneMillion-Bench is a *rubric-graded Q&A* benchmark, not an agent-in-a-repo
benchmark, so its mapping onto the generic framework (`SwebenchSuite` is the
reference) is:

- **Generation** is a single, tool-free model turn: the case's prompt is the
  task, and ``build_agent`` returns a plain LLM agent (no bash/tools) whose only
  job is to answer. Because ``extract_result`` cannot see the agent's messages
  (only the workspace), the agent's ``generate`` is wrapped to persist the final
  answer to ``model_response.txt`` in the workspace — the analog of SWE-bench
  writing a ``git diff`` to the filesystem.
- **Scoring** is the in-environment ``evaluate`` hook (ADR 0020): the host stages
  the case's weighted rubrics via ``eval_inputs`` (gold the agent must not see),
  and this hook calls a *judge* model to grade the response against them,
  reproducing the upstream ``omb`` rubric scoring (see ``grading``). The verdict
  is merged into ``result.json``.

The judge is configured from the environment so the same code grades in-process
(`LocalProcessBackend`) or in a container: ``JUDGE_MODEL`` / ``JUDGE_AUTH_TOKEN``
/ ``JUDGE_BASE_URL`` / ``JUDGE_API_KIND``, each falling back to the generator's
``OPENAI_*`` value. Secrets stay in env (never in the staged ``eval.json``).

It imports only the standard library and the installed wheel (the ``llm`` layer,
``llm_agent``, ``messages``, ``evals.protocols``, and the sibling ``grading``
module), so it runs in any eval environment with no copied files.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from simple_agent_lab.core import Agent
from simple_agent_lab.evals.protocols import AgentSpec
from simple_agent_lab.llm import (
    ApiKind,
    LLMRequest,
    Provider,
    complete_with_retry,
    llm_message,
)
from simple_agent_lab.llm_agent import make_llm_agent
from simple_agent_lab.messages import text_of

from .grading import (
    build_grading_prompt,
    convert_scores,
    parse_grading_response,
    score_summary,
)

# Where the tool-free generator persists its answer so ``extract_result`` (which
# only sees the workspace, never the agent state) can collect it.
RESPONSE_FILENAME = "model_response.txt"

AGENT_NAME = "onemillion_agent"
AGENT_ROLE = "Answer the professional question directly and completely."
# Mirrors the upstream omb generation system prompt: direct, complete answers,
# never asking the user for more information.
AGENT_SYSTEM_PROMPT = (
    "你是一个专业的问题解答助手。你必须直接、完整地回答问题，"
    "禁止向用户提问或要求用户提供更多信息。请基于问题中已有的信息，"
    "给出准确、详细、结构清晰的答案。"
)

# Judge (grader) provider env contract — falls back to the generator's OPENAI_*.
JUDGE_MODEL_ENV = "JUDGE_MODEL"
JUDGE_AUTH_ENV = "JUDGE_AUTH_TOKEN"
JUDGE_BASE_URL_ENV = "JUDGE_BASE_URL"
JUDGE_API_KIND_ENV = "JUDGE_API_KIND"
OPENAI_MODEL_ENV = "OPENAI_MODEL"
OPENAI_AUTH_ENV = "OPENAI_AUTH_TOKEN"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
API_KIND_CHOICES = ("openai-chat", "openai-responses", "anthropic-messages")
DEFAULT_RESPONSES_MAX_OUTPUT_TOKENS = 32768
JUDGE_TIMEOUT_S = 600.0


# --------------------------------------------------------------------------- #
# Generation (tool-free agent + answer capture)
# --------------------------------------------------------------------------- #
def agent_spec() -> AgentSpec:
    """Advertised agent config (the framework default ``build_agent`` is overridden)."""

    return AgentSpec(
        name=AGENT_NAME, role=AGENT_ROLE, system_prompt=AGENT_SYSTEM_PROMPT
    )


def build_agent(
    *,
    provider: Provider,
    cwd: Path,
    request_extra: Mapping[str, Any] | None = None,
) -> Agent:
    """A tool-free LLM agent that answers, persisting its answer to the workspace.

    No tools: the model receives the question and replies once (``end_turn`` ends
    the loop). ``generate`` is wrapped so the latest assistant text is written to
    ``model_response.txt`` under ``cwd`` — the seam ``extract_result`` reads,
    since the container half never sees the agent's state directly.
    """

    agent = make_llm_agent(
        name=AGENT_NAME,
        provider=provider,
        role=AGENT_ROLE,
        tools=(),
        system_prompt=AGENT_SYSTEM_PROMPT,
        target="user",
        request_extra=request_extra,
    )
    response_path = Path(cwd) / RESPONSE_FILENAME
    inner_generate = agent.generate

    def capturing_generate(visible: list[Any]) -> Any:
        message = inner_generate(visible)
        text = text_of(message.content)
        if text.strip():
            response_path.write_text(text, encoding="utf-8")
        return message

    agent.generate = capturing_generate
    return agent


def build_task(instance: Mapping[str, Any], *, workdir: str) -> str:
    """The model-visible task: the case's (optional system) prompt + question."""

    del workdir  # the generator answers in chat; no filesystem context needed
    prompt = _first_str(instance, "prompt", "Prompt", "question")
    system_prompt = _first_str(instance, "system_prompt", "System_Prompt")
    parts: list[str] = []
    if system_prompt.strip():
        parts.append(system_prompt.strip())
        parts.append("")
    parts.append(prompt.strip())
    return "\n".join(parts).strip()


def extract_result(
    workspace: Any,
    instance: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect the generated answer as the run's product."""

    del context
    response = _read_response(Path(workspace))
    return {
        "model_response": response,
        "case_id": instance.get("case_id"),
        "instance_id": instance.get("instance_id"),
    }


# --------------------------------------------------------------------------- #
# In-environment scoring (the rubric judge)
# --------------------------------------------------------------------------- #
def evaluate(
    workspace: Any,
    instance: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Grade the answer against the host-staged rubrics with a judge model.

    The host stages ``{prompt, rubrics, human_scores}`` under EVAL_KEY; the
    generic runner threads it in as ``context["eval"]`` and only calls this hook
    when that gold is present. Reproduces the upstream ``omb`` rubric scoring and
    returns a self-contained verdict merged into ``result.json``.
    """

    gold = dict((context or {}).get("eval") or {})
    rubrics = list(gold.get("rubrics") or [])
    if not rubrics:
        return {"scored": False, "status": "no_rubrics"}

    prompt = str(gold.get("prompt") or _first_str(instance, "prompt", "Prompt"))
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
    }


def judge_provider_from_env(env: Mapping[str, str] | None = None) -> Provider:
    """Build the judge `Provider` from JUDGE_* env, falling back to OPENAI_*."""

    source = env if env is not None else os.environ
    model = (source.get(JUDGE_MODEL_ENV) or source.get(OPENAI_MODEL_ENV) or "").strip()
    auth_env = JUDGE_AUTH_ENV if source.get(JUDGE_AUTH_ENV) else OPENAI_AUTH_ENV
    token = (source.get(auth_env) or "").strip()
    base_url = (
        source.get(JUDGE_BASE_URL_ENV) or source.get(OPENAI_BASE_URL_ENV) or ""
    ).strip() or None
    api_kind = (source.get(JUDGE_API_KIND_ENV) or "openai-chat").strip()

    if api_kind not in API_KIND_CHOICES:
        raise SystemExit(
            f"Unsupported {JUDGE_API_KIND_ENV} {api_kind!r}: {API_KIND_CHOICES}"
        )
    missing = [
        name
        for name, value in ((JUDGE_MODEL_ENV, model), (auth_env, token))
        if not value
    ]
    if missing:
        raise SystemExit(
            "Missing env for the OneMillion-Bench judge: " + ", ".join(missing)
        )
    return Provider(
        id=api_kind,
        api=cast(ApiKind, api_kind),
        model=model,
        base_url=base_url,
        api_key_env=auth_env,
        default_max_tokens=(
            DEFAULT_RESPONSES_MAX_OUTPUT_TOKENS
            if api_kind == "openai-responses"
            else None
        ),
        default_temperature=0.0,
    )


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


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _read_response(workspace: Path) -> str:
    try:
        return (workspace / RESPONSE_FILENAME).read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return ""


def _first_str(instance: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = instance.get(key)
        if value:
            return str(value)
    return ""
