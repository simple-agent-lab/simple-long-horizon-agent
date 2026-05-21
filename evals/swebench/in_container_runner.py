"""Run Simple Agent Lab inside a SWE-bench instance container.

This file is copied into the container by `containerized_agent.py`. The
container installs the `simple-agent-lab` wheel first, so this runner can keep
SWE-bench-specific orchestration in `evals/` while using the installed package
for the actual agent/runtime code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

from simple_agent_lab.agents.bash import make_bash_agent
from simple_agent_lab.llm import Provider as LLMProvider
from simple_agent_lab.protocols import Event, MessageEvent, ModelRequestEvent
from simple_agent_lab.state import State
from simple_agent_lab.trajectory import (
    ModelTurn,
    RunTrace,
    json_safe,
    trace_record,
    write_jsonl,
)

try:
    from evals.swebench.patch_extract import (
        git_diff as extract_git_diff,
        instance_base_commit,
        instance_language,
        prepare_baseline_commit,
    )
except ModuleNotFoundError:  # copied beside this file inside the container
    from patch_extract import (  # type: ignore[import-not-found]
        git_diff as extract_git_diff,
        instance_base_commit,
        instance_language,
        prepare_baseline_commit,
    )

DEFAULT_DATASET = "princeton-nlp/SWE-bench_Verified"
DEFAULT_SPLIT = "test"
OPENAI_MODEL_ENV = "OPENAI_MODEL"
OPENAI_AUTH_ENV = "OPENAI_AUTH_TOKEN"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"

AGENT_NAME = "swebench_agent"
AGENT_ROLE = (
    "Work in the local SWE-bench repository. Use bash for inspection, edits, "
    "and focused tests, then return a concise final note."
)
AGENT_SYSTEM_PROMPT = (
    "You are a tiny SWE-bench repair agent running inside the instance "
    "container. The repository is local. Use bash to inspect files, edit code, "
    "and run focused tests. Work from evidence: inspect relevant files before "
    "editing, reproduce the reported behavior when practical, and make a fix "
    "that is general and consistent with the codebase. Each bash tool call runs "
    "in a fresh shell, so include any needed cd or environment setup in the "
    "command. Use non-interactive command flags and avoid editors that require "
    "user input. Keep command output focused. When the repository is patched, "
    "return a short final summary; the harness will collect git diff separately."
)
LLM_RETRY_MAX_ATTEMPTS = 20
LLM_RETRY_INITIAL_DELAY_SECONDS = 4.0
LLM_RETRY_MAX_DELAY_SECONDS = 60.0


def load_instance(path: str | Path, instance_id: str | None) -> dict[str, Any]:
    """Load one instance record from JSON or JSONL."""

    records = _load_instance_records(Path(path))
    if not records:
        raise SystemExit(f"No instance records found in {path}")
    if instance_id is None:
        return dict(records[0])
    for record in records:
        if str(record.get("instance_id")) == instance_id:
            return dict(record)
    raise SystemExit(f"Instance {instance_id!r} not found in {path}")


def task_from_instance(instance: dict[str, Any], *, workdir: str) -> str:
    """Build the model-visible task for the in-container agent."""

    problem = str(
        instance.get("problem_statement")
        or instance.get("problem")
        or instance.get("description")
        or ""
    )
    lines = [
        "Solve this SWE-bench instance.",
    ]
    lines.extend(
        [
            "",
            "Workspace instructions:",
            "- You are running inside the SWE-bench container.",
            f"- The bash tool runs locally in {workdir}.",
            "- Modify files in that repository to solve the issue in a way that is general and consistent with the codebase.",
            "- Do not modify tests, reproduction files, or configuration files unless the issue explicitly requires it or code evidence shows they are part of the fix.",
            "- Keep temporary reproduction helpers out of the final diff.",
            "- Final answer: summarize changed files and verification; do not include a patch because git diff is collected separately.",
            "",
            "Recommended workflow:",
            "1. Read relevant files before editing.",
            "2. Create or run a small reproduction from the problem statement when practical.",
            "3. Edit the smallest set of source files needed to resolve the issue.",
            "4. Re-run the reproduction or focused failing check.",
            "5. Run relevant existing tests or explain why they are unavailable.",
            "",
            "problem_statement:",
            problem,
        ]
    )
    return "\n".join(lines)


def run_agent(
    *,
    instance: dict[str, Any],
    provider: LLMProvider,
    workdir: Path,
    max_turns: int,
) -> State:
    """Run the bash-use agent inside the local container filesystem."""

    language = instance_language(instance)
    baseline_commit = prepare_baseline_commit(workdir, language=language)
    task = task_from_instance(instance, workdir=str(workdir))
    agent = make_bash_agent(
        provider=provider,
        cwd=workdir,
        name=AGENT_NAME,
        role=AGENT_ROLE,
        system_prompt=AGENT_SYSTEM_PROMPT,
    )
    agent.step = with_llm_retry(agent.step)
    state, events = agent.run(task, max_turns=max_turns)
    for _ in events:
        pass
    state.data.update(
        {
            "suite": "swebench",
            "instance": instance,
            "workspace": str(workdir),
            "model_patch": git_diff(
                workdir,
                language=language,
                commit=baseline_commit or instance_base_commit(instance),
            ),
        }
    )
    return state


def build_openai_provider_from_env() -> LLMProvider:
    """Build the generic OpenAI Chat provider used inside the eval container."""

    model = os.environ.get(OPENAI_MODEL_ENV, "").strip()
    token = os.environ.get(OPENAI_AUTH_ENV, "").strip()
    if not model or not token:
        missing = [
            name
            for name, value in (
                (OPENAI_MODEL_ENV, model),
                (OPENAI_AUTH_ENV, token),
            )
            if not value
        ]
        raise SystemExit(
            "Missing required env vars for --provider openai: " + ", ".join(missing)
        )
    return LLMProvider(
        id="openai-chat",
        api="openai-chat",
        model=model,
        base_url=os.environ.get(OPENAI_BASE_URL_ENV) or None,
        api_key_env=OPENAI_AUTH_ENV,
    )


def with_llm_retry(
    step: Callable[..., Any],
    *,
    max_attempts: int = LLM_RETRY_MAX_ATTEMPTS,
    initial_delay_seconds: float = LLM_RETRY_INITIAL_DELAY_SECONDS,
    max_delay_seconds: float = LLM_RETRY_MAX_DELAY_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
    log_fn: Callable[[str], None] | None = None,
) -> Callable[..., Any]:
    """Retry containerized LLM calls when provider TPM/rate limits fire."""

    def wrapped(agent: Any, visible: list[Any], state: Any) -> Any:
        delay = initial_delay_seconds
        for attempt in range(1, max_attempts + 1):
            try:
                return step(agent, visible, state)
            except Exception as exc:
                if attempt >= max_attempts or not is_retryable_llm_error(exc):
                    raise
                _log_llm_retry(
                    (
                        "LLM call hit retryable rate-limit/TPM error "
                        f"on attempt {attempt}/{max_attempts}; "
                        f"retrying in {delay:g}s: {type(exc).__name__}: {exc}"
                    ),
                    log_fn=log_fn,
                )
                sleep_fn(delay)
                delay = min(delay * 2, max_delay_seconds)
        raise RuntimeError("unreachable LLM retry state")

    return wrapped


def is_retryable_llm_error(exc: BaseException) -> bool:
    """Return True for transient provider throttling errors seen in eval runs."""

    text = f"{type(exc).__name__}: {exc}".casefold()
    return any(
        marker in text
        for marker in (
            "tpm",
            "tokens per minute",
            "rate limit",
            "rate_limit",
            "too many requests",
            "429",
        )
    )


def _log_llm_retry(
    message: str,
    *,
    log_fn: Callable[[str], None] | None,
) -> None:
    if log_fn is not None:
        log_fn(message)
        return
    print(message, file=sys.stderr, flush=True)


def git_diff(
    repo_dir: Path,
    *,
    language: str = "python",
    commit: str | None = None,
) -> str:
    """Return the current repo diff in SWE-bench prediction format."""

    return extract_git_diff(repo_dir, language=language, commit=commit)


def prediction_record(instance_id: str, model_name: str, patch: str) -> dict[str, str]:
    return {
        "instance_id": instance_id,
        "model_name_or_path": model_name,
        "model_patch": patch,
    }


def trace_from_state(
    *,
    state: State,
    instance: dict[str, Any],
    dataset_name: str,
    split: str,
    model_name: str,
    patch_source: str,
) -> RunTrace:
    instance_id = str(instance["instance_id"])
    trace_id = f"swebench.{instance_id}"
    return RunTrace(
        trace_id=trace_id,
        producer="suite:swebench",
        task=state.task,
        messages=json_safe(state.messages),
        events=json_safe(state.events),
        model_turns=model_turns_from_events(trace_id, state.events),
        meta={
            "suite": "swebench",
            "dataset_name": dataset_name,
            "split": split,
            "instance_id": instance_id,
            "model_name_or_path": model_name,
            "patch_source": patch_source,
            "patch_chars": len(str(state.data.get("model_patch") or "")),
            "workspace": state.data.get("workspace"),
        },
    )


def model_turns_from_events(trace_id: str, events: list[Event]) -> list[ModelTurn]:
    turns: list[ModelTurn] = []
    pending: dict[str, Any] | None = None
    model_call_index = 0

    for event in events:
        if isinstance(event, ModelRequestEvent):
            model_call_index += 1
            pending = {
                "agent": str(event.agent or ""),
                "input_messages": event.llm_payload,
                "tools": event.tools,
                "request_event_index": event.index,
                "meta": {
                    "visible_count": event.visible_count,
                    "model_message_count": event.llm_message_count,
                },
            }
            continue

        if not isinstance(event, MessageEvent) or pending is None:
            continue
        message = event.message
        if message.role != "assistant":
            continue
        agent = pending["agent"] or message.sender
        if message.sender != agent:
            continue
        turns.append(
            ModelTurn(
                step_id=f"{trace_id}.model{model_call_index}",
                agent=agent,
                input_messages=json_safe(pending["input_messages"]),
                output_message=json_safe(message),
                tools=json_safe(pending["tools"]),
                meta={
                    **pending["meta"],
                    "request_event_index": pending["request_event_index"],
                    "message_event_index": event.index,
                },
            )
        )
        pending = None

    return turns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Simple Agent Lab from inside a SWE-bench container."
    )
    parser.add_argument("--instance-json", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--model-name", default="simple-agent-lab-containerized")
    parser.add_argument("--provider", choices=["fake", "openai"], default="openai")
    parser.add_argument("--workdir", default="/testbed")
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()

    provider = (
        build_openai_provider_from_env()
        if args.provider == "openai"
        else LLMProvider(id="fake", api="fake", model="fake-model")
    )
    instance = load_instance(args.instance_json, args.instance_id)
    state = run_agent(
        instance=instance,
        provider=provider,
        workdir=Path(args.workdir),
        max_turns=args.max_turns,
    )
    patch = str(state.data.get("model_patch") or "")
    trace = trace_from_state(
        state=state,
        instance=instance,
        dataset_name=args.dataset_name,
        split=args.split,
        model_name=args.model_name,
        patch_source="containerized-diff",
    )
    prediction = prediction_record(str(instance["instance_id"]), args.model_name, patch)
    write_jsonl(args.traces, [trace_record(trace)])
    write_jsonl(args.predictions, [prediction])
    print(f"wrote 1 SWE-bench trajectory to {args.traces}")
    print(f"wrote 1 SWE-bench prediction to {args.predictions}")


def _load_instance_records(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("[") or raw.startswith("{"):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            if raw.startswith("["):
                raise SystemExit(f"Expected valid JSON list in {path}")
        else:
            return _records_from_json(parsed, path)
    records: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            records.append(dict(json.loads(line)))
    return records


def _records_from_json(parsed: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(parsed, list):
        return [dict(item) for item in parsed]
    if isinstance(parsed, dict):
        if "instances" in parsed:
            instances = parsed["instances"]
            if not isinstance(instances, list):
                raise SystemExit(f"Expected instances to be a JSON list in {path}")
            return [dict(item) for item in instances]
        return [dict(parsed)]
    raise SystemExit(f"Expected JSON object, JSON list, or JSONL records in {path}")


if __name__ == "__main__":
    main()
