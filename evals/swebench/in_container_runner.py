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
from typing import Any, Callable, Mapping

from simple_agent_lab.agents.bash import make_bash_agent
from simple_agent_lab.agents.bash_task import (
    BASH_TASK_EXPLORER_ADDENDUM,
    make_bash_task_agent,
)
from simple_agent_lab.core import Agent
from simple_agent_lab.llm import Provider as LLMProvider
from simple_agent_lab.protocols import Event, MessageEvent, ModelRequestEvent
from simple_agent_lab.state import State
from simple_agent_lab.trajectory import (
    LiveTraceSession,
    ModelTurn,
    RunTrace,
    default_stderr_flush_error,
    json_safe,
    run_trace_from_state,
    trace_record,
    write_canonical_trace,
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
DEFAULT_RESPONSES_MAX_OUTPUT_TOKENS = 32768
OPENAI_MODEL_ENV = "OPENAI_MODEL"
OPENAI_AUTH_ENV = "OPENAI_AUTH_TOKEN"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
OPENAI_SESSION_ID_ENV = "OPENAI_SESSION_ID"
OPENAI_LOG_ID_ENV = "OPENAI_LOG_ID"
API_KIND_ENV = "API_KIND"
API_KIND_CHOICES = ("openai-chat", "openai-responses")
AGENT_FLAVOR_CHOICES = ("bash", "bash_task")
DEFAULT_AGENT_FLAVOR = "bash"
SWE_BENCH_PRO_DATASET_MARKER = "swe-bench_pro"

AGENT_NAME = "swebench_agent"
AGENT_ROLE = (
    "Work in the local SWE-bench repository. Use bash for inspection, edits, "
    "and focused tests, then return a concise final note."
)
AGENT_SYSTEM_PROMPT = (
    "You are a software engineer interacting with a SWE-bench instance "
    "container through the bash tool. Each bash call runs in a fresh shell "
    "rooted at the workspace, so include any cd or env setup in the command "
    "and use non-interactive flags (`-y`, `--no-pager`, avoid `vi`/`nano`). "
    "Independent read-only bash calls may run in parallel; never run parallel "
    "writes against the same file. Work from evidence: inspect, reproduce, "
    "edit, verify — make a fix that is general and consistent with the "
    "codebase. Keep command output focused. When the repository is patched, "
    "return a short final summary; the harness collects git diff separately."
)
# The bash_task flavor appends the package's canonical explorer addendum
# (`BASH_TASK_EXPLORER_ADDENDUM`) to AGENT_SYSTEM_PROMPT so the parent
# prompt stays a small superset of the plain bash-agent prompt and the
# wording can never drift from the shipped preset.
AGENT_DEFAULT_TEMPERATURE = 0.0
LLM_RETRY_MAX_ATTEMPTS = 20
LLM_RETRY_INITIAL_DELAY_SECONDS = 4.0
LLM_RETRY_MAX_DELAY_SECONDS = 60.0
# Incremental flush cadence: a fresh trajectory.jsonl snapshot every ~2s
# keeps the viewer responsive without rewriting multi-MB JSON per event.
INCREMENTAL_TRACE_FLUSH_INTERVAL_S = 2.0


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

    problem = _optional_context(
        instance.get("problem_statement")
        or instance.get("problem")
        or instance.get("description")
        or ""
    )
    requirements = _optional_context(instance.get("requirements"))
    interface = _optional_context(instance.get("interface"))
    lines = [
        "Solve this SWE-bench instance.",
        "",
        "## Environment",
        "- You are running inside the SWE-bench container.",
        f"- The bash tool runs locally in {workdir}.",
        "- A full Linux shell is available; install missing tools only if strictly needed.",
        "- Always pass non-interactive flags (`-y`, `--no-pager`); avoid editors that wait for input.",
        "",
        "## What to modify",
        "- MODIFY: regular source files in the repository.",
        "- DO NOT MODIFY: tests, reproduction scripts you create, configuration files",
        "  (pyproject.toml, setup.cfg, tox.ini, etc.) unless code evidence shows the fix",
        "  belongs there.",
        "- Keep temporary reproduction helpers out of the final diff (write them under",
        "  `/tmp/` or delete them before you stop).",
        "",
        "## Workflow",
        "1. Locate the relevant code. Prefer parallel read-only commands",
        "   (`grep -rn`, `find`, `sed -n 'A,Bp'`) over reading whole files.",
        "2. Reproduce the reported behavior with a tiny script when practical.",
        "3. Edit the smallest set of source files needed for a general fix.",
        "4. Re-run the reproduction. Then run a focused subset of existing tests",
        "   (single file or `-k pattern`) and explain if any are unavailable.",
        "5. Stop as soon as the fix is in place and verified. Do not keep exploring",
        "   once you can describe the change.",
        "",
        "## Final answer",
        "Return a short summary of the files you changed and how you verified the fix.",
        "Do NOT paste the patch — the harness collects `git diff` separately.",
        "",
        "## Problem statement",
        problem,
    ]
    if requirements:
        lines.extend(["", "requirements:", requirements])
    if interface:
        lines.extend(["", "interface:", interface])
    return "\n".join(lines)


def run_agent(
    *,
    instance: dict[str, Any],
    provider: LLMProvider,
    request_extra: Mapping[str, Any] | None = None,
    workdir: Path,
    max_turns: int,
    agent_flavor: str = DEFAULT_AGENT_FLAVOR,
    dataset_name: str = DEFAULT_DATASET,
    incremental_trace_path: str | Path | None = None,
    incremental_trace_meta_fn: Callable[[State], dict[str, Any]] | None = None,
    incremental_trace_id: str | None = None,
    incremental_trace_producer: str | None = None,
    incremental_flush_interval_s: float = INCREMENTAL_TRACE_FLUSH_INTERVAL_S,
) -> State:
    """Run the configured agent flavor inside the local container filesystem.

    ``agent_flavor`` selects between the plain bash agent (``"bash"``) and
    the bash + task delegation agent (``"bash_task"``). Both presets share
    the same ``cwd`` and SWE-bench task / system prompt so the only
    difference is the parent's tool surface.

    When ``incremental_trace_path`` is provided a background writer
    re-serializes the in-flight ``RunTrace`` to that path every
    ``incremental_flush_interval_s`` seconds so live viewers can tail the
    file.  The canonical end-of-run write is still performed by
    :func:`main` so the on-disk shape after the run matches the
    pre-incremental behavior exactly.
    """

    language = instance_language(instance)
    instance_id = str(instance["instance_id"])
    suite = suite_for_instance(dataset_name=dataset_name, instance_id=instance_id)
    baseline_commit = prepare_baseline_commit(workdir, language=language)
    task = task_from_instance(instance, workdir=str(workdir))
    agent = build_swebench_agent(
        flavor=agent_flavor,
        provider=provider,
        cwd=workdir,
        request_extra=request_extra,
    )
    agent.generate = with_llm_retry(agent.generate)
    state, events = agent.run(task, max_turns=max_turns)
    state.data.update(
        {
            "suite": suite,
            "instance": instance,
            "workspace": str(workdir),
        }
    )

    trace_id = incremental_trace_id or f"swebench.{instance.get('instance_id', '?')}"
    meta_fn = (
        (lambda: incremental_trace_meta_fn(state))
        if incremental_trace_meta_fn is not None
        else None
    )
    if incremental_trace_path is not None:
        with LiveTraceSession(
            incremental_trace_path,
            state,
            trace_id=trace_id,
            producer=incremental_trace_producer or f"suite:{suite}",
            meta_fn=meta_fn,
            flush_interval_s=incremental_flush_interval_s,
            on_error=default_stderr_flush_error,
        ) as session:
            # main() writes the canonical final record — stop skips final_flush.
            session.drain(events)
    else:
        for _ in events:
            pass
    state.data["model_patch"] = git_diff(
        workdir,
        language=language,
        commit=baseline_commit or instance_base_commit(instance),
    )
    return state


def build_swebench_agent(
    *,
    flavor: str,
    provider: LLMProvider,
    cwd: Path,
    request_extra: Mapping[str, Any] | None = None,
) -> Agent:
    """Build the SWE-bench agent for ``flavor`` with shared name/role/prompt.

    The ``bash_task`` flavor reuses ``AGENT_SYSTEM_PROMPT`` and only
    appends a small addendum describing the ``task`` tool, so behavior
    is comparable between flavors (only the parent's tool surface and
    one addendum paragraph differ). ``request_extra`` (eval-only header
    extras) is forwarded so every flavor's model calls carry it.
    """

    if flavor == "bash":
        return make_bash_agent(
            provider=provider,
            cwd=cwd,
            name=AGENT_NAME,
            role=AGENT_ROLE,
            system_prompt=AGENT_SYSTEM_PROMPT,
            request_extra=request_extra,
        )
    if flavor == "bash_task":
        return make_bash_task_agent(
            provider=provider,
            cwd=cwd,
            name=AGENT_NAME,
            role=AGENT_ROLE,
            system_prompt=(AGENT_SYSTEM_PROMPT + "\n\n" + BASH_TASK_EXPLORER_ADDENDUM),
            request_extra=request_extra,
        )
    raise SystemExit(
        f"Unsupported agent flavor {flavor!r}; expected one of: "
        + ", ".join(AGENT_FLAVOR_CHOICES)
    )


def build_openai_provider_from_env(api_kind: str = "openai-chat") -> LLMProvider:
    """Build the configured OpenAI provider used inside the eval container."""

    if api_kind not in API_KIND_CHOICES:
        raise SystemExit(
            f"Unsupported API_KIND {api_kind!r}; expected one of: "
            + ", ".join(API_KIND_CHOICES)
        )

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
        id=api_kind,
        api=api_kind,
        model=model,
        base_url=os.environ.get(OPENAI_BASE_URL_ENV) or None,
        api_key_env=OPENAI_AUTH_ENV,
        default_max_tokens=(
            DEFAULT_RESPONSES_MAX_OUTPUT_TOKENS
            if api_kind == "openai-responses"
            else None
        ),
        default_temperature=AGENT_DEFAULT_TEMPERATURE,
    )


def build_openai_request_extra_from_env(
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build request extras needed by eval-only OpenAI-compatible endpoints."""

    source = env if env is not None else os.environ
    session_id = source.get(OPENAI_SESSION_ID_ENV, "").strip()
    log_id = source.get(OPENAI_LOG_ID_ENV, "").strip()
    if not session_id and not log_id:
        return {}
    missing = [
        name
        for name, value in (
            (OPENAI_SESSION_ID_ENV, session_id),
            (OPENAI_LOG_ID_ENV, log_id),
        )
        if not value
    ]
    if missing:
        raise SystemExit(
            "Missing required env vars for OpenAI extra headers: " + ", ".join(missing)
        )
    return {
        "extra_headers": {
            "extra": json.dumps({"session_id": session_id}, separators=(",", ":")),
            "X-TT-logid": log_id,
        }
    }


def with_llm_retry(
    generate: Callable[..., Any],
    *,
    max_attempts: int = LLM_RETRY_MAX_ATTEMPTS,
    initial_delay_seconds: float = LLM_RETRY_INITIAL_DELAY_SECONDS,
    max_delay_seconds: float = LLM_RETRY_MAX_DELAY_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
    log_fn: Callable[[str], None] | None = None,
) -> Callable[..., Any]:
    """Retry containerized LLM calls when provider TPM/rate limits fire."""

    def wrapped(visible: list[Any]) -> Any:
        delay = initial_delay_seconds
        for attempt in range(1, max_attempts + 1):
            try:
                return generate(visible)
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


def prediction_record(
    instance_id: str,
    model_name: str,
    patch: str,
    *,
    dataset_name: str = DEFAULT_DATASET,
) -> dict[str, str]:
    if is_swebench_pro(dataset_name=dataset_name, instance_id=instance_id):
        return {
            "instance_id": instance_id,
            "prefix": model_name,
            "patch": patch,
        }
    return {
        "instance_id": instance_id,
        "model_name_or_path": model_name,
        "model_patch": patch,
    }


def is_swebench_pro(*, dataset_name: str = "", instance_id: str = "") -> bool:
    dataset = dataset_name.casefold()
    return SWE_BENCH_PRO_DATASET_MARKER in dataset or instance_id.startswith(
        "instance_"
    )


def suite_for_instance(*, dataset_name: str, instance_id: str) -> str:
    if is_swebench_pro(dataset_name=dataset_name, instance_id=instance_id):
        return "swebench_pro"
    return "swebench"


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
    suite = suite_for_instance(dataset_name=dataset_name, instance_id=instance_id)
    return run_trace_from_state(
        state=state,
        trace_id=trace_id,
        producer=f"suite:{suite}",
        meta={
            "suite": suite,
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
    parser.add_argument(
        "--api-kind",
        choices=API_KIND_CHOICES,
        default=os.environ.get(API_KIND_ENV, "openai-chat"),
        help="Adapter API kind to use when --provider openai.",
    )
    parser.add_argument("--workdir", default="/testbed")
    parser.add_argument("--max-turns", type=int, default=75)
    parser.add_argument(
        "--agent-flavor",
        choices=AGENT_FLAVOR_CHOICES,
        default=DEFAULT_AGENT_FLAVOR,
        help=(
            "Agent preset: 'bash' (parent has only the bash tool, current "
            "baseline) or 'bash_task' (parent has bash + task with an "
            "explorer sub-agent for context-isolating heavy reads)."
        ),
    )
    parser.add_argument("--traces", required=True)
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()

    provider = (
        build_openai_provider_from_env(args.api_kind)
        if args.provider == "openai"
        else LLMProvider(id="fake", api="fake", model="fake-model")
    )
    instance = load_instance(args.instance_json, args.instance_id)
    instance_id = str(instance["instance_id"])
    suite = suite_for_instance(dataset_name=args.dataset_name, instance_id=instance_id)

    def live_meta_fn(state: State) -> dict[str, Any]:
        return {
            "suite": suite,
            "dataset_name": args.dataset_name,
            "split": args.split,
            "instance_id": instance_id,
            "model_name_or_path": args.model_name,
            "patch_source": "containerized-diff",
            "patch_chars": len(str(state.data.get("model_patch") or "")),
            "workspace": state.data.get("workspace"),
            "in_progress": True,
        }

    state = run_agent(
        instance=instance,
        provider=provider,
        request_extra=(
            build_openai_request_extra_from_env() if args.provider == "openai" else {}
        ),
        workdir=Path(args.workdir),
        max_turns=args.max_turns,
        agent_flavor=args.agent_flavor,
        dataset_name=args.dataset_name,
        incremental_trace_path=args.traces,
        incremental_trace_meta_fn=live_meta_fn,
        incremental_trace_id=f"swebench.{instance_id}",
        incremental_trace_producer=f"suite:{suite}",
    )
    state.data["agent_flavor"] = args.agent_flavor
    patch = str(state.data.get("model_patch") or "")
    trace = trace_from_state(
        state=state,
        instance=instance,
        dataset_name=args.dataset_name,
        split=args.split,
        model_name=args.model_name,
        patch_source="containerized-diff",
    )
    prediction = prediction_record(
        str(instance["instance_id"]),
        args.model_name,
        patch,
        dataset_name=args.dataset_name,
    )
    write_canonical_trace(args.traces, record=trace_record(trace))
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


def _optional_context(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.casefold() in {"none", "null", "nan"}:
        return ""
    if text.startswith('"') and text.endswith('"'):
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(decoded, str):
            return decoded.strip()
    return text


if __name__ == "__main__":
    main()
