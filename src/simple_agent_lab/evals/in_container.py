"""Generic in-container runner (ADR 0017).

This module is what runs *inside* the eval container. It imports the suite's
container half by dotted path (`Suite.container_module`), builds the agent,
drives the loop with rate-limit retry, pushes the trajectory to a `TraceSink`,
then calls the suite's `extract_result` and writes the raw product to
``result.json``. The host (`run_suite_instance`) shapes the scorer-facing
``prediction.jsonl`` from that product via `Suite.prediction_record`.

Everything here is suite-agnostic: a new benchmark supplies only
``build_task`` / ``extract_result`` (and an optional ``agent_spec``) in its
container module. Nothing about SWE-bench, datasets, or patches lives here.

The agent runtime comes from the installed ``simple-agent-lab`` wheel, so this
runner stays small and the same code path works for every suite.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, cast

from ..agents.bash import make_bash_agent
from ..agents.bash_task import make_bash_task_agent
from ..core import Agent
from ..llm import ApiKind, Provider
from ..state import State
from ..trajectory import run_trace_from_state, trace_record
from .protocols import RESULT_FILE, AgentSpec, TraceSink
from .trace_sink import FileTraceSink, HttpTraceSink

__all__ = [
    "RESULT_FILE",
    "build_agent",
    "main",
    "provider_from_env",
    "run_in_container",
    "with_llm_retry",
]

TRACE_FLUSH_INTERVAL_S = 2.0

LLM_RETRY_MAX_ATTEMPTS = 20
LLM_RETRY_INITIAL_DELAY_S = 4.0
LLM_RETRY_MAX_DELAY_S = 60.0

# Env contract for the OpenAI-compatible provider, shared by every suite.
OPENAI_MODEL_ENV = "OPENAI_MODEL"
OPENAI_AUTH_ENV = "OPENAI_AUTH_TOKEN"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
OPENAI_SESSION_ID_ENV = "OPENAI_SESSION_ID"
OPENAI_LOG_ID_ENV = "OPENAI_LOG_ID"
API_KIND_ENV = "API_KIND"
API_KIND_CHOICES = ("openai-chat", "openai-responses")
DEFAULT_RESPONSES_MAX_OUTPUT_TOKENS = 32768
TRACE_PUSH_URL_ENV = "TRACE_PUSH_URL"


# --------------------------------------------------------------------------- #
# Agent construction (the only suite-tunable part, via AgentSpec)
# --------------------------------------------------------------------------- #
def build_agent(
    *,
    spec: AgentSpec,
    provider: Provider,
    cwd: Path,
    request_extra: Mapping[str, Any] | None = None,
) -> Agent:
    """Build the agent for `spec.flavor` with the suite's prompt/role/name."""

    if spec.flavor == "bash":
        return make_bash_agent(
            provider=provider,
            cwd=cwd,
            name=spec.name,
            role=spec.role,
            system_prompt=spec.system_prompt,
            request_extra=request_extra,
        )
    if spec.flavor == "bash_task":
        return make_bash_task_agent(
            provider=provider,
            cwd=cwd,
            name=spec.name,
            role=spec.role,
            system_prompt=spec.system_prompt,
            request_extra=request_extra,
        )
    raise SystemExit(
        f"Unsupported agent flavor {spec.flavor!r}; expected 'bash' or 'bash_task'."
    )


def _agent_spec(module: Any) -> AgentSpec:
    factory = getattr(module, "agent_spec", None)
    return factory() if callable(factory) else AgentSpec()


# --------------------------------------------------------------------------- #
# Rate-limit retry (generic; provider-throttling is suite-independent)
# --------------------------------------------------------------------------- #
def is_retryable_llm_error(exc: BaseException) -> bool:
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


def with_llm_retry(
    generate: Callable[..., Any],
    *,
    max_attempts: int = LLM_RETRY_MAX_ATTEMPTS,
    initial_delay_s: float = LLM_RETRY_INITIAL_DELAY_S,
    max_delay_s: float = LLM_RETRY_MAX_DELAY_S,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Callable[..., Any]:
    """Retry transient provider throttling so long runs survive TPM limits."""

    def wrapped(visible: list[Any]) -> Any:
        delay = initial_delay_s
        for attempt in range(1, max_attempts + 1):
            try:
                return generate(visible)
            except Exception as exc:
                if attempt >= max_attempts or not is_retryable_llm_error(exc):
                    raise
                print(
                    f"LLM retryable error {attempt}/{max_attempts}; "
                    f"retry in {delay:g}s: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                sleep_fn(delay)
                delay = min(delay * 2, max_delay_s)
        raise RuntimeError("unreachable LLM retry state")

    return wrapped


# --------------------------------------------------------------------------- #
# Provider from env (generic OpenAI-compatible + fake)
# --------------------------------------------------------------------------- #
def provider_from_env(*, kind: str, api_kind: str = "openai-chat") -> Provider:
    if kind == "fake":
        return Provider(id="fake", api="fake", model="fake-model")
    if api_kind not in API_KIND_CHOICES:
        raise SystemExit(f"Unsupported API_KIND {api_kind!r}: {API_KIND_CHOICES}")
    model = os.environ.get(OPENAI_MODEL_ENV, "").strip()
    token = os.environ.get(OPENAI_AUTH_ENV, "").strip()
    missing = [
        n for n, v in ((OPENAI_MODEL_ENV, model), (OPENAI_AUTH_ENV, token)) if not v
    ]
    if missing:
        raise SystemExit("Missing env for openai provider: " + ", ".join(missing))
    return Provider(
        id=api_kind,
        api=cast(ApiKind, api_kind),
        model=model,
        base_url=os.environ.get(OPENAI_BASE_URL_ENV) or None,
        api_key_env=OPENAI_AUTH_ENV,
        default_max_tokens=(
            DEFAULT_RESPONSES_MAX_OUTPUT_TOKENS
            if api_kind == "openai-responses"
            else None
        ),
        default_temperature=0.0,
    )


def request_extra_from_env(*, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    source = env if env is not None else os.environ
    session_id = source.get(OPENAI_SESSION_ID_ENV, "").strip()
    log_id = source.get(OPENAI_LOG_ID_ENV, "").strip()
    if not session_id and not log_id:
        return {}
    return {
        "extra_headers": {
            "extra": json.dumps({"session_id": session_id}, separators=(",", ":")),
            "X-TT-logid": log_id,
        }
    }


def trace_sink_from_env(traces_path: str | Path) -> TraceSink:
    """A file sink locally; an HTTP push sink when TRACE_PUSH_URL is set (cloud)."""

    url = os.environ.get(TRACE_PUSH_URL_ENV, "").strip()
    if url:
        return HttpTraceSink(url)
    return FileTraceSink(traces_path)


# --------------------------------------------------------------------------- #
# The generic run
# --------------------------------------------------------------------------- #
def run_in_container(
    *,
    instance: Mapping[str, Any],
    container_module: str,
    provider: Provider,
    workdir: Path,
    max_turns: int,
    trace_sink: TraceSink,
    trace_id: str,
    producer: str,
    suite_name: str,
    request_extra: Mapping[str, Any] | None = None,
    flush_interval_s: float = TRACE_FLUSH_INTERVAL_S,
) -> tuple[dict[str, Any], State]:
    """Drive one instance: build task → run agent → extract result.

    Returns the raw `extract_result` product and the finished `State`. Pushes
    incremental trajectory records to `trace_sink` during the run and a final
    record once the result is known.
    """

    module = importlib.import_module(container_module)
    spec = _agent_spec(module)

    # Optional pre-run setup (checkout, snapshot a baseline, install ignore
    # rules). Its returned dict is threaded into extract_result as `context`
    # for suites whose result depends on pre-run state.
    context: dict[str, Any] = {}
    prepare = getattr(module, "prepare", None)
    if callable(prepare):
        context = dict(prepare(workdir, instance) or {})

    task = module.build_task(instance, workdir=str(workdir))

    agent = build_agent(
        spec=spec, provider=provider, cwd=workdir, request_extra=request_extra
    )
    agent.generate = with_llm_retry(agent.generate)
    state, events = agent.run(task, max_turns=max_turns)

    instance_id = str(instance.get("instance_id", "?"))

    def record(*, in_progress: bool) -> dict[str, Any]:
        trace = run_trace_from_state(
            state=state,
            trace_id=trace_id,
            producer=producer,
            meta={
                "suite": suite_name,
                "instance_id": instance_id,
                "in_progress": in_progress,
                "result_keys": sorted(state.data.get("result", {})),
            },
        )
        return trace_record(trace)

    last = 0.0
    for _ in events:
        now = time.monotonic()
        if now - last >= flush_interval_s:
            trace_sink.emit(record(in_progress=True))
            last = now

    extract = module.extract_result
    extract_kwargs = (
        {"context": context}
        if "context" in inspect.signature(extract).parameters
        else {}
    )
    result = dict(extract(workdir, instance, **extract_kwargs))
    state.data["result"] = result
    trace_sink.emit(record(in_progress=False))
    trace_sink.close()
    return result, state


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generic in-container eval runner.")
    parser.add_argument("--instance-json", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--container-module", required=True)
    parser.add_argument("--suite-name", default="suite")
    parser.add_argument("--workdir", default="/testbed")
    parser.add_argument("--max-turns", type=int, default=75)
    parser.add_argument("--provider", choices=["fake", "openai"], default="openai")
    parser.add_argument(
        "--api-kind",
        choices=API_KIND_CHOICES,
        default=os.environ.get(API_KIND_ENV, "openai-chat"),
    )
    parser.add_argument("--traces", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args(argv)

    instance = _load_instance(Path(args.instance_json), args.instance_id)
    provider = provider_from_env(kind=args.provider, api_kind=args.api_kind)
    trace_id = f"{args.suite_name}.{args.instance_id}"
    result, _state = run_in_container(
        instance=instance,
        container_module=args.container_module,
        provider=provider,
        workdir=Path(args.workdir),
        max_turns=args.max_turns,
        trace_sink=trace_sink_from_env(args.traces),
        trace_id=trace_id,
        producer=f"suite:{args.suite_name}",
        suite_name=args.suite_name,
        request_extra=(request_extra_from_env() if args.provider == "openai" else {}),
    )
    Path(args.results).write_text(
        json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote result to {args.results} and trajectory to {args.traces}")


def _load_instance(path: Path, instance_id: str | None) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8").strip()
    records: list[dict[str, Any]] = []
    if raw.startswith("{"):
        records = [json.loads(raw)]
    elif raw.startswith("["):
        records = list(json.loads(raw))
    else:
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not records:
        raise SystemExit(f"No instance records in {path}")
    if instance_id is None:
        return dict(records[0])
    for record in records:
        if str(record.get("instance_id")) == instance_id:
            return dict(record)
    raise SystemExit(f"Instance {instance_id!r} not found in {path}")


if __name__ == "__main__":
    main()
