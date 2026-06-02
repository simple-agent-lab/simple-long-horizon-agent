"""Generic in-container runner (ADR 0017).

This module is what runs *inside* the eval container, invoked as
``python -m simple_agent_lab.evals.in_container`` (it ships in the wheel, so
nothing is copied in). It imports the suite's container half by dotted path
(`Suite.container_module`), builds the agent, drives the loop, and reads/writes
everything through one `ArtifactStore` (the agent's `generate` retries transient
provider throttling on its own — see `simple_agent_lab.llm.retry`):

- reads the sanitized instance from ``input/instance.json``,
- re-writes ``out/trajectory.jsonl`` on a cadence (the live trace push),
- writes the raw `extract_result` product to ``out/result.json``.

``out/result.json`` is the single decoupling artifact. A suite that scores in
the run environment may expose an optional ``evaluate(workspace, instance, *,
context)``; when the suite staged ``eval_inputs`` (gold, threaded in as
``context["eval"]``) the runner calls it and merges its verdict into the result
here. Otherwise scoring is a follow-up run or an external oracle reading
``out/result.json``. Everything here is suite-agnostic: a new benchmark supplies only
``build_task`` / ``extract_result`` (and optional ``prepare`` / ``evaluate`` /
``agent_spec`` / ``build_agent``) in its container module. Nothing about
SWE-bench, datasets, or patches lives here.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, cast

from ..agents.bash import make_bash_agent
from ..agents.bash_task import make_bash_task_agent
from ..core import Agent
from ..llm import ApiKind, Provider
from ..state import State
from ..trajectory import run_trace_from_state, trace_record
from .protocols import RESULT_KEY, TRACE_KEY, AgentSpec, ArtifactStore, ContainerTask
from .stores import container_store_from_env

__all__ = [
    "build_agent",
    "main",
    "provider_from_env",
    "resume_in_container",
    "run_in_container",
]

TRACE_FLUSH_INTERVAL_S = 2.0

# Env contract for the OpenAI-compatible provider, shared by every suite.
OPENAI_MODEL_ENV = "OPENAI_MODEL"
OPENAI_AUTH_ENV = "OPENAI_AUTH_TOKEN"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
OPENAI_SESSION_ID_ENV = "OPENAI_SESSION_ID"
OPENAI_LOG_ID_ENV = "OPENAI_LOG_ID"
API_KIND_ENV = "API_KIND"
API_KIND_CHOICES = ("openai-chat", "openai-responses")
DEFAULT_RESPONSES_MAX_OUTPUT_TOKENS = 32768


# --------------------------------------------------------------------------- #
# Agent construction (suite-tunable via agent_spec or a full build_agent hook)
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


def _run_oracle(
    module: ModuleType, *, workdir: Path, instance: Mapping[str, Any]
) -> None:
    """Apply the suite's reference ("oracle") solution instead of running a model.

    Oracle mode is a deterministic, model-free check that a suite is wired
    correctly: the container half supplies ``apply_oracle(workspace, instance)``
    that lands the known-good solution in the workspace, after which
    ``extract_result`` should reproduce the gold product. A suite that does not
    expose ``apply_oracle`` cannot be oracle-checked, which is an explicit error
    rather than a silent no-op.
    """

    apply_oracle = getattr(module, "apply_oracle", None)
    if not callable(apply_oracle):
        raise RuntimeError(
            f"oracle run needs apply_oracle(workspace, instance) in "
            f"{module.__name__!r}; none found."
        )
    apply_oracle(workdir, instance)


def _resolve_agent(
    module: ModuleType,
    *,
    provider: Provider,
    cwd: Path,
    request_extra: Mapping[str, Any] | None,
) -> Agent:
    """A container module may supply `build_agent` for full control, else `agent_spec`.

    Both are *optional* members, so they're probed with `getattr(..., None)`;
    `module` is the honest `ModuleType` rather than the required-surface
    `ContainerTask` (which `run_in_container` casts to for the required calls).
    """

    custom = getattr(module, "build_agent", None)
    if callable(custom):
        return custom(provider=provider, cwd=cwd, request_extra=request_extra)
    factory = getattr(module, "agent_spec", None)
    spec = factory() if callable(factory) else AgentSpec()
    return build_agent(
        spec=spec, provider=provider, cwd=cwd, request_extra=request_extra
    )


# --------------------------------------------------------------------------- #
# Provider from env (generic OpenAI-compatible + fake)
# --------------------------------------------------------------------------- #
def provider_from_env(
    *, kind: str, api_kind: str = "openai-chat", env: Mapping[str, str] | None = None
) -> Provider:
    source = env if env is not None else os.environ
    if kind == "fake":
        return Provider(id="fake", api="fake", model="fake-model")
    if api_kind not in API_KIND_CHOICES:
        raise SystemExit(f"Unsupported API_KIND {api_kind!r}: {API_KIND_CHOICES}")
    model = source.get(OPENAI_MODEL_ENV, "").strip()
    token = source.get(OPENAI_AUTH_ENV, "").strip()
    missing = [
        n for n, v in ((OPENAI_MODEL_ENV, model), (OPENAI_AUTH_ENV, token)) if not v
    ]
    if missing:
        raise SystemExit("Missing env for openai provider: " + ", ".join(missing))
    return Provider(
        id=api_kind,
        api=cast(ApiKind, api_kind),
        model=model,
        base_url=source.get(OPENAI_BASE_URL_ENV) or None,
        api_key_env=OPENAI_AUTH_ENV,
        default_max_tokens=(
            DEFAULT_RESPONSES_MAX_OUTPUT_TOKENS
            if api_kind == "openai-responses"
            else None
        ),
        default_temperature=1.0,
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


# --------------------------------------------------------------------------- #
# The generic run
# --------------------------------------------------------------------------- #
def run_in_container(
    *,
    instance: Mapping[str, Any],
    container_module: str,
    provider: Provider | None,
    workdir: Path,
    max_turns: int,
    store: ArtifactStore,
    trace_id: str,
    producer: str,
    suite_name: str,
    request_extra: Mapping[str, Any] | None = None,
    flush_interval_s: float = TRACE_FLUSH_INTERVAL_S,
    oracle: bool = False,
) -> tuple[dict[str, Any], State]:
    """Drive one instance: build task → run agent (or oracle) → extract result.

    Reads nothing from `store` (the caller passes the loaded `instance`); writes
    the live trajectory to ``out/trajectory.jsonl`` on a cadence and the raw
    `extract_result` product to ``out/result.json``. Returns both.

    When `oracle` is set, the agent loop is replaced by the suite's
    ``apply_oracle`` (the reference solution); `provider` is unused and may be
    ``None``. This is the deterministic, model-free suite self-check.
    """

    module = importlib.import_module(container_module)
    # The import is a dynamic boundary: `module` is a ModuleType the checker
    # can't introspect. Cast to the required-surface protocol so the build_task
    # / extract_result calls below are type-checked (a typo or wrong arg is
    # caught); optional members stay dynamic via getattr on the raw module.
    tasks = cast(ContainerTask, module)

    # Optional pre-run setup (checkout, snapshot a baseline, install ignore
    # rules). Its returned dict is threaded into extract_result as `context`.
    context: dict[str, Any] = {}
    prepare = getattr(module, "prepare", None)
    if callable(prepare):
        context = dict(prepare(workdir, instance) or {})

    task = tasks.build_task(instance, workdir=str(workdir))
    instance_id = str(instance.get("instance_id", "?"))
    meta_base = {"suite": suite_name, "instance_id": instance_id, "oracle": oracle}

    if oracle:
        # No model, no turns: apply the reference solution, then extract.
        _run_oracle(module, workdir=workdir, instance=instance)
        state = State(task=task)
        events: Iterable[Any] = ()
    else:
        if provider is None:
            raise SystemExit("a Provider is required unless oracle=True")
        agent = _resolve_agent(
            module, provider=provider, cwd=workdir, request_extra=request_extra
        )
        state, events = agent.run(task, max_turns=max_turns)

    trace_bytes = _live_trace_bytes(
        state, trace_id=trace_id, producer=producer, meta_base=meta_base
    )
    _drain_with_live_trace(
        events, store=store, trace_bytes=trace_bytes, flush_interval_s=flush_interval_s
    )

    result = _finalize_run(
        module=module,
        tasks=tasks,
        state=state,
        workdir=workdir,
        instance=instance,
        context=context,
        store=store,
        trace_bytes=trace_bytes,
    )
    return result, state


def _finalize_run(
    *,
    module: ModuleType,
    tasks: ContainerTask,
    state: State,
    workdir: Path,
    instance: Mapping[str, Any],
    context: Mapping[str, Any],
    store: ArtifactStore,
    trace_bytes: Callable[..., bytes],
) -> dict[str, Any]:
    """Extract the result, run optional in-env scoring, persist result + trace.

    Shared tail for both a fresh run (`run_in_container`) and a resumed
    replay (`resume_in_container`): once the agent loop has populated
    `state`, the product is derived from the workspace the same way
    regardless of how the loop got there.
    """

    extract = tasks.extract_result
    result = dict(extract(workdir, instance, **_context_kwargs(extract, context)))

    # Optional in-environment scoring: a suite that scores where the run ran
    # exposes ``evaluate(workspace, instance, *, context)`` and stages gold via
    # ``eval_inputs`` (the host writes it under EVAL_KEY). Staged gold is the
    # toggle — present gold means score here and merge the verdict into the
    # result; absent means score elsewhere (a follow-up run or external oracle).
    # Environment-neutral: this runs in-process or in-container, wherever the run
    # ran.
    evaluate = getattr(module, "evaluate", None)
    eval_inputs = _load_eval_inputs(store)
    if callable(evaluate) and eval_inputs:
        eval_context = {**context, "eval": eval_inputs}
        verdict = evaluate(workdir, instance, **_context_kwargs(evaluate, eval_context))
        if verdict:
            result.update(dict(verdict))

    state.data["result"] = result
    store.put(
        RESULT_KEY, (json.dumps(result, ensure_ascii=False) + "\n").encode("utf-8")
    )
    store.put(TRACE_KEY, trace_bytes(in_progress=False))
    return result


def _live_trace_bytes(
    state: State,
    *,
    trace_id: str,
    producer: str,
    meta_base: Mapping[str, Any],
) -> Callable[..., bytes]:
    """Build the ``trace_bytes(in_progress=...)`` serializer for a live trace.

    `meta_base` carries the run-shape-specific fields (e.g. ``oracle`` for a
    fresh run, ``resumed_from_message`` for a replay); ``in_progress`` and the
    live ``result_keys`` are stamped per call. Shared by `run_in_container`
    and `resume_in_container` so the trace record shape stays identical.
    """

    def trace_bytes(*, in_progress: bool) -> bytes:
        trace = run_trace_from_state(
            state=state,
            trace_id=trace_id,
            producer=producer,
            meta={
                **meta_base,
                "in_progress": in_progress,
                "result_keys": sorted(state.data.get("result", {})),
            },
        )
        return (json.dumps(trace_record(trace), ensure_ascii=False) + "\n").encode(
            "utf-8"
        )

    return trace_bytes


def _drain_with_live_trace(
    events: Iterable[Any],
    *,
    store: ArtifactStore,
    trace_bytes: Callable[..., bytes],
    flush_interval_s: float,
) -> None:
    """Consume the agent event stream, flushing the live trace on a cadence."""

    last = 0.0
    for _ in events:
        now = time.monotonic()
        if now - last >= flush_interval_s:
            store.put(TRACE_KEY, trace_bytes(in_progress=True))
            last = now


def resume_in_container(
    *,
    prior_trace: Mapping[str, Any],
    fork_message_index: int,
    instance: Mapping[str, Any],
    container_module: str,
    provider: Provider,
    workdir: Path,
    max_turns: int,
    store: ArtifactStore,
    trace_id: str,
    producer: str,
    suite_name: str,
    request_extra: Mapping[str, Any] | None = None,
    flush_interval_s: float = TRACE_FLUSH_INTERVAL_S,
    rebuild_side_effects: bool = True,
) -> tuple[dict[str, Any], State]:
    """Replay one instance from `fork_message_index` of a prior trace.

    The "rebuild, don't snapshot" path (replay-to-rebuild): this assumes a
    **fresh** environment at the suite's baseline (a container booted from the
    run's baseline image, or a freshly checked-out workspace), then —

    1. runs the suite's ``prepare`` hook so the workspace matches the original
       run's starting point (checkout, baseline commit, ignore rules);
    2. rebuilds the agent's `State` from `prior_trace` and forks it at
       `fork_message_index`;
    3. when `rebuild_side_effects` is set, re-executes the tool calls recorded
       in the kept prefix (via `simple_agent_lab.replay.replay_side_effects`)
       so the workspace's filesystem reaches the fork point without any stored
       per-step snapshot — and without adding commits to the testbed history;
    4. resumes the model loop from there, writing the new trajectory on the
       same cadence as a fresh run.

    Fork at a tool-result / task boundary (not mid-turn) so every recorded
    call's effect had been applied by that point — see `replay_side_effects`.
    """

    from ..replay import replay_side_effects, resume_from_trace_record

    module = importlib.import_module(container_module)
    tasks = cast(ContainerTask, module)

    context: dict[str, Any] = {}
    prepare = getattr(module, "prepare", None)
    if callable(prepare):
        context = dict(prepare(workdir, instance) or {})

    agent = _resolve_agent(
        module, provider=provider, cwd=workdir, request_extra=request_extra
    )
    instance_id = str(instance.get("instance_id", "?"))

    def on_fork(forked: State) -> None:
        if rebuild_side_effects:
            replay_side_effects(forked.messages, agent.tools)

    state, events = resume_from_trace_record(
        agent,
        dict(prior_trace),
        fork_message_index,
        max_turns=max_turns,
        on_fork=on_fork,
    )

    trace_bytes = _live_trace_bytes(
        state,
        trace_id=trace_id,
        producer=producer,
        meta_base={
            "suite": suite_name,
            "instance_id": instance_id,
            "resumed_from_message": fork_message_index,
        },
    )
    _drain_with_live_trace(
        events, store=store, trace_bytes=trace_bytes, flush_interval_s=flush_interval_s
    )

    result = _finalize_run(
        module=module,
        tasks=tasks,
        state=state,
        workdir=workdir,
        instance=instance,
        context=context,
        store=store,
        trace_bytes=trace_bytes,
    )
    return result, state


def _context_kwargs(
    fn: Callable[..., Any], context: Mapping[str, Any]
) -> dict[str, Any]:
    """Pass ``context`` only to hooks that declare it (keeps the surface optional)."""

    return {"context": context} if "context" in inspect.signature(fn).parameters else {}


def _load_eval_inputs(store: ArtifactStore) -> dict[str, Any]:
    """Read the host-staged gold scoring inputs (EVAL_KEY), or {} if none."""

    from .protocols import EVAL_KEY

    try:
        raw = store.get(EVAL_KEY)
    except (FileNotFoundError, OSError):
        return {}
    return json.loads(raw.decode("utf-8") or "{}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generic in-container eval runner.")
    parser.add_argument("--container-module", required=True)
    parser.add_argument("--suite-name", default="suite")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--workdir", default="/testbed")
    parser.add_argument("--max-turns", type=int, default=75)
    parser.add_argument(
        "--provider", choices=["fake", "openai", "oracle"], default="openai"
    )
    parser.add_argument(
        "--api-kind",
        choices=API_KIND_CHOICES,
        default=os.environ.get(API_KIND_ENV, "openai-chat"),
    )
    args = parser.parse_args(argv)

    from .protocols import INSTANCE_KEY

    store = container_store_from_env()
    instance = json.loads(store.get(INSTANCE_KEY).decode("utf-8"))
    oracle = args.provider == "oracle"
    provider = (
        None
        if oracle
        else provider_from_env(kind=args.provider, api_kind=args.api_kind)
    )
    run_in_container(
        instance=instance,
        container_module=args.container_module,
        provider=provider,
        workdir=Path(args.workdir),
        max_turns=args.max_turns,
        store=store,
        trace_id=f"{args.suite_name}.{args.instance_id}",
        producer=f"suite:{args.suite_name}",
        suite_name=args.suite_name,
        request_extra=request_extra_from_env() if args.provider == "openai" else {},
        oracle=oracle,
    )
    print(f"wrote result + trajectory for {args.instance_id} via artifact store")


if __name__ == "__main__":
    main()
