"""Generic in-container runner (ADR generic-containerized-eval-framework).

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

from ..agents.flavors import (
    ArtifactPut,
    build_flavor_agent,
)
from ..core import Agent
from ..hooks import HookMap
from ..llm import Provider
from ..llm.env import (
    API_KIND_CHOICES,
    API_KIND_ENV,
    FAKE_PROVIDER,
    OPENAI_ENV,
    request_extra_from_env,
)
from ..llm.env import provider_from_env as _env_provider_from_env
from ..state import State
from ..trace import event_stream, run_trace_from_state
from .protocols import (
    MEMORY_HOME_ENV,
    MEMORY_NAME_ENV,
    MEMORY_RUN_ID_ENV,
    RESULT_KEY,
    TRACE_KEY,
    TRACE_RAW_KEY,
    AgentSpec,
    ArtifactStore,
    ContainerTask,
)
from .stores import container_store_from_env

__all__ = [
    "build_agent",
    "main",
    "memory_home_from_env",
    "memory_hooks_from_env",
    "provider_from_env",
    "run_in_container",
]

TRACE_FLUSH_INTERVAL_S = 2.0

# The OpenAI-compatible env contract (`OPENAI_MODEL`/`OPENAI_AUTH_TOKEN`/...),
# the `.env` loader, and the provider builder now live in `simple_agent_lab.llm.env`
# — the single source of truth (see ADR consolidate-provider-env). `API_KIND_*`
# are re-imported above only because the argparse parser below references them.


def memory_home_from_env(env: Mapping[str, str] | None = None) -> Path | None:
    """Return the optional in-container persistent-memory directory."""

    source = env if env is not None else os.environ  # env-ok: default to process env
    value = source.get(MEMORY_HOME_ENV, "").strip()
    if not value:
        return None
    return Path(value).expanduser()


def memory_hooks_from_env(
    provider: Provider,
    *,
    agent_name: str,
    request_extra: Mapping[str, Any] | None = None,
    artifact_builder: Callable[[Any], Iterable[Any]] | None = None,
    env: Mapping[str, str] | None = None,
) -> HookMap:
    """Build filesystem-memory lifecycle hooks for an in-container run.

    With no ``SAL_MEMORY_HOME`` this returns ``{}``, leaving non-memory runs
    unchanged. When active, the distiller reuses the agent provider and request
    extras so memory's model call carries the same gateway headers as the main
    agent.
    """

    memory_home = memory_home_from_env(env)
    if memory_home is None:
        return {}

    from ..memory import (
        FilesystemMemory,
        MemoryContext,
        make_filesystem_distiller,
    )

    source = env if env is not None else os.environ  # env-ok: default to process env
    memory = FilesystemMemory(
        root=memory_home,
        distiller=make_filesystem_distiller(provider, request_extra=request_extra),
        artifact_builder=artifact_builder,
    )
    ctx = MemoryContext(
        agent=agent_name,
        task="",  # filled from runtime State at recall and finish
        run_id=source.get(MEMORY_RUN_ID_ENV, "").strip(),
        memory_name=source.get(MEMORY_NAME_ENV, "").strip(),
    )
    return memory.bind(ctx).hooks


def _memory_artifact_builder(
    module: ModuleType,
    *,
    workdir: Path,
    instance: Mapping[str, Any],
    context: Mapping[str, Any],
) -> Callable[[Any], Iterable[Any]] | None:
    """Adapt a suite's optional ``memory_artifacts`` hook."""

    collector = getattr(module, "memory_artifacts", None)
    if not callable(collector):
        return None

    def build(_memory_ctx: Any) -> Iterable[Any]:
        del _memory_ctx
        return collector(workdir, instance, **_context_kwargs(collector, context))

    return build


# --------------------------------------------------------------------------- #
# Agent construction (suite-tunable via agent_spec or a full build_agent hook)
# --------------------------------------------------------------------------- #
def build_agent(
    *,
    spec: AgentSpec,
    provider: Provider,
    cwd: Path,
    request_extra: Mapping[str, Any] | None = None,
    hooks: HookMap | None = None,
) -> Agent:
    """Build the agent for `spec.flavor` with the suite's prompt/role/name.

    Kept as the eval-facing compatibility wrapper; the flavor implementation is
    owned by `simple_agent_lab.agents.flavors`.
    """

    return build_flavor_agent(
        flavor=spec.flavor,
        provider=provider,
        cwd=cwd,
        name=spec.name,
        role=spec.role,
        system_prompt=spec.system_prompt,
        request_extra=request_extra,
        hooks=hooks,
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
    instance: Mapping[str, Any],
    context: Mapping[str, Any],
    trace_put: ArtifactPut | None = None,
) -> Agent:
    """A container module may supply `build_agent` for full control, else `agent_spec`.

    Both are *optional* members, so they're probed with `getattr(..., None)`;
    `module` is the honest `ModuleType` rather than the required-surface
    `ContainerTask` (which `run_in_container` casts to for the required calls).
    """

    # A custom build_agent may handle only SOME flavors (e.g. the workflow arms)
    # and return None for the rest, delegating those back to the agent_spec path
    # so they still get memory hooks. Non-None wins.
    agent = _custom_build_agent(
        module,
        provider=provider,
        cwd=cwd,
        request_extra=request_extra,
        context=context,
        trace_put=trace_put,
    )
    if agent is not None:
        return agent
    factory = getattr(module, "agent_spec", None)
    spec = factory() if callable(factory) else AgentSpec()
    hooks = memory_hooks_from_env(
        provider,
        agent_name=spec.name,
        request_extra=request_extra,
        artifact_builder=_memory_artifact_builder(
            module, workdir=cwd, instance=instance, context=context
        ),
    )
    return build_agent(
        spec=spec,
        provider=provider,
        cwd=cwd,
        request_extra=request_extra,
        hooks=hooks,
    )


def _custom_build_agent(
    module: ModuleType,
    *,
    provider: Provider,
    cwd: Path,
    request_extra: Mapping[str, Any] | None,
    context: Mapping[str, Any] | None = None,
    trace_put: ArtifactPut | None = None,
) -> Agent | None:
    """The module's optional `build_agent` hook, or None when absent or declined.

    A custom `build_agent` claims a flavor by returning an `Agent` and declines
    one by returning None (so it falls through to the `agent_spec` path).

    ``trace_put`` is the artifact sink for workflow sub-agent traces; it is
    passed only to hooks that declare it, so a suite building a workflow arm
    does not re-fetch the container store itself.
    """
    custom = getattr(module, "build_agent", None)
    if not callable(custom):
        return None
    extra = _context_kwargs(custom, context or {})
    if "trace_put" in inspect.signature(custom).parameters:
        extra["trace_put"] = trace_put
    return custom(
        provider=provider,
        cwd=cwd,
        request_extra=request_extra,
        **extra,
    )


# --------------------------------------------------------------------------- #
# Provider from env (generic OpenAI-compatible + fake)
# --------------------------------------------------------------------------- #
def provider_from_env(
    *, kind: str, api_kind: str = "openai-chat", env: Mapping[str, str] | None = None
) -> Provider:
    """Build the in-container provider: ``kind="fake"`` or an OpenAI-compatible one.

    Thin wrapper over `simple_agent_lab.llm.env.provider_from_env` that keeps the
    framework's ``kind``/``api_kind`` calling convention. The OpenAI path reads
    the reasoning effort from the environment and stamps ``temperature=1.0``.
    `request_extra_from_env` is re-exported from the same module.
    """
    if kind == "fake":
        return FAKE_PROVIDER
    return _env_provider_from_env(
        OPENAI_ENV,
        env=env,
        api_kind=api_kind,
        default_temperature=1.0,
        read_reasoning=True,
        label="the openai provider",
    )


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
    wall_time_seconds: float | None = None,
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

    def trace_artifacts(
        *, in_progress: bool, trace_state: State | None = None
    ) -> tuple[bytes, bytes | None]:
        src = trace_state if trace_state is not None else state
        trace = run_trace_from_state(
            state=src,
            trace_id=trace_id,
            producer=producer,
            meta={
                "suite": suite_name,
                "instance_id": instance_id,
                "in_progress": in_progress,
                "oracle": oracle,
                "result_keys": sorted(src.data.get("result", {})),
            },
        )
        header, lines, raw_pool = event_stream(trace)
        raw_bytes = None
        if raw_pool:
            raw_bytes = "".join(
                json.dumps(blob, ensure_ascii=False) + "\n" for blob in raw_pool
            ).encode("utf-8")
        trace_bytes = "".join(
            json.dumps(rec, ensure_ascii=False) + "\n" for rec in (header, *lines)
        ).encode("utf-8")
        return trace_bytes, raw_bytes

    def put_trace(*, in_progress: bool, trace_state: State | None = None) -> None:
        trace_data, raw_data = trace_artifacts(
            in_progress=in_progress,
            trace_state=trace_state,
        )
        store.put(TRACE_KEY, trace_data)
        if raw_data is not None:
            store.put(TRACE_RAW_KEY, raw_data)

    if oracle:
        # No model, no turns: apply the reference solution, then extract.
        _run_oracle(module, workdir=workdir, instance=instance)
        state = State(task=task)
    else:
        if provider is None:
            raise SystemExit("a Provider is required unless oracle=True")
        trace_agent: Agent | None = None
        abort_fn = lambda: False  # noqa: E731
        context["runtime"] = {
            "max_turns": max_turns,
            "wall_time_seconds": wall_time_seconds,
            "started_monotonic": time.monotonic(),
        }
        if wall_time_seconds is not None:
            _deadline = time.monotonic() + wall_time_seconds
            abort_fn = lambda: time.monotonic() >= _deadline  # noqa: E731
            print(
                f"[in_container] wall-time limit: {wall_time_seconds:.0f}s "
                f"({wall_time_seconds / 3600:.1f}h)",
                flush=True,
            )
        last = 0.0
        agent = _resolve_agent(
            module,
            provider=provider,
            cwd=workdir,
            request_extra=request_extra,
            instance=instance,
            context=context,
            trace_put=store.put,
        )
        trace_agent = agent
        state, events = agent.run(task, max_turns=max_turns, abort=abort_fn)
        for _ in events:
            now = time.monotonic()
            if now - last >= flush_interval_s:
                put_trace(in_progress=True)
                last = now

    extract = tasks.extract_result
    result = dict(extract(workdir, instance, **_context_kwargs(extract, context)))

    # A workflow facade stashes its per-step breakdown on the run state at
    # session end; fold it into the result here so every suite reports it
    # without each re-implementing the recording + extract wiring.
    workflow_breakdown = state.data.get("workflow")
    if workflow_breakdown is not None:
        result.setdefault("workflow", workflow_breakdown)

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
    # An agent may compose a richer FINAL trace than the bare run state — e.g. a
    # workflow facade whose real work ran in sub-agents folds them into a
    # lightweight tree (one node per sub-agent) for the viewer. Optional + best
    # effort; the live in-progress writes above always used the real run state.
    final_trace_state = state
    if not oracle and trace_agent is not None:
        try:
            final_trace_state = trace_agent.trace_state(state)
        except Exception:
            pass
    put_trace(in_progress=False, trace_state=final_trace_state)
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
        "--wall-time-seconds",
        type=float,
        default=None,
        help="Wall-clock time limit for the agent run (seconds). None = no limit.",
    )
    parser.add_argument(
        "--provider", choices=["fake", "openai", "oracle"], default="openai"
    )
    parser.add_argument(
        "--api-kind",
        choices=API_KIND_CHOICES,
        # env-ok: CLI default mirrors API_KIND for the in-container entrypoint
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
        wall_time_seconds=args.wall_time_seconds,
        store=store,
        trace_id=f"{args.suite_name}.{args.instance_id}",
        producer=f"suite:{args.suite_name}",
        suite_name=args.suite_name,
        request_extra=(request_extra_from_env() if args.provider == "openai" else {}),
        oracle=oracle,
    )
    print(f"wrote result + trajectory for {args.instance_id} via artifact store")


if __name__ == "__main__":
    main()
