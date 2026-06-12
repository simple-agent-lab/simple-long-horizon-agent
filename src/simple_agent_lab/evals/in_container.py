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
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, cast

from ..agents.starter import (
    MCP_ADDENDUM,
    AgentSession,
    agent_session,
    make_bash_agent,
    make_bash_task_agent,
)
from ..core import Agent
from ..hooks import HookMap
from ..llm import REASONING_EFFORTS, ApiKind, Provider, ReasoningEffort
from ..llm_agent import make_llm_agent
from ..skills import system_prompt_with_skills
from ..state import State
from ..tools.bash import make_bash_tool
from ..tools.read import make_read_tool
from ..trace import run_trace_from_state, trace_record
from .protocols import (
    MEMORY_HOME_ENV,
    MEMORY_NAME_ENV,
    MEMORY_RUN_ID_ENV,
    RESULT_KEY,
    TRACE_KEY,
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

# Env contract for the OpenAI-compatible provider, shared by every suite.
OPENAI_MODEL_ENV = "OPENAI_MODEL"
OPENAI_AUTH_ENV = "OPENAI_AUTH_TOKEN"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
OPENAI_SESSION_ID_ENV = "OPENAI_SESSION_ID"
OPENAI_LOG_ID_ENV = "OPENAI_LOG_ID"
# Provider-agnostic reasoning depth, set on the provider so every adapter maps
# it to its own wire shape. The legacy OpenAI-specific name is still honored.
REASONING_EFFORT_ENV = "REASONING_EFFORT"
OPENAI_REASONING_EFFORT_ENV = "OPENAI_REASONING_EFFORT"
API_KIND_ENV = "API_KIND"
API_KIND_CHOICES = ("openai-chat", "openai-responses")
DEFAULT_RESPONSES_MAX_OUTPUT_TOKENS = 32768


def memory_home_from_env(env: Mapping[str, str] | None = None) -> Path | None:
    """Return the optional in-container persistent-memory directory."""

    source = env if env is not None else os.environ
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

    source = env if env is not None else os.environ
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
    """Build the agent for `spec.flavor` with the suite's prompt/role/name."""

    hooks = hooks or {}

    if spec.flavor == "bash":
        return make_bash_agent(
            provider=provider,
            cwd=cwd,
            name=spec.name,
            role=spec.role,
            system_prompt=spec.system_prompt,
            request_extra=request_extra,
            hooks=hooks,
        )
    if spec.flavor == "bash_task":
        return make_bash_task_agent(
            provider=provider,
            cwd=cwd,
            name=spec.name,
            role=spec.role,
            system_prompt=spec.system_prompt,
            request_extra=request_extra,
            hooks=hooks,
        )
    if spec.flavor == "bash_skills":
        # bash + read, with agent skills discovered under `cwd` and advertised
        # in the system prompt; the model loads a skill by reading its SKILL.md
        # and runs its scripts via bash (ADR add-agent-skills).
        return make_llm_agent(
            name=spec.name,
            provider=provider,
            role=spec.role,
            tools=[make_bash_tool(cwd=cwd), make_read_tool(cwd=cwd)],
            system_prompt=system_prompt_with_skills(spec.system_prompt, cwd=cwd),
            target="user",
            request_extra=request_extra,
            hooks=hooks,
        )
    raise SystemExit(
        f"Unsupported agent flavor {spec.flavor!r}; "
        "expected 'bash', 'bash_task', or 'bash_skills'."
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


def _agent_spec(module: ModuleType) -> AgentSpec:
    factory = getattr(module, "agent_spec", None)
    return factory() if callable(factory) else AgentSpec()


def _resolve_mcp_session(
    module: ModuleType,
    *,
    provider: Provider,
    cwd: Path,
    request_extra: Mapping[str, Any] | None,
    mcp_servers: Sequence[Any],
    max_turns: int,
    instance: Mapping[str, Any],
    context: Mapping[str, Any],
) -> AgentSession:
    """Build an MCP-owning session for the suite's supported agent flavor."""

    if callable(getattr(module, "build_agent", None)):
        raise SystemExit(
            "MCP config is not supported with a custom container build_agent hook; "
            "use agent_spec() with a supported flavor instead."
        )

    spec = _agent_spec(module)
    hooks = memory_hooks_from_env(
        provider,
        agent_name=spec.name,
        request_extra=request_extra,
        artifact_builder=_memory_artifact_builder(
            module, workdir=cwd, instance=instance, context=context
        ),
    )
    system_prompt = _with_mcp_addendum(spec.system_prompt)
    if spec.flavor == "bash":
        return agent_session(
            provider,
            cwd=cwd,
            name=spec.name,
            role=spec.role,
            system_prompt=system_prompt,
            mcp_servers=mcp_servers,
            request_extra=request_extra,
            max_turns=max_turns,
            hooks=hooks,
        )
    if spec.flavor == "bash_task":
        return agent_session(
            provider,
            cwd=cwd,
            explorer=True,
            name=spec.name,
            role=spec.role,
            system_prompt=system_prompt,
            mcp_servers=mcp_servers,
            request_extra=request_extra,
            max_turns=max_turns,
            hooks=hooks,
        )
    if spec.flavor == "bash_skills":
        return agent_session(
            provider,
            cwd=cwd,
            read=True,
            name=spec.name,
            role=spec.role,
            system_prompt=_with_mcp_addendum(
                system_prompt_with_skills(spec.system_prompt, cwd=cwd)
            ),
            mcp_servers=mcp_servers,
            request_extra=request_extra,
            max_turns=max_turns,
            hooks=hooks,
        )
    raise SystemExit(
        f"Unsupported agent flavor {spec.flavor!r}; "
        "expected 'bash', 'bash_task', or 'bash_skills'."
    )


def _with_mcp_addendum(system_prompt: str) -> str:
    if MCP_ADDENDUM in system_prompt:
        return system_prompt
    return "\n\n".join(part for part in (system_prompt, MCP_ADDENDUM) if part)


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
        default_reasoning=_reasoning_from_env(source),
    )


def _reasoning_from_env(source: Mapping[str, str]) -> ReasoningEffort | None:
    """Read the normalized reasoning effort; the adapter maps it per-model.

    Honors the provider-agnostic ``REASONING_EFFORT`` and the legacy
    ``OPENAI_REASONING_EFFORT`` name. An unrecognized value is a hard error so
    a typo fails fast rather than silently reaching the model unmapped.
    """
    effort = (
        source.get(REASONING_EFFORT_ENV, "")
        or source.get(OPENAI_REASONING_EFFORT_ENV, "")
    ).strip()
    if not effort:
        return None
    if effort not in REASONING_EFFORTS:
        raise SystemExit(
            f"Unsupported reasoning effort {effort!r}; "
            f"expected one of {REASONING_EFFORTS}."
        )
    return cast(ReasoningEffort, effort)


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

    def trace_bytes(*, in_progress: bool) -> bytes:
        trace = run_trace_from_state(
            state=state,
            trace_id=trace_id,
            producer=producer,
            meta={
                "suite": suite_name,
                "instance_id": instance_id,
                "in_progress": in_progress,
                "oracle": oracle,
                "result_keys": sorted(state.data.get("result", {})),
            },
        )
        return (json.dumps(trace_record(trace), ensure_ascii=False) + "\n").encode(
            "utf-8"
        )

    if oracle:
        # No model, no turns: apply the reference solution, then extract.
        _run_oracle(module, workdir=workdir, instance=instance)
        state = State(task=task)
    else:
        if provider is None:
            raise SystemExit("a Provider is required unless oracle=True")
        mcp_servers = _load_mcp_servers(store)
        abort_fn = lambda: False  # noqa: E731
        if wall_time_seconds is not None:
            _deadline = time.monotonic() + wall_time_seconds
            abort_fn = lambda: time.monotonic() >= _deadline  # noqa: E731
            print(
                f"[in_container] wall-time limit: {wall_time_seconds:.0f}s "
                f"({wall_time_seconds / 3600:.1f}h)",
                flush=True,
            )
        last = 0.0
        if mcp_servers:
            if not isinstance(task, str):
                raise SystemExit("MCP-enabled eval runs currently require a text task")
            with _resolve_mcp_session(
                module,
                provider=provider,
                cwd=workdir,
                request_extra=request_extra,
                mcp_servers=mcp_servers,
                max_turns=max_turns,
                instance=instance,
                context=context,
            ) as session:
                state, events = session.run(
                    task, max_turns=max_turns, abort=abort_fn
                )
                for _ in events:
                    now = time.monotonic()
                    if now - last >= flush_interval_s:
                        store.put(TRACE_KEY, trace_bytes(in_progress=True))
                        last = now
        else:
            agent = _resolve_agent(
                module,
                provider=provider,
                cwd=workdir,
                request_extra=request_extra,
                instance=instance,
                context=context,
            )
            state, events = agent.run(
                task, max_turns=max_turns, abort=abort_fn
            )
            for _ in events:
                now = time.monotonic()
                if now - last >= flush_interval_s:
                    store.put(TRACE_KEY, trace_bytes(in_progress=True))
                    last = now

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


def _load_mcp_servers(store: ArtifactStore) -> tuple[Any, ...]:
    """Read optional staged MCP config and return validated server configs."""

    from .protocols import MCP_KEY

    try:
        raw = store.get(MCP_KEY)
    except (FileNotFoundError, OSError):
        return ()
    from simple_agent_lab.mcp.config_file import mcp_server_configs_from_json

    return mcp_server_configs_from_json(raw.decode("utf-8"), source=MCP_KEY)


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
