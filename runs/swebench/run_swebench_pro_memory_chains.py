"""Run the full SWE-bench Pro split as memory-sharing issue chains.

This is the *memory-based* peer of ``run_swebench_pro_repo_chains.py``. Each
issue runs as an ordinary, isolated SWE-bench Pro instance in a *fresh* agent
context (no transcript/handoff is carried in-context), and the only thing that
crosses instance boundaries is Simple Agent Lab filesystem memory scoped per
chain. Within a chain the memory dir is shared (``SAL_MEMORY_NAME=<chain_id>``);
the run-end distiller updates it, so a later issue in the same chain can reuse an
earlier one's lessons. Each container bind-mounts only that chain's namespaced
subdirectory; sibling chains under the host memory root are not visible.

Key differences from the repo-chain runner:

- Chains come from a pre-analyzed chain manifest (``--chains-json``), not from
  splitting a repo by commit time. The runner intentionally has no default:
  pass the exact chain file for each experiment run. The deep chain-nodes JSONL
  is vendored under ``evals/swebench/data/`` so a run needs no external checkout.
- The full split still runs: every dataset instance not covered by a chain
  becomes a length-1 singleton (memory off by default; ``--singleton-memory``
  turns it on).
- Run units are ordered longest-first and handed to a *fixed* worker pool, so
  the long chains start early and the many short/singleton runs backfill the
  remaining lanes while those long chains keep occupying theirs.
- Memory replaces context handoff, so there is no ``--compression-strategy`` /
  ``--handoff`` / ``chain_state.json`` plumbing here.

Example smoke (no Docker cost past a couple of instances):

    uv run --extra swebench python runs/swebench/run_swebench_pro_memory_chains.py \
      --chains-json evals/swebench/data/swe_bench_pro_chain_experiment_nodes_deep.jsonl \
      --max-chains 1 --limit 2 --max-turns 5

Formal run shape:

    uv run --extra swebench python runs/swebench/run_swebench_pro_memory_chains.py \
      --all \
      --chains-json evals/swebench/data/swe_bench_pro_chain_experiment_nodes_deep.jsonl \
      --parallel 23 \
      --provider-auth-envs OPENAI_AUTH_TOKEN:12,OPENAI_AUTH_TOKEN2:11 \
      --api-kind openai-responses \
      --max-turns 250 \
      --run-official-eval

Pass ``--chains-json PATH`` explicitly for every experiment run. The path can be
the vendored flat chain-nodes JSONL or another manifest with the same shape.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evals.swebench import harness  # noqa: E402
from evals.swebench import pro_chain_runner as chain_runner  # noqa: E402
from evals.swebench.pro_memory_chain import (  # noqa: E402
    MEMORY_CHAIN_AGENT_FLAVORS,
    MemoryChain,
    ProMemoryChainConfig,
    RawIssueChain,
    load_issue_chains,
    model_name_for_config,
    plan_manifest,
    plan_memory_chains,
)
from evals.swebench.suite import SwebenchSuite  # noqa: E402
from simple_agent_lab.agent_flavors import AGENT_FLAVOR_ENV  # noqa: E402
from simple_agent_lab.evals import LocalDirStore, LocalDockerBackend  # noqa: E402
from simple_agent_lab.evals.protocols import (  # noqa: E402
    DEFAULT_MEMORY_CONTAINER_HOME,
    MEMORY_NAME_ENV,
    MEMORY_RUN_ID_ENV,
    RESULT_KEY,
)
from simple_agent_lab.evals.runner import (  # noqa: E402
    canonical_run_id,
    container_name,
    prepare_run_directory,
    run_suite_instance,
    safe_path_part,
)
from simple_agent_lab.llm.env import (  # noqa: E402
    OPENAI_MODEL_ENV,
    OPENAI_REASONING_EFFORT_ENV,
    REASONING_EFFORT_ENV,
)
from simple_agent_lab.memory import (  # noqa: E402
    FilesystemMemory,
    FilesystemMemoryLimits,
)
from simple_agent_lab.memory.filesystem import safe_memory_name  # noqa: E402
from simple_agent_lab.trace import write_jsonl_atomic  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    chain_runner.add_common_arguments(
        parser,
        dataset_name=ProMemoryChainConfig.dataset_name,
        split=ProMemoryChainConfig.split,
        max_turns=ProMemoryChainConfig.max_turns,
        api_kind=ProMemoryChainConfig.api_kind,
    )
    parser.add_argument(
        "--agent-flavor",
        default=ProMemoryChainConfig.agent_flavor,
        choices=MEMORY_CHAIN_AGENT_FLAVORS,
        help=(
            "Agent flavor for every instance. Only simple flavors are allowed "
            "because the generic runner's agent_spec path is what installs the "
            "filesystem-memory hooks (workflow arms bypass them)."
        ),
    )
    parser.add_argument(
        "--memory",
        action=argparse.BooleanOptionalAction,
        default=ProMemoryChainConfig.memory,
        help="Enable filesystem memory for multi-issue chains (default: enabled).",
    )
    parser.add_argument(
        "--singleton-memory",
        action=argparse.BooleanOptionalAction,
        default=ProMemoryChainConfig.singleton_memory,
        help=(
            "Also give each off-chain singleton its own memory namespace "
            "(default: disabled — singletons run without memory)."
        ),
    )
    parser.add_argument(
        "--memory-home",
        default=None,
        help=(
            "Host root for per-chain read-write memory directories. Each "
            "container mounts only its own child. Defaults to "
            "<run-root>/<run-id>/memory so each run starts from a clean slate."
        ),
    )
    parser.add_argument(
        "--memory-max-namespaces",
        type=int,
        default=None,
        help=(
            "Namespace admission cap. Persistent --memory-home roots default to "
            "128; a run-local root automatically fits the finite planned batch."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.all and not args.repos and not args.instance_json:
        raise SystemExit("Pass --all, --repos REPO..., or --instance-json PATH.")
    chains_path = chain_runner.chains_json_path(args.chains_json)

    harness.load_dotenv(args.dotenv)
    chain_runner.apply_provider_env_overrides(
        model=args.model, reasoning_effort=args.reasoning_effort
    )
    api_kind = harness.resolve_api_kind(args.api_kind)
    config = _experiment_config_from_args(args, api_kind=api_kind)
    args.run_id = canonical_run_id(_resolve_run_id(args.run_id, config))
    run_root = Path(args.run_root)

    rows = chain_runner.load_rows(
        instance_json=args.instance_json,
        dataset_name=args.dataset_name,
        split=args.split,
        repos=args.repos,
    )
    raw_chains = _filter_raw_chains(load_issue_chains(chains_path), repos=args.repos)
    plan = plan_memory_chains(
        rows,
        raw_chains,
        memory=config.memory,
        singleton_memory=config.singleton_memory,
    )
    units = chain_runner.select_units(
        plan.chains,
        max_chains=args.max_chains,
        limit=args.limit,
    )
    if not units:
        raise SystemExit("No SWE-bench Pro instances selected.")

    auth_lanes = chain_runner.resolve_auth_lanes(args.provider_auth_envs, args.parallel)

    memory_home = _resolve_memory_home(args, run_root=run_root)
    memory_namespaces = {
        unit.chain_id: safe_memory_name(unit.chain_id)
        for unit in units
        if unit.memory_enabled
    }
    reverse: dict[str, str] = {}
    for chain_id, namespace in memory_namespaces.items():
        prior = reverse.setdefault(namespace.casefold(), chain_id)
        if prior != chain_id:
            raise SystemExit(
                "filesystem memory namespace collision between chain ids "
                f"{prior!r} and {chain_id!r}: {namespace!r}"
            )
    memory_namespace_limit = _resolve_memory_namespace_limit(
        args,
        requested_count=len(memory_namespaces),
    )
    if not args.memory_home and len(memory_namespaces) > memory_namespace_limit:
        raise SystemExit(
            "filesystem memory namespace cap cannot fit this batch: "
            f"requested={len(memory_namespaces)}, cap={memory_namespace_limit}. "
            "Use a fresh run-local memory root or raise "
            "--memory-max-namespaces to an intentional finite bound."
        )
    manifest = plan_manifest(
        plan,
        config=config,
        run_id=args.run_id,
        parallel=auth_lanes.parallel,
        run_units=units,
    )
    manifest["chains_json"] = str(chains_path)
    manifest["memory_home"] = str(memory_home)
    manifest["memory_max_namespaces"] = memory_namespace_limit
    manifest["provider_auth"] = auth_lanes.as_manifest()

    planned_rows = [row for unit in units for row in unit.rows]
    output = chain_runner.prepare_batch_output(
        run_root=run_root,
        run_id=args.run_id,
        rows=planned_rows,
        manifest=manifest,
        model_name=config.model_name,
        dataset_name=config.dataset_name,
    )

    _print_plan_banner(
        args,
        config=config,
        units=units,
        parallel=auth_lanes.parallel,
        memory_home=memory_home,
    )
    if args.plan_only:
        print(f"plan written: {output.batch_dir / 'experiment.json'}")
        print(f"instances written: {output.instances_json}")
        return

    chain_runner.validate_provider_envs(
        auth_lanes.slots,
        api_kind=config.api_kind,
    )

    wheelhouse = Path(args.wheelhouse or harness.DEFAULT_PRO_WHEELHOUSE).resolve()
    harness.prepare_wheelhouse_for_run(
        wheelhouse, prepare_all=args.prepare_wheelhouse, extras=()
    )
    host_memory = (
        FilesystemMemory(
            root=memory_home,
            limits=FilesystemMemoryLimits(
                max_namespaces_per_root=memory_namespace_limit,
            ),
        )
        if memory_namespaces
        else None
    )
    if host_memory is not None:
        # Child-only container mounts cannot maintain sibling namespaces. Run
        # the same simple per-namespace cleanup from the host before and after
        # the batch, under the lock directory shared with containers.
        host_memory.maintain()
        requested_namespaces = tuple(memory_namespaces.values())
        if not host_memory.admit_namespaces(requested_namespaces):
            raise SystemExit(
                "filesystem memory namespace admission refused: "
                f"root={memory_home}, requested={len(requested_namespaces)}, "
                f"cap={memory_namespace_limit}. Use a fresh --memory-home or "
                "raise --memory-max-namespaces to an intentional finite bound."
            )
    suite = SwebenchSuite(
        dataset_name=config.dataset_name,
        namespace="swebench",
        network_mode=args.network_mode,
        mem_limit=args.mem_limit,
    )
    memory_backends: dict[str, LocalDockerBackend] = {}
    for unit in units:
        if not unit.memory_enabled:
            continue
        namespace = memory_namespaces[unit.chain_id]
        host_home = memory_home / namespace
        container_mount = f"{DEFAULT_MEMORY_CONTAINER_HOME}/{namespace}"
        memory_backends[unit.chain_id] = LocalDockerBackend(
            pull=args.pull,
            keep_container=args.keep_container,
            wheelhouse=wheelhouse,
            uv_binary=args.uv_binary or None,
            docker_timeout_s=args.docker_timeout_seconds,
            memory_home=host_home,
            memory_mount=container_mount,
            memory_env_home=DEFAULT_MEMORY_CONTAINER_HOME,
            memory_lock_dir=memory_home / ".memory-lock",
            force_existing=args.force,
        )
    plain_backend = LocalDockerBackend(
        pull=args.pull,
        keep_container=args.keep_container,
        wheelhouse=wheelhouse,
        uv_binary=args.uv_binary or None,
        docker_timeout_s=args.docker_timeout_seconds,
        force_existing=args.force,
    )
    store = LocalDirStore(run_root)

    skipped_instances: list[dict[str, Any]] = []

    def run_chain(unit: MemoryChain, auth_env: str) -> dict[str, Any]:
        return _run_chain_with_slot(
            unit,
            args=args,
            config=config,
            provider_auth_env=auth_env,
            run_root=run_root,
            suite=suite,
            backend=(
                memory_backends[unit.chain_id] if unit.memory_enabled else plain_backend
            ),
            memory_name=(
                memory_namespaces[unit.chain_id] if unit.memory_enabled else ""
            ),
            store=store,
            predictions=output.predictions,
        )

    def chain_done(chain_id: str, result: dict[str, Any]) -> None:
        skipped_instances.extend(result.get("skipped_records", []))
        print(
            f"[DONE] {chain_id}: {result['instances']} instance(s), "
            f"{result['errors']} error(s), {result['skipped']} skipped, "
            f"memory={'on' if result['memory_enabled'] else 'off'}",
            flush=True,
        )

    failures = chain_runner.run_auth_lanes(
        units,
        lanes=auth_lanes,
        chain_id=lambda unit: unit.chain_id,
        worker=run_chain,
        on_done=chain_done,
    )

    if host_memory is not None:
        host_memory.maintain()

    if skipped_instances:
        skipped_path = output.batch_dir / "skipped_instances.jsonl"
        write_jsonl_atomic(skipped_path, skipped_instances)
        print(f"wrote {len(skipped_instances)} skipped instances: {skipped_path}")

    predictions = output.predictions.write()
    print(f"wrote {len(predictions)} predictions: {output.predictions.path}")

    if args.run_official_eval:
        chain_runner.run_official_eval(
            predictions_path=output.predictions.path,
            instances_json=output.instances_json,
            run_id=args.run_id,
            max_workers=auth_lanes.parallel,
        )

    if failures:
        chain_runner.write_json_atomic(
            output.batch_dir / "chain_failures.json", failures
        )
        raise SystemExit(f"{len(failures)} chain(s) failed")


def _experiment_config_from_args(
    args: argparse.Namespace, *, api_kind: str
) -> ProMemoryChainConfig:
    return ProMemoryChainConfig(
        dataset_name=args.dataset_name,
        split=args.split,
        model=(os.environ.get(OPENAI_MODEL_ENV) or "").strip(),
        api_kind=api_kind,
        reasoning_effort=(
            (os.environ.get(REASONING_EFFORT_ENV) or "").strip()
            or (os.environ.get(OPENAI_REASONING_EFFORT_ENV) or "").strip()
        ),
        max_turns=args.max_turns,
        agent_flavor=args.agent_flavor,
        memory=bool(args.memory),
        singleton_memory=bool(args.singleton_memory),
        model_name=model_name_for_config(
            agent_flavor=args.agent_flavor,
            memory=bool(args.memory),
            singleton_memory=bool(args.singleton_memory),
        ),
    )


def _resolve_run_id(
    value: str | None,
    config: ProMemoryChainConfig,
    *,
    now: datetime | None = None,
) -> str:
    explicit = str(value).strip() if value is not None else ""
    if explicit:
        return explicit
    memory = "memory" if config.memory else "nomemory"
    agent = "" if config.agent_flavor == "bash" else f"-{config.agent_flavor}"
    prefix = f"pro-memory-chain{agent}-{memory}"
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{timestamp}"


def _filter_raw_chains(
    chains: list[RawIssueChain], *, repos: list[str]
) -> list[RawIssueChain]:
    """Keep manifest chains in the same repo scope as the selected rows."""

    if not repos:
        return list(chains)
    requested = set(repos)
    return [chain for chain in chains if chain.repo in requested]


def _resolve_memory_home(args: argparse.Namespace, *, run_root: Path) -> Path:
    if args.memory_home:
        return Path(args.memory_home).expanduser().resolve()
    return (run_root / args.run_id / "memory").resolve()


def _resolve_memory_namespace_limit(
    args: argparse.Namespace,
    *,
    requested_count: int,
) -> int:
    """Choose a finite namespace cap without breaking a planned run-local batch."""

    override = args.memory_max_namespaces
    if override is not None:
        if override <= 0:
            raise SystemExit("--memory-max-namespaces must be positive")
        return override
    default = FilesystemMemoryLimits().max_namespaces_per_root
    if args.memory_home:
        # A reused root can contain unknown historical namespaces, so retain the
        # conservative default unless the operator explicitly raises the cap.
        return default
    # A run-local root is new and its complete namespace set is already known.
    # Fit that finite set exactly instead of making --singleton-memory fail on
    # full-split plans that intentionally contain more than 128 singletons.
    return max(default, requested_count)


def _print_plan_banner(
    args: argparse.Namespace,
    *,
    config: ProMemoryChainConfig,
    units: list[MemoryChain],
    parallel: int,
    memory_home: Path,
) -> None:
    chain_units = [unit for unit in units if not unit.is_singleton]
    singleton_units = [unit for unit in units if unit.is_singleton]
    longest = max((unit.length for unit in chain_units), default=0)
    print("=== SWE-bench Pro memory-chain experiment ===")
    print(f"run_id: {args.run_id}")
    print(f"run_units: {len(units)}")
    print(f"chains: {len(chain_units)} (longest {longest})")
    print(f"singletons: {len(singleton_units)}")
    print(f"instances: {sum(unit.length for unit in units)}")
    print(f"parallel: {parallel}")
    print(f"memory: {'on' if config.memory else 'off'} home={memory_home}")
    print(f"singleton_memory: {'on' if config.singleton_memory else 'off'}")
    print(
        f"model: {config.model} api_kind={config.api_kind} "
        f"reasoning={config.reasoning_effort}"
    )
    print(f"agent: {config.agent_flavor}")
    print("")


def _run_chain_with_slot(
    unit: MemoryChain,
    *,
    args: argparse.Namespace,
    config: ProMemoryChainConfig,
    provider_auth_env: str,
    run_root: Path,
    suite: SwebenchSuite,
    backend: LocalDockerBackend,
    memory_name: str,
    store: LocalDirStore,
    predictions: chain_runner.PredictionWriter,
) -> dict[str, Any]:
    errors = 0
    skipped_records: list[dict[str, Any]] = []

    for position, instance in enumerate(unit.rows, start=1):
        instance_id = str(instance["instance_id"])
        paths = prepare_run_directory(
            run_root=run_root, run_id=args.run_id, instance_id=instance_id
        )
        container = container_name(
            "swebench_pro_memory",
            instance_id,
            args.run_id,
            namespace=unit.chain_id,
        )
        result: dict[str, Any]
        try:
            provider_env = _provider_env_for_instance(
                provider_auth_env,
                api_kind=config.api_kind,
                agent_flavor=config.agent_flavor,
                memory_name=memory_name,
                memory_run_id=f"{position:03d}_{instance_id}",
            )
            artifacts = run_suite_instance(
                suite=suite,
                instance=instance,
                backend=backend,
                store=store,
                run_root=run_root,
                run_id=args.run_id,
                api_kind=config.api_kind,
                max_turns=config.max_turns,
                provider_env=provider_env,
                wheelhouse_mount=harness.DEFAULT_WHEELHOUSE_MOUNT,
                name=container,
            )
            if artifacts.logs:
                print(
                    artifacts.logs,
                    end="" if artifacts.logs.endswith("\n") else "\n",
                    flush=True,
                )
            result = _load_result_or_error(
                paths.output_dir / RESULT_KEY.split("/", 1)[1],
                instance_id=instance_id,
                unit=unit,
                provider_auth_env=provider_auth_env,
                error=(
                    ""
                    if artifacts.status_code == 0
                    else f"container exited with status {artifacts.status_code}"
                ),
            )
            if artifacts.status_code != 0:
                result["status"] = "error"
                result["error"] = result.get("error") or (
                    f"container exited with status {artifacts.status_code}"
                )
        except Exception as exc:
            result = _load_result_or_error(
                paths.output_dir / RESULT_KEY.split("/", 1)[1],
                instance_id=instance_id,
                unit=unit,
                provider_auth_env=provider_auth_env,
                error=f"{type(exc).__name__}: {exc}",
            )
            result["status"] = "error"
            result["error"] = result.get("error") or f"{type(exc).__name__}: {exc}"
            print(
                f"[{unit.chain_id} #{position}/{unit.length}] {instance_id}: "
                f"{result['error']}",
                flush=True,
            )

        result.setdefault("agent_flavor", config.agent_flavor)
        result.setdefault("chain_id", unit.chain_id)
        result.setdefault("chain_source", unit.source)
        result.setdefault("memory_enabled", unit.memory_enabled)
        result.setdefault("provider_auth_env", provider_auth_env)
        if result.get("status") == "error":
            errors += 1
        if result.get("status") == "skipped":
            skipped_records.append(
                {
                    "chain_id": unit.chain_id,
                    "chain_source": unit.source,
                    "repo": unit.repo,
                    "instance_id": instance_id,
                    "position": position,
                    "reason": str(result.get("skip_reason") or ""),
                    "error": str(result.get("error") or ""),
                }
            )

        chain_runner.write_json_atomic(paths.output_dir / "result.json", result)
        predictions.write()

    chain_dir = (
        run_root / args.run_id / "_memory_chains" / safe_path_part(unit.chain_id)
    )
    if skipped_records:
        write_jsonl_atomic(chain_dir / "skipped_instances.jsonl", skipped_records)
    chain_runner.write_json_atomic(
        chain_dir / "summary.json",
        {
            "chain_id": unit.chain_id,
            "repo": unit.repo,
            "source": unit.source,
            "instances": unit.length,
            "errors": errors,
            "skipped": len(skipped_records),
            "memory_enabled": unit.memory_enabled,
            "provider_auth_env": provider_auth_env,
            "agent_flavor": config.agent_flavor,
            "instance_ids": unit.instance_ids,
        },
    )
    return {
        "instances": unit.length,
        "errors": errors,
        "skipped": len(skipped_records),
        "memory_enabled": unit.memory_enabled,
        "skipped_records": skipped_records,
    }


def _provider_env_for_instance(
    auth_env: str,
    *,
    api_kind: str,
    agent_flavor: str,
    memory_name: str,
    memory_run_id: str,
) -> dict[str, str]:
    """Build the in-container env for one instance (provider + flavor + memory)."""

    env = chain_runner.provider_env_for_auth_env(auth_env, api_kind=api_kind)
    env[AGENT_FLAVOR_ENV] = agent_flavor
    # SAL_MEMORY_HOME is set by the backend's memory mount; NAME/RUN_ID are the
    # per-instance scope. An empty NAME (singleton on the plain backend) is not
    # set, so the container has no SAL_MEMORY_HOME and builds no memory hooks.
    if memory_name:
        env[MEMORY_NAME_ENV] = memory_name
        env[MEMORY_RUN_ID_ENV] = memory_run_id
    return env


def _load_result_or_error(
    path: Path,
    *,
    instance_id: str,
    unit: MemoryChain,
    provider_auth_env: str,
    error: str,
) -> dict[str, Any]:
    if path.exists():
        return chain_runner.read_json(path)
    return {
        "model_patch": "",
        "instance_id": instance_id,
        "repo": unit.repo,
        "chain_id": unit.chain_id,
        "chain_source": unit.source,
        "memory_enabled": unit.memory_enabled,
        "provider_auth_env": provider_auth_env,
        "status": "error" if error else "ok",
        "error": error,
        "skip_reason": "",
    }


if __name__ == "__main__":
    main()
