"""Run the full SWE-bench Pro split as memory-sharing issue chains.

This is the *memory-based* peer of ``run_swebench_pro_repo_chains.py``. It
borrows the filesystem-memory chain shape from the ``mini-memory`` project: each
issue runs as an ordinary, isolated SWE-bench Pro instance in a *fresh* agent
context (no transcript/handoff is carried in-context), and the only thing that
crosses instance boundaries is Simple Agent Lab filesystem memory scoped per
chain. Within a chain the memory dir is shared (``SAL_MEMORY_NAME=<chain_id>``);
the run-end distiller updates it, so a later issue in the same chain can reuse an
earlier one's lessons.

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
      --max-chains 1 --limit 2 --max-turns 5 --skip-official-eval

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
the vendored flat chain-nodes JSONL or another flat JSONL / nested JSON manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evals.swebench import harness  # noqa: E402
from evals.swebench.evaluate_predictions import predictions_from_run_dirs  # noqa: E402
from evals.swebench.pro_memory_chain import (  # noqa: E402
    MEMORY_CHAIN_AGENT_FLAVORS,
    MemoryChain,
    ProMemoryChainConfig,
    expand_auth_slots,
    lane_auth_slots,
    load_issue_chains,
    model_name_for_config,
    plan_manifest,
    plan_memory_chains,
)
from evals.swebench.suite import SwebenchSuite  # noqa: E402
from simple_agent_lab.agent_flavors import AGENT_FLAVOR_ENV  # noqa: E402
from simple_agent_lab.evals import LocalDirStore, LocalDockerBackend  # noqa: E402
from simple_agent_lab.evals.protocols import (  # noqa: E402
    MEMORY_NAME_ENV,
    MEMORY_RUN_ID_ENV,
    RESULT_KEY,
)
from simple_agent_lab.evals.runner import (  # noqa: E402
    container_name,
    prepare_new_run_directory,
    prepare_run_directory,
    run_suite_instance,
)
from simple_agent_lab.llm.env import (  # noqa: E402
    API_KIND_ENV,
    OPENAI_AUTH_ENV,
    OPENAI_ENV,
    OPENAI_MODEL_ENV,
    OPENAI_REASONING_EFFORT_ENV,
    REASONING_EFFORT_ENV,
)
from simple_agent_lab.trace import write_jsonl_atomic  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all", action="store_true", help="Run the whole selected split."
    )
    parser.add_argument(
        "--repos",
        nargs="*",
        default=[],
        help="Optional exact repo names to keep, e.g. NodeBB/NodeBB.",
    )
    parser.add_argument(
        "--chains-json",
        default=None,
        help=(
            "Required chain manifest: either a flat chain-nodes JSONL "
            "(one node per line) or a nested issue-chains JSON. To use the "
            "vendored deep manifest, pass "
            "evals/swebench/data/swe_bench_pro_chain_experiment_nodes_deep.jsonl."
        ),
    )
    parser.add_argument("--dataset-name", default=ProMemoryChainConfig.dataset_name)
    parser.add_argument("--split", default=ProMemoryChainConfig.split)
    parser.add_argument("--instance-json", help="Use a local JSON/JSONL dataset file.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the total number of dataset instances (smoke runs).",
    )
    parser.add_argument(
        "--max-chains",
        type=int,
        default=None,
        help="Keep only the first N multi-issue chains after longest-first sort.",
    )
    parser.add_argument(
        "--run-root",
        default=str(harness.DEFAULT_PRO_RUN_ROOT),
        help="Output root. Defaults to evals/out/swebench_pro.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Output run id. Defaults to a timestamped id derived from the arm.",
    )
    parser.add_argument(
        "--model", default=None, help="Override OPENAI_MODEL from the environment/.env."
    )
    parser.add_argument(
        "--api-kind",
        default=ProMemoryChainConfig.api_kind,
        help="LLM adapter API kind. Defaults to openai-responses for this experiment.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="Override REASONING_EFFORT from the environment/.env.",
    )
    parser.add_argument("--max-turns", type=int, default=ProMemoryChainConfig.max_turns)
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
            "Host directory bind-mounted read-write as the memory root. Defaults "
            "to <run-root>/<run-id>/memory so each run's chains build memory from "
            "a clean slate."
        ),
    )
    parser.add_argument(
        "--parallel",
        default="slots",
        help="'slots' to size the pool from --provider-auth-envs, or an integer.",
    )
    parser.add_argument(
        "--provider-auth-envs",
        default=None,
        help=(
            "Comma-separated auth env slots, e.g. "
            "OPENAI_AUTH_TOKEN:12,OPENAI_AUTH_TOKEN2:11. Each concurrent lane "
            "holds one slot; slots cycle if there are more lanes than slots. "
            "Defaults to a single OPENAI_AUTH_TOKEN slot."
        ),
    )
    parser.add_argument("--dotenv", default=str(ROOT / ".env"))
    parser.add_argument("--network-mode", default="host")
    parser.add_argument("--mem-limit", default="16g")
    parser.add_argument("--wheelhouse", default=None)
    parser.add_argument("--prepare-wheelhouse", action="store_true")
    parser.add_argument("--uv-binary", default=harness.DEFAULT_UV_BINARY)
    parser.add_argument(
        "--pull",
        default="missing",
        choices=("missing", "always", "never"),
        help="Docker image pull policy for SWE-bench Pro instance images.",
    )
    parser.add_argument("--keep-container", action="store_true")
    parser.add_argument("--docker-timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--skip-official-eval",
        action="store_true",
        help="Only run inference and collect predictions.",
    )
    parser.add_argument(
        "--run-official-eval",
        action="store_true",
        help="Run the official SWE-bench Pro evaluator after inference.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write the chain plan and exit before provider or Docker setup.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove any leftover deterministic container before starting it.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.all and not args.repos and not args.instance_json:
        raise SystemExit("Pass --all, --repos REPO..., or --instance-json PATH.")
    _chains_json_path(args)

    harness.load_dotenv(args.dotenv)
    _apply_provider_env_overrides(args)
    api_kind = harness.resolve_api_kind(args.api_kind)
    config = _experiment_config_from_args(args, api_kind=api_kind)
    args.run_id = _resolve_run_id(args.run_id, config)
    run_root = Path(args.run_root)

    rows = _load_rows(args)
    raw_chains = _load_chains(args)
    plan = plan_memory_chains(
        rows,
        raw_chains,
        memory=config.memory,
        singleton_memory=config.singleton_memory,
    )
    units = _select_units(plan.chains, args)
    if not units:
        raise SystemExit("No SWE-bench Pro instances selected.")

    expanded_slots = _expand_auth_slots(args.provider_auth_envs)
    parallel = _resolve_parallel(args.parallel, slot_count=len(expanded_slots))
    lane_slots = lane_auth_slots(expanded_slots, parallel)

    memory_home = _resolve_memory_home(args, run_root=run_root)
    manifest = plan_manifest(
        plan,
        config=config,
        run_id=args.run_id,
        parallel=parallel,
        run_units=units,
    )
    manifest["chains_json"] = str(_chains_json_path(args))
    manifest["memory_home"] = str(memory_home)
    manifest["provider_auth"] = {
        "spec": args.provider_auth_envs or f"{OPENAI_ENV.auth}:1",
        "lane_slots": lane_slots,
    }

    planned_rows = [row for unit in units for row in unit.rows]
    expected_instance_ids = tuple(str(row["instance_id"]) for row in planned_rows)
    try:
        batch_dir = prepare_new_run_directory(run_root=run_root, run_id=args.run_id)
    except FileExistsError as exc:
        raise SystemExit(str(exc)) from None
    predictions_path = batch_dir / f"{args.run_id}_predictions.jsonl"
    instances_json = batch_dir / "instances.jsonl"
    _write_jsonl_records(instances_json, planned_rows)
    _write_json(batch_dir / "experiment.json", manifest)

    _print_plan_banner(
        args, config=config, units=units, parallel=parallel, memory_home=memory_home
    )
    if args.plan_only:
        print(f"plan written: {batch_dir / 'experiment.json'}")
        print(f"instances written: {instances_json}")
        return

    wheelhouse = _resolve_wheelhouse(args)
    harness.prepare_wheelhouse_for_run(
        wheelhouse, prepare_all=args.prepare_wheelhouse, extras=()
    )
    suite = SwebenchSuite(
        dataset_name=config.dataset_name,
        namespace="swebench",
        network_mode=args.network_mode,
        mem_limit=args.mem_limit,
    )
    memory_backend = LocalDockerBackend(
        pull=args.pull,
        keep_container=args.keep_container,
        wheelhouse=wheelhouse,
        uv_binary=args.uv_binary or None,
        docker_timeout_s=args.docker_timeout_seconds,
        memory_home=memory_home,
    )
    plain_backend = LocalDockerBackend(
        pull=args.pull,
        keep_container=args.keep_container,
        wheelhouse=wheelhouse,
        uv_binary=args.uv_binary or None,
        docker_timeout_s=args.docker_timeout_seconds,
    )
    store = LocalDirStore(run_root)
    prediction_lock = threading.Lock()
    slot_pool: queue.Queue[str] = queue.Queue()
    for slot in lane_slots:
        slot_pool.put(slot)

    failures: list[dict[str, str]] = []
    skipped_instances: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(
                _run_chain,
                unit,
                args=args,
                config=config,
                slot_pool=slot_pool,
                run_root=run_root,
                suite=suite,
                memory_backend=memory_backend,
                plain_backend=plain_backend,
                store=store,
                predictions_path=predictions_path,
                prediction_lock=prediction_lock,
                expected_instance_ids=expected_instance_ids,
            ): unit.chain_id
            for unit in units
        }
        for future in as_completed(futures):
            chain_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failures.append(
                    {"chain_id": chain_id, "error": f"{type(exc).__name__}: {exc}"}
                )
                print(f"[FAIL] {chain_id}: {type(exc).__name__}: {exc}", flush=True)
            else:
                skipped_instances.extend(result.get("skipped_records", []))
                print(
                    f"[DONE] {chain_id}: {result['instances']} instance(s), "
                    f"{result['errors']} error(s), {result['skipped']} skipped, "
                    f"memory={'on' if result['memory_enabled'] else 'off'}",
                    flush=True,
                )

    if skipped_instances:
        skipped_path = batch_dir / "skipped_instances.jsonl"
        _write_jsonl_records(skipped_path, skipped_instances)
        print(f"wrote {len(skipped_instances)} skipped instances: {skipped_path}")

    predictions = predictions_from_run_dirs(
        run_root,
        run_id=args.run_id,
        model_name=config.model_name,
        dataset_name=config.dataset_name,
        expected_instance_ids=expected_instance_ids,
    )
    write_jsonl_atomic(predictions_path, predictions)
    print(f"wrote {len(predictions)} predictions: {predictions_path}")

    if args.run_official_eval and not args.skip_official_eval:
        _run_official_eval(
            predictions_path=predictions_path,
            instances_json=instances_json,
            run_id=args.run_id,
            max_workers=parallel,
        )

    if failures:
        _write_json(batch_dir / "chain_failures.json", failures)
        raise SystemExit(f"{len(failures)} chain(s) failed")


def _apply_provider_env_overrides(args: argparse.Namespace) -> None:
    """Apply explicit CLI model/reasoning overrides onto the process env.

    Model and reasoning only override when passed explicitly, so ``.env`` wins
    otherwise (mirrors the repo-chain runner). The flavor reaches the container
    per instance through ``provider_env``, not the host env.
    """

    model = str(args.model).strip() if args.model is not None else ""
    if model:
        os.environ[OPENAI_MODEL_ENV] = model
    reasoning_effort = (
        str(args.reasoning_effort).strip() if args.reasoning_effort is not None else ""
    )
    if reasoning_effort:
        os.environ[REASONING_EFFORT_ENV] = reasoning_effort


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
        task_tool=args.agent_flavor in ("bash_task", "bash_task_read"),
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


def _load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.instance_json:
        return harness._load_instance_records(Path(args.instance_json))
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Install SWE-bench extras first: uv sync --extra swebench"
        ) from exc
    rows = [dict(row) for row in load_dataset(args.dataset_name, split=args.split)]
    if args.repos:
        requested = set(args.repos)
        rows = [row for row in rows if str(row.get("repo") or "") in requested]
    return rows


def _load_chains(args: argparse.Namespace):
    path = _chains_json_path(args)
    if not path.exists():
        raise SystemExit(
            f"--chains-json not found: {path}. Pass a chain-nodes JSONL "
            "or issue-chains JSON."
        )
    return load_issue_chains(path)


def _chains_json_path(args: argparse.Namespace) -> Path:
    value = str(args.chains_json or "").strip()
    if not value:
        raise SystemExit(
            "Pass --chains-json PATH for the exact chain manifest to use, e.g. "
            "evals/swebench/data/swe_bench_pro_chain_experiment_nodes_deep.jsonl."
        )
    return Path(value).expanduser()


def _select_units(
    chains: tuple[MemoryChain, ...], args: argparse.Namespace
) -> list[MemoryChain]:
    """Apply --max-chains / --limit while keeping the longest-first order."""

    units = list(chains)
    if args.max_chains is not None:
        kept_chains = 0
        selected: list[MemoryChain] = []
        for unit in units:
            if unit.is_singleton:
                selected.append(unit)
                continue
            if kept_chains < args.max_chains:
                selected.append(unit)
                kept_chains += 1
        units = selected
    if args.limit is not None:
        limited: list[MemoryChain] = []
        seen = 0
        for unit in units:
            if seen >= args.limit:
                break
            take = min(unit.length, args.limit - seen)
            limited.append(
                replace(unit, rows=unit.rows[:take]) if take < unit.length else unit
            )
            seen += take
        units = limited
    return units


def _expand_auth_slots(spec: str | None) -> list[str]:
    try:
        return expand_auth_slots(spec, default_env=OPENAI_ENV.auth)
    except ValueError as exc:
        raise SystemExit(f"--provider-auth-envs {exc}") from None


def _resolve_parallel(value: str, *, slot_count: int) -> int:
    if value == "slots":
        return max(1, slot_count)
    try:
        parsed = int(value)
    except ValueError:
        raise SystemExit("--parallel must be 'slots' or a positive integer") from None
    if parsed <= 0:
        raise SystemExit("--parallel must be positive")
    return parsed


def _resolve_memory_home(args: argparse.Namespace, *, run_root: Path) -> Path:
    if args.memory_home:
        return Path(args.memory_home).expanduser().resolve()
    return (run_root / args.run_id / "memory").resolve()


def _resolve_wheelhouse(args: argparse.Namespace) -> Path | None:
    wheelhouse_arg = args.wheelhouse or str(harness.DEFAULT_PRO_WHEELHOUSE)
    return Path(wheelhouse_arg).resolve() if wheelhouse_arg else None


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


def _run_chain(
    unit: MemoryChain,
    *,
    args: argparse.Namespace,
    config: ProMemoryChainConfig,
    slot_pool: queue.Queue[str],
    run_root: Path,
    suite: SwebenchSuite,
    memory_backend: LocalDockerBackend,
    plain_backend: LocalDockerBackend,
    store: LocalDirStore,
    predictions_path: Path,
    prediction_lock: threading.Lock,
    expected_instance_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Run one run unit's instances in order, holding one auth slot for its life."""

    provider_auth_env = slot_pool.get()
    try:
        return _run_chain_with_slot(
            unit,
            args=args,
            config=config,
            provider_auth_env=provider_auth_env,
            run_root=run_root,
            suite=suite,
            backend=memory_backend if unit.memory_enabled else plain_backend,
            store=store,
            predictions_path=predictions_path,
            prediction_lock=prediction_lock,
            expected_instance_ids=expected_instance_ids,
        )
    finally:
        slot_pool.put(provider_auth_env)


def _run_chain_with_slot(
    unit: MemoryChain,
    *,
    args: argparse.Namespace,
    config: ProMemoryChainConfig,
    provider_auth_env: str,
    run_root: Path,
    suite: SwebenchSuite,
    backend: LocalDockerBackend,
    store: LocalDirStore,
    predictions_path: Path,
    prediction_lock: threading.Lock,
    expected_instance_ids: tuple[str, ...],
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
        provider_env = _provider_env_for_instance(
            provider_auth_env,
            api_kind=config.api_kind,
            agent_flavor=config.agent_flavor,
            memory_name=unit.chain_id if unit.memory_enabled else "",
            memory_run_id=f"{position:03d}_{instance_id}",
        )
        result: dict[str, Any]
        try:
            if args.force:
                _force_remove_container(container)
            artifacts = run_suite_instance(
                suite=suite,
                instance=instance,
                backend=backend,
                store=store,
                run_root=run_root,
                run_id=args.run_id,
                provider="openai",
                api_kind=config.api_kind,
                max_turns=config.max_turns,
                provider_env=provider_env,
                package_extras=(),
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

        _write_json(paths.output_dir / "result.json", result)
        _write_incremental_predictions(
            predictions_path=predictions_path,
            run_root=run_root,
            run_id=args.run_id,
            model_name=config.model_name,
            dataset_name=config.dataset_name,
            expected_instance_ids=expected_instance_ids,
            lock=prediction_lock,
        )

    chain_dir = (
        run_root / args.run_id / "_memory_chains" / _safe_path_part(unit.chain_id)
    )
    chain_dir.mkdir(parents=True, exist_ok=True)
    if skipped_records:
        _write_jsonl_records(chain_dir / "skipped_instances.jsonl", skipped_records)
    _write_json(
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
        "provider_auth_env": provider_auth_env,
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

    env: dict[str, str] = {}
    for name in harness.OPENAI_PASSTHROUGH_ENVS:
        if name == OPENAI_AUTH_ENV:
            continue
        value = os.environ.get(name)
        if value:
            env[name] = value.strip()
    for name in ("NO_PROXY", "no_proxy"):
        value = os.environ.get(name)
        if value:
            env[name] = value

    model = (os.environ.get(OPENAI_MODEL_ENV) or "").strip()
    token = (os.environ.get(auth_env) or "").strip()
    missing = []
    if not model:
        missing.append(OPENAI_MODEL_ENV)
    if not token:
        missing.append(auth_env)
    if missing:
        raise SystemExit(
            "Missing required env vars for SWE-bench Pro container run: "
            + ", ".join(missing)
        )

    env[OPENAI_MODEL_ENV] = model
    env[OPENAI_AUTH_ENV] = token
    env[API_KIND_ENV] = api_kind
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
        return _read_json(path)
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_incremental_predictions(
    *,
    predictions_path: Path,
    run_root: Path,
    run_id: str,
    model_name: str,
    dataset_name: str,
    expected_instance_ids: tuple[str, ...],
    lock: threading.Lock | None,
) -> None:
    def write() -> None:
        predictions = predictions_from_run_dirs(
            run_root,
            run_id=run_id,
            model_name=model_name,
            dataset_name=dataset_name,
            expected_instance_ids=expected_instance_ids,
        )
        write_jsonl_atomic(predictions_path, predictions)

    if lock is None:
        write()
        return
    with lock:
        write()


def _run_official_eval(
    *,
    predictions_path: Path,
    instances_json: Path,
    run_id: str,
    max_workers: int,
) -> None:
    command = [
        sys.executable,
        str(ROOT / "evals/swebench/evaluate_predictions.py"),
        "--pro",
        "--run-official",
        "--predictions",
        str(predictions_path),
        "--instances",
        str(instances_json),
        "--run-id",
        run_id,
        "--max-workers",
        str(max_workers),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def _force_remove_container(name: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", name],
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    try:
        tmp.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _write_jsonl_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _safe_path_part(value: str) -> str:
    return "".join(char if char.isalnum() or char in "_.-" else "_" for char in value)


if __name__ == "__main__":
    main()
