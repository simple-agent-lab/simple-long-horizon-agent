"""Run SWE-bench Pro as repo-chain continuations over issue-chain units.

This is the experimental runner for studying context compression on long
SWE-bench Pro chains. It borrows the issue-chain planning and longest-first
ordering from ``run_swebench_pro_memory_chains.py``, then runs each planned unit
with the repo-chain continuation runner: the agent runs inside each task's
SWE-bench Pro instance container, and the host passes ``out/chain_state.json``
from one instance to the next within that unit.

The default flavor is ``bash`` with no chain compression. Use ``--task-tool``
to add the task tool and ``--compression-strategy summarize`` to turn on chain
compression.

Example smoke:

    uv run --extra swebench python -m runs.swebench.run_swebench_pro_repo_chains \
      --chains-json evals/swebench/data/swe_bench_pro_chain_experiment_nodes_deep.jsonl \
      --max-chains 1 --limit 2 --max-turns 5

Formal run shape:

    uv run --extra swebench python -m runs.swebench.run_swebench_pro_repo_chains \
      --all \
      --chains-json evals/swebench/data/swe_bench_pro_chain_experiment_nodes_deep.jsonl \
      --provider-auth-envs OPENAI_AUTH_TOKEN:12,OPENAI_AUTH_TOKEN2:11 \
      --api-kind openai-responses \
      --max-turns 250 \
      --run-official-eval

Full trajectories are written by default for this runner. Repo chains can
accumulate hundreds of thousands of prompt tokens, and the full trace records
every model request payload; pass ``--no-write-trajectories`` only for runs
where that storage cost is unacceptable.

Pass ``--chains-json PATH`` explicitly for every experiment run. There is no
default, so the run command records exactly which chain manifest it used.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from evals.swebench import harness
from evals.swebench.pro_memory_chain import (
    MemoryChain,
    RawIssueChain,
    load_issue_chains,
    plan_memory_chains,
)
from evals.swebench import pro_chain_runner as chain_runner
from evals.swebench.pro_repo_chain import (
    ProRepoExperimentConfig,
    group_instances_by_repo,
)
from evals.swebench.suite import SwebenchSuite
from simple_long_horizon_agent.evals import LocalDirStore, LocalDockerBackend
from simple_long_horizon_agent.evals.protocols import RESULT_KEY
from simple_long_horizon_agent.evals.runner import (
    canonical_run_id,
    container_name,
    prepare_run_directory,
    run_suite_instance,
    safe_path_part,
)
from simple_long_horizon_agent.evals.chain import (
    CHAIN_CONFIG_KEY,
    CHAIN_STATE_INPUT_KEY,
    CHAIN_STATE_OUTPUT_KEY,
)
from simple_long_horizon_agent.llm.env import (
    OPENAI_MODEL_ENV,
    OPENAI_REASONING_EFFORT_ENV,
    REASONING_EFFORT_ENV,
)
from simple_long_horizon_agent.trace import write_jsonl_atomic

PRO_REPO_CHAIN_RUNNER_MODULE = "simple_long_horizon_agent.evals.chain"
REPO_CHAIN_AGENT_FLAVORS = ("bash", "loop", "pdr")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    chain_runner.add_common_arguments(
        parser,
        dataset_name=ProRepoExperimentConfig.dataset_name,
        split=ProRepoExperimentConfig.split,
        max_turns=ProRepoExperimentConfig.max_turns,
        api_kind=ProRepoExperimentConfig.api_kind,
    )
    parser.add_argument("--max-repos", type=int, help="Limit repo count after filter.")
    parser.add_argument(
        "--limit-per-repo",
        type=int,
        help="Limit selected instances per repo after memory-chain ordering.",
    )
    parser.add_argument(
        "--agent-flavor",
        default=ProRepoExperimentConfig.agent_flavor,
        choices=REPO_CHAIN_AGENT_FLAVORS,
        help=(
            "Agent flavor for each repo-chain unit. Repo chains intentionally "
            "exclude read-tool flavors; use --task-tool to add task."
        ),
    )
    parser.add_argument(
        "--task-tool",
        action=argparse.BooleanOptionalAction,
        default=ProRepoExperimentConfig.task_tool,
        help="Add the task tool to the selected chain agent/solver (default: disabled).",
    )
    parser.add_argument(
        "--compression-strategy",
        default=ProRepoExperimentConfig.compression_strategy,
        choices=("summarize", "none"),
        help=(
            "Context strategy for the repo-chain continuation "
            "(default: none). Pass 'summarize' to turn on chain compression."
        ),
    )
    parser.add_argument(
        "--handoff",
        action=argparse.BooleanOptionalAction,
        default=ProRepoExperimentConfig.handoff,
        help=(
            "When context reaches --context-window-tokens, have the model write "
            "a handoff document immediately. If the current instance is done, "
            "the next instance starts in a fresh window seeded with that "
            "handoff; if it is not done, the SAME instance keeps working in a "
            "fresh window seeded with the handoff (default: enabled). Ignored "
            "when --compression-strategy summarize."
        ),
    )
    parser.add_argument(
        "--context-window-tokens",
        type=int,
        default=ProRepoExperimentConfig.context_window_tokens,
        help=(
            "Handoff trigger: reset the repo-chain window once the estimated "
            "active context reaches this many tokens, mid-instance if needed "
            "(default: 217600 = 272000 * 0.8, the same trigger as the summarize "
            "--threshold-tokens so handoff and compression are compared fairly; "
            "the real 272000 window leaves headroom above the trigger)."
        ),
    )
    parser.add_argument(
        "--threshold-tokens",
        type=int,
        default=ProRepoExperimentConfig.threshold_tokens,
    )
    parser.add_argument(
        "--keep-recent", type=int, default=ProRepoExperimentConfig.keep_recent
    )
    parser.add_argument(
        "--write-trajectories",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write the full per-instance out/trajectory.jsonl file "
            "(default: enabled). Use --no-write-trajectories to suppress them."
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
    )
    raw_chains = load_issue_chains(chains_path)
    chains, manifest = _plan_groups(
        rows,
        raw_chains=raw_chains,
        chains_path=chains_path,
        args=args,
        config=config,
    )
    if not chains:
        raise SystemExit("No SWE-bench Pro instances selected.")
    auth_lanes = chain_runner.resolve_auth_lanes(args.provider_auth_envs, args.parallel)
    manifest["resolved_parallel"] = auth_lanes.parallel
    manifest["provider_auth"] = auth_lanes.as_manifest()

    planned_rows = [row for chain in chains.values() for row in chain.rows]
    output = chain_runner.prepare_batch_output(
        run_root=run_root,
        run_id=args.run_id,
        rows=planned_rows,
        manifest=manifest,
    )

    print("=== SWE-bench Pro repo-chain experiment ===")
    print(f"run_id: {args.run_id}")
    print(f"plan_source: {manifest['plan_source']}")
    print(f"chains_json: {manifest['chains_json']}")
    print(f"repos: {manifest['repo_count']}")
    print(f"run_units: {manifest['run_unit_count']}")
    print(f"chains: {manifest['chain_count']} (longest {_longest_chain(manifest)})")
    print(f"singletons: {manifest['singleton_count']}")
    print(f"instances: {sum(len(chain.rows) for chain in chains.values())}")
    print(f"parallel: {auth_lanes.parallel}")
    print(
        "provider_auth: "
        + ", ".join(
            f"{entry['auth_env']} x{entry['lanes']}"
            for entry in chain_runner.provider_auth_slot_summary(
                manifest["provider_auth"]["lane_slots"]
            )
        )
    )
    print(
        f"model: {config.model} api_kind={config.api_kind} "
        f"reasoning={config.reasoning_effort}"
    )
    print(
        f"agent: {config.agent_flavor} tools={'bash,task' if config.task_tool else 'bash'}"
    )
    if config.compression_strategy == "summarize":
        print(
            "compression: summarize "
            f"threshold={config.threshold_tokens} keep_recent={config.keep_recent}"
        )
    else:
        print("compression: none")
    print(f"trajectories: {'full' if args.write_trajectories else 'disabled'}")
    print("")
    if args.plan_only:
        print(f"plan written: {output.batch_dir / 'experiment.json'}")
        print(f"instances written: {output.instances_json}")
        return

    chain_runner.validate_provider_envs(auth_lanes.slots, api_kind=config.api_kind)

    wheelhouse = Path(args.wheelhouse or harness.DEFAULT_PRO_WHEELHOUSE).resolve()
    harness.prepare_wheelhouse_for_run(
        wheelhouse,
        prepare_all=args.prepare_wheelhouse,
        extras=(),
    )
    suite = SwebenchSuite(
        dataset_name=config.dataset_name,
        namespace="swebench",
        network_mode=args.network_mode,
        mem_limit=args.mem_limit,
    )
    backend = LocalDockerBackend(
        pull=args.pull,
        keep_container=args.keep_container,
        wheelhouse=wheelhouse,
        uv_binary=args.uv_binary or None,
        docker_timeout_s=args.docker_timeout_seconds,
        force_existing=args.force,
    )
    store = LocalDirStore(run_root)

    skipped_instances: list[dict[str, Any]] = []

    def run_repo(chain: MemoryChain, auth_env: str) -> dict[str, Any]:
        return _run_repo(
            chain,
            args=args,
            config=config,
            provider_auth_env=auth_env,
            run_root=run_root,
            suite=suite,
            backend=backend,
            store=store,
        )

    def repo_done(chain_id: str, result: dict[str, Any]) -> None:
        skipped_instances.extend(result.get("skipped_records", []))
        handoffs = int(result.get("handoffs", 0) or 0)
        mid_handoffs = int(result.get("mid_instance_handoffs", 0) or 0)
        boundary_handoffs = int(result.get("boundary_handoffs", 0) or 0)
        print(
            f"[DONE] {chain_id}: {result['instances']} instance(s), "
            f"{result['errors']} error(s), {result['skipped']} skipped, "
            f"{handoffs} handoff(s) ({mid_handoffs} mid-instance, "
            f"{boundary_handoffs} boundary)",
            flush=True,
        )

    failures = chain_runner.run_auth_lanes(
        chains.values(),
        lanes=auth_lanes,
        chain_id=lambda chain: chain.chain_id,
        worker=run_repo,
        on_done=repo_done,
    )

    if skipped_instances:
        skipped_path = output.batch_dir / "skipped_instances.jsonl"
        write_jsonl_atomic(skipped_path, skipped_instances)
        print(f"wrote {len(skipped_instances)} skipped instances: {skipped_path}")

    predictions = chain_runner.write_predictions(
        output,
        model_name=config.model_name,
        dataset_name=config.dataset_name,
    )
    print(f"wrote {len(predictions)} predictions: {output.predictions_path}")

    if args.run_official_eval:
        chain_runner.run_official_eval(
            predictions_path=output.predictions_path,
            instances_json=output.instances_json,
            run_id=args.run_id,
            max_workers=auth_lanes.parallel,
        )

    if failures:
        chain_runner.write_json_atomic(
            output.batch_dir / "repo_failures.json", failures
        )
        raise SystemExit(f"{len(failures)} repo chain(s) failed")


def _experiment_config_from_args(
    args: argparse.Namespace, *, api_kind: str
) -> ProRepoExperimentConfig:
    """Build the manifest config from args plus the effective provider env."""

    return ProRepoExperimentConfig(
        dataset_name=args.dataset_name,
        split=args.split,
        model=(os.environ.get(OPENAI_MODEL_ENV) or "").strip(),
        api_kind=api_kind,
        reasoning_effort=(
            (os.environ.get(REASONING_EFFORT_ENV) or "").strip()
            or (os.environ.get(OPENAI_REASONING_EFFORT_ENV) or "").strip()
        ),
        max_turns=args.max_turns,
        context_window_tokens=args.context_window_tokens,
        threshold_tokens=args.threshold_tokens,
        keep_recent=args.keep_recent,
        agent_flavor=args.agent_flavor,
        task_tool=bool(args.task_tool),
        compression_strategy=args.compression_strategy,
        handoff=bool(args.handoff),
        model_name=_model_name_for_mode(
            agent_flavor=args.agent_flavor,
            compression_strategy=args.compression_strategy,
            task_tool=bool(args.task_tool),
        ),
    )


def _model_name_for_mode(
    *, agent_flavor: str, compression_strategy: str, task_tool: bool
) -> str:
    if agent_flavor == "bash" and not task_tool:
        return f"simple-long-horizon-agent-pro-repo-chain-bash-{compression_strategy}"
    task = "task" if task_tool else "bash"
    return f"simple-long-horizon-agent-pro-repo-chain-{agent_flavor}-{task}-{compression_strategy}"


def _resolve_run_id(
    value: str | None,
    config: ProRepoExperimentConfig,
    *,
    now: datetime | None = None,
) -> str:
    explicit = str(value).strip() if value is not None else ""
    if explicit:
        return explicit
    compression = "summarize" if config.compression_strategy == "summarize" else "none"
    agent = "" if config.agent_flavor == "bash" else f"-{config.agent_flavor}"
    task = "-task" if config.task_tool else ""
    prefix = f"pro-repo-chain{agent}{task}-{compression}"
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{timestamp}"


def _plan_groups(
    rows: list[dict[str, Any]],
    *,
    raw_chains: list[RawIssueChain],
    chains_path: Path,
    args: argparse.Namespace,
    config: ProRepoExperimentConfig,
) -> tuple[dict[str, MemoryChain], dict[str, Any]]:
    selected_rows, selected_repos = _selected_rows(rows, args=args)
    selected_raw_chains = [
        chain
        for chain in raw_chains
        if not selected_repos or chain.repo in selected_repos
    ]
    plan = plan_memory_chains(
        selected_rows,
        selected_raw_chains,
        memory=False,
        singleton_memory=False,
    )
    units = chain_runner.select_units(
        plan.chains,
        max_chains=args.max_chains,
        limit=args.limit,
        limit_per_repo=args.limit_per_repo,
    )
    chains: dict[str, MemoryChain] = {}
    order_manifest: list[dict[str, Any]] = []
    print(
        f"[plan] ordering {len(units)} memory-chain unit(s) longest-first",
        flush=True,
    )
    for index, unit in enumerate(units, start=1):
        print(
            f"[plan] {index}/{len(units)} {unit.chain_id}: {unit.length} instance(s)",
            flush=True,
        )
        chains[unit.chain_id] = unit
        order_manifest.append(
            {
                "chain_id": unit.chain_id,
                "repo": unit.repo,
                "source": unit.source,
                "length": unit.length,
                "instance_ids": unit.instance_ids,
            }
        )

    manifest = {
        "schema": "simple-long-horizon-agent.swebench-pro-repo-chain-experiment.v1",
        "plan_source": "memory_issue_chains",
        "chains_json": str(chains_path),
        "run_id": args.run_id,
        "config": config.as_record(),
        "repo_count": len({chain.repo for chain in chains.values()}),
        "run_unit_count": len(units),
        "chain_count": sum(1 for unit in units if not unit.is_singleton),
        "instance_count": sum(chain.length for chain in chains.values()),
        "chain_instance_count": sum(
            unit.length for unit in units if not unit.is_singleton
        ),
        "singleton_count": sum(1 for unit in units if unit.is_singleton),
        "missing_instance_ids": list(plan.missing_instance_ids),
        "duplicate_instance_ids": list(plan.duplicate_instance_ids),
        "chain_length_histogram": _chain_length_histogram(units),
        "per_repo": _per_repo_plan_counts(units),
        "parallel": args.parallel,
        "write_trajectories": bool(args.write_trajectories),
        "order": order_manifest,
    }
    return chains, manifest


def _longest_chain(manifest: Mapping[str, Any]) -> int:
    return max(
        (
            int(entry.get("length") or 0)
            for entry in manifest.get("order", [])
            if entry.get("source") != "singleton"
        ),
        default=0,
    )


def _selected_rows(
    rows: list[dict[str, Any]], *, args: argparse.Namespace
) -> tuple[list[dict[str, Any]], set[str]]:
    groups = group_instances_by_repo(rows)
    if args.repos:
        requested = set(args.repos)
        groups = {
            repo: group_rows for repo, group_rows in groups.items() if repo in requested
        }
    groups = dict(sorted(groups.items()))
    if args.max_repos is not None:
        groups = dict(list(groups.items())[: args.max_repos])
    return [row for group_rows in groups.values() for row in group_rows], set(groups)


def _chain_length_histogram(units: list[MemoryChain]) -> dict[str, int]:
    histogram: dict[int, int] = {}
    for unit in units:
        if unit.is_singleton:
            continue
        histogram[unit.length] = histogram.get(unit.length, 0) + 1
    return {str(length): count for length, count in sorted(histogram.items())}


def _per_repo_plan_counts(units: list[MemoryChain]) -> dict[str, dict[str, int]]:
    per_repo: dict[str, dict[str, int]] = {}
    for unit in units:
        bucket = per_repo.setdefault(
            unit.repo, {"chains": 0, "chain_instances": 0, "singletons": 0}
        )
        if unit.is_singleton:
            bucket["singletons"] += 1
        else:
            bucket["chains"] += 1
            bucket["chain_instances"] += unit.length
    return {repo: dict(counts) for repo, counts in sorted(per_repo.items())}


def _chain_config_payload(
    chain: MemoryChain,
    *,
    config: ProRepoExperimentConfig,
    provider_auth_env: str,
    position: int,
    write_trajectories: bool,
) -> dict[str, Any]:
    return {
        "repo": chain.repo,
        "chain_id": chain.chain_id,
        "position": position,
        "instances_in_chain": chain.length,
        "provider_auth_env": provider_auth_env,
        "write_trajectories": bool(write_trajectories),
        "config": config.as_record(),
    }


def _run_repo(
    chain: MemoryChain,
    *,
    args: argparse.Namespace,
    config: ProRepoExperimentConfig,
    provider_auth_env: str,
    run_root: Path,
    suite: SwebenchSuite,
    backend: LocalDockerBackend,
    store: LocalDirStore,
) -> dict[str, Any]:
    repo = chain.repo
    rows = list(chain.rows)
    chain_id = chain.chain_id
    handoffs_total = 0
    mid_instance_handoffs_total = 0
    boundary_handoffs_total = 0
    errors = 0
    skipped_records: list[dict[str, Any]] = []
    chain_state_payload: dict[str, Any] | None = None

    for position, instance in enumerate(rows, start=1):
        instance_id = str(instance["instance_id"])
        paths = prepare_run_directory(
            run_root=run_root, run_id=args.run_id, instance_id=instance_id
        )
        config_payload = _chain_config_payload(
            chain,
            config=config,
            provider_auth_env=provider_auth_env,
            position=position,
            write_trajectories=bool(args.write_trajectories),
        )
        chain_runner.write_json_atomic(
            paths.input_dir / CHAIN_CONFIG_KEY.split("/", 1)[1], config_payload
        )
        state_input = paths.input_dir / CHAIN_STATE_INPUT_KEY.split("/", 1)[1]
        if chain_state_payload is None:
            state_input.unlink(missing_ok=True)
        else:
            chain_runner.write_json_atomic(state_input, chain_state_payload)
        container = container_name(
            _container_suite_name(config),
            instance_id,
            args.run_id,
            namespace=repo,
        )
        result: dict[str, Any]
        try:
            artifacts = run_suite_instance(
                suite=suite,
                instance=instance,
                backend=backend,
                store=store,
                run_root=run_root,
                run_id=args.run_id,
                api_kind=config.api_kind,
                max_turns=config.max_turns,
                provider_env=chain_runner.provider_env_for_auth_env(
                    provider_auth_env, api_kind=config.api_kind
                ),
                runner_module=PRO_REPO_CHAIN_RUNNER_MODULE,
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
                chain=chain,
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
                chain=chain,
                provider_auth_env=provider_auth_env,
                error=f"{type(exc).__name__}: {exc}",
            )
            result["status"] = "error"
            result["error"] = result.get("error") or f"{type(exc).__name__}: {exc}"
            print(
                f"[{chain_id} #{position}/{len(rows)}] {instance_id}: "
                f"{result['error']}",
                flush=True,
            )

        result.setdefault("agent_flavor", config.agent_flavor)
        result.setdefault("task_tool", config.task_tool)
        result.setdefault("compression_strategy", config.compression_strategy)
        result.setdefault("handoff_written", False)
        result.setdefault("boundary_handoff_written", False)
        result.setdefault("context_window_handoffs", 0)
        if result.get("status") == "error":
            errors += 1
        handoffs, mid_handoffs, boundary_handoffs = _handoff_counts(result)
        handoffs_total += handoffs
        mid_instance_handoffs_total += mid_handoffs
        boundary_handoffs_total += boundary_handoffs
        state_output = paths.output_dir / CHAIN_STATE_OUTPUT_KEY.split("/", 1)[1]
        if state_output.exists():
            chain_state_payload = chain_runner.read_json(state_output)

        if result.get("status") == "skipped":
            skipped_record = {
                "repo": repo,
                "chain_id": chain_id,
                "instance_id": instance_id,
                "position": position,
                "reason": str(result.get("skip_reason") or ""),
                "error": str(result.get("error") or ""),
                "invalid_prompt_retries": int(
                    result.get("invalid_prompt_retries", 0) or 0
                ),
            }
            skipped_records.append(skipped_record)

        chain_runner.write_json_atomic(paths.output_dir / "result.json", result)

    repo_dir = run_root / args.run_id / "_repo_chains" / safe_path_part(chain_id)
    if skipped_records:
        write_jsonl_atomic(repo_dir / "skipped_instances.jsonl", skipped_records)
    chain_runner.write_json_atomic(
        repo_dir / "summary.json",
        {
            "repo": repo,
            "chain_id": chain_id,
            "instances": len(rows),
            "errors": errors,
            "skipped": len(skipped_records),
            "provider_auth_env": provider_auth_env,
            "agent_flavor": config.agent_flavor,
            "task_tool": config.task_tool,
            "compression_strategy": config.compression_strategy,
            "handoff": config.handoff,
            "handoffs": handoffs_total,
            "mid_instance_handoffs": mid_instance_handoffs_total,
            "boundary_handoffs": boundary_handoffs_total,
        },
    )
    return {
        "instances": len(rows),
        "errors": errors,
        "skipped": len(skipped_records),
        "skipped_records": skipped_records,
        "handoffs": handoffs_total,
        "mid_instance_handoffs": mid_instance_handoffs_total,
        "boundary_handoffs": boundary_handoffs_total,
    }


def _container_suite_name(config: ProRepoExperimentConfig) -> str:
    suffix = "task" if config.task_tool else "bash"
    return f"swebench_pro_repo_{config.agent_flavor}_{suffix}"


def _handoff_counts(result: Mapping[str, Any]) -> tuple[int, int, int]:
    """Return total, mid-instance, and boundary handoff occurrences."""

    mid = max(0, int(result.get("context_window_handoffs", 0) or 0))
    boundary = int(bool(result.get("boundary_handoff_written", False)))
    return mid + boundary, mid, boundary


def _load_result_or_error(
    path: Path,
    *,
    instance_id: str,
    chain: MemoryChain,
    provider_auth_env: str,
    error: str,
) -> dict[str, Any]:
    if path.exists():
        return chain_runner.read_json(path)
    return {
        "model_patch": "",
        "instance_id": instance_id,
        "chain_id": chain.chain_id,
        "status": "error" if error else "ok",
        "error": error,
        "provider_auth_env": provider_auth_env,
    }


if __name__ == "__main__":
    main()
