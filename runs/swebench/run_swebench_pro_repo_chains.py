"""Run SWE-bench Pro as one agent chain per repository part.

This is the experimental runner for studying context compression on long
SWE-bench Pro repo chains. It groups instances by repository, orders each group by
the timestamp of ``base_commit``, and runs each task with the agent inside that
task's SWE-bench Pro instance container. The host only plans chain parts, stages
chain artifacts, launches containers through the generic eval backend, and
passes ``out/chain_state.json`` from one instance to the next.

The default flavor is ``goal`` (the current thread-goal loop) with bash-only
solver tools and no chain compression. Use ``--agent-flavor`` to select
another no-read repo-chain flavor, ``--task-tool`` to add the task tool, and
``--compression-strategy summarize`` to turn on chain compression.

Example smoke:

    uv run --extra swebench python runs/swebench/run_swebench_pro_repo_chains.py \
      --repos NodeBB/NodeBB --limit-per-repo 1 --max-turns 5 --skip-official-eval

Formal run shape:

    uv run --extra swebench python runs/swebench/run_swebench_pro_repo_chains.py \
      --all \
      --parallel parts \
      --provider-auth-envs OPENAI_AUTH_TOKEN:12,OPENAI_AUTH_TOKEN2:11 \
      --api-kind openai-responses \
      --max-turns 250 \
      --run-official-eval

Full trajectories are written by default for this runner. Repo chains can
accumulate hundreds of thousands of prompt tokens, and the full trace records
every model request payload; pass ``--no-write-trajectories`` only for runs
where that storage cost is unacceptable.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from collections.abc import Mapping
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
from evals.swebench.pro_repo_chain import (  # noqa: E402
    CommitTimeResolver,
    DEFAULT_PRESERVE_KINDS,
    ProRepoExperimentConfig,
    RepoChainPart,
    group_instances_by_repo,
    split_repo_chain_parts,
    sort_repo_instances,
)
from evals.swebench.suite import SwebenchSuite  # noqa: E402
from simple_agent_lab.evals import LocalDirStore, LocalDockerBackend  # noqa: E402
from simple_agent_lab.evals.protocols import RESULT_KEY  # noqa: E402
from simple_agent_lab.evals.runner import (  # noqa: E402
    container_name,
    prepare_run_directory,
    run_suite_instance,
)
from simple_agent_lab.evals.chain import (  # noqa: E402
    CHAIN_CONFIG_KEY,
    CHAIN_STATE_INPUT_KEY,
    CHAIN_STATE_OUTPUT_KEY,
)
from simple_agent_lab.llm.env import (  # noqa: E402
    API_KIND_ENV,
    OPENAI_AUTH_ENV,
    OPENAI_ENV,
    OPENAI_MODEL_ENV,
    OPENAI_REASONING_EFFORT_ENV,
    REASONING_EFFORT_ENV,
    provider_from_env,
    request_extra_from_env,
)
from simple_agent_lab.llm.provider import api_kind_defaults  # noqa: E402
from simple_agent_lab.trace import (  # noqa: E402
    run_trace_from_state,
    write_event_stream,
    write_jsonl_atomic,
)
from simple_agent_lab.protocols import Event, TurnStartEvent  # noqa: E402

ENCRYPTED_REASONING_INCLUDE = "reasoning.encrypted_content"
PRO_REPO_CHAIN_RUNNER_MODULE = "simple_agent_lab.evals.chain"
REPO_CHAIN_MODE = "repo_chain"
REPO_CHAIN_AGENT_FLAVORS = ("bash", "loop", "goal", "pdr")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Run all selected repos.")
    parser.add_argument(
        "--repos",
        nargs="*",
        default=[],
        help="Optional exact repo names to run, e.g. NodeBB/NodeBB.",
    )
    parser.add_argument("--max-repos", type=int, help="Limit repo count after filter.")
    parser.add_argument(
        "--limit-per-repo",
        type=int,
        help="Limit instances per repo after commit-time sorting.",
    )
    parser.add_argument("--dataset-name", default=ProRepoExperimentConfig.dataset_name)
    parser.add_argument("--split", default=ProRepoExperimentConfig.split)
    parser.add_argument("--instance-json", help="Use a local JSON/JSONL dataset file.")
    parser.add_argument(
        "--run-root",
        default=str(harness.DEFAULT_PRO_RUN_ROOT),
        help="Output root. Defaults to evals/out/swebench_pro.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Output run id. Defaults to a timestamped id derived from selected variables.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override OPENAI_MODEL from the environment or .env.",
    )
    parser.add_argument(
        "--api-kind",
        default=ProRepoExperimentConfig.api_kind,
        help="LLM adapter API kind. Defaults to openai-responses for this experiment.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="Override REASONING_EFFORT from the environment or .env.",
    )
    parser.add_argument(
        "--max-turns", type=int, default=ProRepoExperimentConfig.max_turns
    )
    parser.add_argument(
        "--agent-flavor",
        default=ProRepoExperimentConfig.agent_flavor,
        choices=REPO_CHAIN_AGENT_FLAVORS,
        help=(
            "Agent flavor for each repo-chain part. Repo chains intentionally "
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
            "When context reaches --context-window-tokens, finish the current "
            "instance, have the model write a handoff document, and start the "
            "next instance in a fresh window seeded only with that handoff "
            "(default: enabled). Ignored when --compression-strategy summarize."
        ),
    )
    parser.add_argument(
        "--context-window-tokens",
        type=int,
        default=ProRepoExperimentConfig.context_window_tokens,
        help=(
            "Handoff trigger: reset the repo-chain window once the estimated "
            "active context reaches this many tokens (default: 272000; the real "
            "model window is larger, leaving buffer for one instance)."
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
        "--parallel",
        default="parts",
        help=(
            "'parts' to use one worker per planned chain part, or a positive integer."
        ),
    )
    parser.add_argument(
        "--provider-auth-envs",
        default=None,
        help=(
            "Comma-separated auth env slots, e.g. "
            "OPENAI_AUTH_TOKEN:12,OPENAI_AUTH_TOKEN2:11. "
            "Each planned repo chain gets one slot in plan order. "
            "Defaults to OPENAI_AUTH_TOKEN for every chain."
        ),
    )
    parser.add_argument("--dotenv", default=str(ROOT / ".env"))
    parser.add_argument("--network-mode", default="host")
    parser.add_argument("--mem-limit", default="8g")
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
        "--plan-only",
        action="store_true",
        help="Write the repo-chain plan and exit before provider or Docker setup.",
    )
    parser.add_argument(
        "--run-official-eval",
        action="store_true",
        help="Run the official SWE-bench Pro evaluator after inference.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove any leftover deterministic container before starting it.",
    )
    parser.add_argument(
        "--write-trajectories",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write full per-instance and per-repo trajectory.jsonl files "
            "(default: enabled). Use --no-write-trajectories to suppress them."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.all and not args.repos and not args.instance_json:
        raise SystemExit("Pass --all, --repos REPO..., or --instance-json PATH.")

    harness.load_dotenv(args.dotenv)
    _apply_provider_env_overrides(args)
    api_kind = harness.resolve_api_kind(args.api_kind)
    config = _experiment_config_from_args(args, api_kind=api_kind)
    args.run_id = _resolve_run_id(args.run_id, config)
    run_root = Path(args.run_root)
    rows = _load_rows(args)
    chains, manifest = _plan_groups(rows, args=args, run_root=run_root, config=config)
    if not chains:
        raise SystemExit("No SWE-bench Pro instances selected.")
    parallel = _resolve_parallel(args.parallel, chain_count=len(chains))
    manifest["resolved_parallel"] = parallel
    provider_auth_envs = _expand_provider_auth_envs(
        args.provider_auth_envs, chain_count=len(chains)
    )
    chain_auth_envs = dict(zip(chains, provider_auth_envs))
    manifest["provider_auth"] = _provider_auth_manifest(
        chain_auth_envs, spec=args.provider_auth_envs
    )

    batch_dir = run_root / args.run_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = batch_dir / f"{args.run_id}_predictions.jsonl"
    instances_json = batch_dir / "instances.jsonl"
    _write_jsonl_records(
        instances_json, [row for chain in chains.values() for row in chain.rows]
    )
    _write_json(batch_dir / "experiment.json", manifest)

    print("=== SWE-bench Pro repo-chain experiment ===")
    print(f"run_id: {args.run_id}")
    print(f"repos: {manifest['repo_count']}")
    print(f"chain_parts: {len(chains)}")
    print(f"instances: {sum(len(chain.rows) for chain in chains.values())}")
    print(f"parallel: {parallel}")
    print(
        "provider_auth: "
        + ", ".join(
            f"{entry['auth_env']} x{entry['chain_slots']}"
            for entry in manifest["provider_auth"]["slot_summary"]
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
        print(f"plan written: {batch_dir / 'experiment.json'}")
        print(f"instances written: {instances_json}")
        return

    wheelhouse = _resolve_wheelhouse(args)
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
    )
    store = LocalDirStore(run_root)
    prediction_lock = threading.Lock()

    failures: list[dict[str, str]] = []
    skipped_instances: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(
                _run_repo,
                chain,
                args=args,
                config=config,
                provider_auth_env=chain_auth_envs[chain.chain_id],
                run_root=run_root,
                suite=suite,
                backend=backend,
                store=store,
                predictions_path=predictions_path,
                prediction_lock=prediction_lock,
            ): chain.chain_id
            for chain in chains.values()
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
                handoffs = int(result.get("handoffs", 0) or 0)
                print(
                    f"[DONE] {chain_id}: {result['instances']} instance(s), "
                    f"{result['errors']} error(s), {result['skipped']} skipped, "
                    f"{handoffs} handoff(s)",
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
        _write_json(batch_dir / "repo_failures.json", failures)
        raise SystemExit(f"{len(failures)} repo chain(s) failed")


def _apply_provider_env_overrides(args: argparse.Namespace) -> None:
    """Apply explicit CLI provider overrides without masking .env defaults."""

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
        preserve_kinds=DEFAULT_PRESERVE_KINDS,
        agent_flavor=args.agent_flavor,
        solver_read=False,
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
    task = "task" if task_tool else "bash"
    return (
        f"simple-agent-lab-pro-repo-chain-{agent_flavor}-{task}-{compression_strategy}"
    )


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
    agent = "" if config.agent_flavor == "goal" else f"-{config.agent_flavor}"
    task = "-task" if config.task_tool else ""
    prefix = f"pro-repo-chain{agent}{task}-{compression}"
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
    return [dict(row) for row in load_dataset(args.dataset_name, split=args.split)]


def _plan_groups(
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    run_root: Path,
    config: ProRepoExperimentConfig,
) -> tuple[dict[str, RepoChainPart], dict[str, Any]]:
    groups = group_instances_by_repo(rows)
    if args.repos:
        requested = set(args.repos)
        groups = {repo: rows for repo, rows in groups.items() if repo in requested}
    groups = dict(sorted(groups.items()))
    if args.max_repos is not None:
        groups = dict(list(groups.items())[: args.max_repos])

    resolver = CommitTimeResolver(cache_root=run_root / "repo-cache")
    chains: dict[str, RepoChainPart] = {}
    order_manifest: dict[str, dict[str, Any]] = {}
    print(f"[plan] ordering {len(groups)} repo(s) by base commit time", flush=True)
    for index, (repo, repo_rows) in enumerate(groups.items(), start=1):
        print(
            f"[plan] {index}/{len(groups)} {repo}: {len(repo_rows)} instance(s)",
            flush=True,
        )
        resolved_commit_times = resolver.timestamps(
            repo,
            (
                str(row.get("base_commit") or "")
                for row in repo_rows
                if row.get("base_commit")
            ),
        )
        commit_times = {
            commit: timestamp
            for commit, timestamp in resolved_commit_times.items()
            if timestamp is not None
        }
        sorted_rows = sort_repo_instances(repo_rows, commit_times=commit_times)
        if args.limit_per_repo is not None:
            sorted_rows = sorted_rows[: args.limit_per_repo]
        repo_parts = split_repo_chain_parts(repo, sorted_rows)
        for part in repo_parts:
            chains[part.chain_id] = part
        ordered_instances = [
            {
                "instance_id": str(row.get("instance_id") or ""),
                "base_commit": str(row.get("base_commit") or ""),
                "commit_time": resolved_commit_times.get(
                    str(row.get("base_commit") or "")
                ),
            }
            for row in sorted_rows
        ]
        order_manifest[repo] = {
            "instance_count": len(sorted_rows),
            "part_count": len(repo_parts),
            "instances": ordered_instances,
            "parts": [
                {
                    "chain_id": part.chain_id,
                    "part_index": part.part_index,
                    "part_count": part.part_count,
                    "instance_count": len(part.rows),
                    "instance_ids": [
                        str(row.get("instance_id") or "") for row in part.rows
                    ],
                }
                for part in repo_parts
            ],
        }

    manifest = {
        "schema": "simple-agent-lab.swebench-pro-repo-chain-experiment.v1",
        "run_id": args.run_id,
        "config": config.as_record(),
        "repo_count": len(groups),
        "chain_part_count": len(chains),
        "instance_count": sum(len(chain.rows) for chain in chains.values()),
        "parallel": args.parallel,
        "write_trajectories": bool(args.write_trajectories),
        "order": order_manifest,
        "commit_time_warnings": resolver.warnings,
    }
    return chains, manifest


def _expand_provider_auth_envs(value: str | None, *, chain_count: int) -> list[str]:
    """Expand ``ENV:COUNT`` auth slots into one env var per repo chain."""

    if chain_count < 0:
        raise ValueError("chain_count must be non-negative")
    if chain_count == 0:
        return []
    spec = (value or "").strip()
    if not spec:
        return [OPENAI_ENV.auth] * chain_count

    expanded: list[str] = []
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            raise SystemExit("--provider-auth-envs contains an empty entry")
        auth_env, separator, raw_count = part.partition(":")
        auth_env = auth_env.strip()
        if not _is_env_var_name(auth_env):
            raise SystemExit(
                f"--provider-auth-envs has invalid env var name {auth_env!r}"
            )
        if not separator:
            raise SystemExit(
                f"--provider-auth-envs entries must use ENV:COUNT, got {part!r}"
            )
        count_text = raw_count.strip()
        try:
            count = int(count_text)
        except ValueError:
            raise SystemExit(
                f"--provider-auth-envs count for {auth_env} must be an integer"
            ) from None
        if count <= 0:
            raise SystemExit(
                f"--provider-auth-envs count for {auth_env} must be positive"
            )
        expanded.extend([auth_env] * count)

    if len(expanded) < chain_count:
        raise SystemExit(
            f"--provider-auth-envs provides {len(expanded)} auth slot(s) "
            f"for {chain_count} chain(s)"
        )
    return expanded[:chain_count]


def _is_env_var_name(value: str) -> bool:
    if not value:
        return False
    first = value[0]
    if not (first.isalpha() or first == "_"):
        return False
    return all(char.isalnum() or char == "_" for char in value)


def _provider_auth_manifest(
    chain_auth_envs: Mapping[str, str], *, spec: str | None
) -> dict[str, Any]:
    auth_envs = list(chain_auth_envs.values())
    return {
        "spec": spec or f"{OPENAI_ENV.auth}:{len(auth_envs)}",
        "slot_summary": _provider_auth_slot_summary(auth_envs),
        "chains": dict(chain_auth_envs),
    }


def _provider_auth_slot_summary(auth_envs: list[str]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for auth_env in auth_envs:
        if summary and summary[-1]["auth_env"] == auth_env:
            summary[-1]["chain_slots"] += 1
        else:
            summary.append({"auth_env": auth_env, "chain_slots": 1})
    return summary


def _providers_from_auth_envs(auth_envs: list[str], *, api_kind: str) -> dict[str, Any]:
    providers: dict[str, Any] = {}
    defaults = api_kind_defaults(api_kind)
    for auth_env in dict.fromkeys(auth_envs):
        providers[auth_env] = provider_from_env(
            replace(OPENAI_ENV, auth=auth_env),
            api_kind=api_kind,
            default_temperature=defaults.default_temperature,
            read_reasoning=True,
            label=f"SWE-bench Pro repo-chain provider ({auth_env})",
        )
    return providers


def _request_extra_for_api_kind(api_kind: str) -> dict[str, Any]:
    """Build per-request extras for the selected adapter."""

    extra = request_extra_from_env()
    if api_kind != "openai-responses":
        return extra
    include = list(extra.get("include") or [])
    if ENCRYPTED_REASONING_INCLUDE not in include:
        include.append(ENCRYPTED_REASONING_INCLUDE)
    extra["include"] = include
    return extra


def _resolve_wheelhouse(args: argparse.Namespace) -> Path | None:
    wheelhouse_arg = args.wheelhouse or str(harness.DEFAULT_PRO_WHEELHOUSE)
    return Path(wheelhouse_arg).resolve() if wheelhouse_arg else None


def _provider_env_for_auth_env(auth_env: str, *, api_kind: str) -> dict[str, str]:
    """Build the in-container provider env for one assigned auth slot."""

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
    return env


def _chain_config_payload(
    chain: RepoChainPart,
    *,
    config: ProRepoExperimentConfig,
    provider_auth_env: str,
    position: int,
    write_trajectories: bool,
) -> dict[str, Any]:
    return {
        "mode": REPO_CHAIN_MODE,
        "repo": chain.repo,
        "chain_id": chain.chain_id,
        "chain_display_name": _chain_display_name(chain),
        "part_index": chain.part_index,
        "part_count": chain.part_count,
        "position": position,
        "instances_in_chain": len(chain.rows),
        "provider_auth_env": provider_auth_env,
        "write_trajectories": bool(write_trajectories),
        "config": config.as_record(),
    }


def _run_repo(
    chain: RepoChainPart,
    *,
    args: argparse.Namespace,
    config: ProRepoExperimentConfig,
    provider_auth_env: str,
    run_root: Path,
    suite: SwebenchSuite,
    backend: LocalDockerBackend,
    store: LocalDirStore,
    predictions_path: Path,
    prediction_lock: threading.Lock | None,
) -> dict[str, Any]:
    repo = chain.repo
    rows = list(chain.rows)
    chain_id = chain.chain_id
    handoffs_total = 0
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
        _write_json(paths.input_dir / CHAIN_CONFIG_KEY.split("/", 1)[1], config_payload)
        state_input = paths.input_dir / CHAIN_STATE_INPUT_KEY.split("/", 1)[1]
        if chain_state_payload is None:
            state_input.unlink(missing_ok=True)
        else:
            _write_json(state_input, chain_state_payload)
        container = container_name(
            _container_suite_name(config),
            instance_id,
            args.run_id,
            namespace=repo,
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
                provider_env=_provider_env_for_auth_env(
                    provider_auth_env, api_kind=config.api_kind
                ),
                runner_module=PRO_REPO_CHAIN_RUNNER_MODULE,
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
        result.setdefault("solver_read", config.solver_read)
        result.setdefault("task_tool", config.task_tool)
        result.setdefault("compression_strategy", config.compression_strategy)
        result.setdefault("handoff_written", False)
        if result.get("status") == "error":
            errors += 1
        if bool(result.get("handoff_written")):
            handoffs_total += 1
        state_output = paths.output_dir / CHAIN_STATE_OUTPUT_KEY.split("/", 1)[1]
        if state_output.exists():
            chain_state_payload = _read_json(state_output)

        if result.get("status") == "skipped":
            skipped_record = {
                "repo": repo,
                "chain_id": chain_id,
                "part_index": chain.part_index,
                "part_count": chain.part_count,
                "instance_id": instance_id,
                "position": position,
                "reason": str(result.get("skip_reason") or ""),
                "error": str(result.get("error") or ""),
                "invalid_prompt_retries": int(
                    result.get("invalid_prompt_retries", 0) or 0
                ),
            }
            skipped_records.append(skipped_record)

        _write_json(paths.output_dir / "result.json", result)
        _write_incremental_predictions(
            predictions_path=predictions_path,
            run_root=run_root,
            run_id=args.run_id,
            model_name=config.model_name,
            dataset_name=config.dataset_name,
            lock=prediction_lock,
        )

    repo_dir = run_root / args.run_id / "_repo_chains" / _safe_path_part(chain_id)
    repo_dir.mkdir(parents=True, exist_ok=True)
    if skipped_records:
        _write_jsonl_records(repo_dir / "skipped_instances.jsonl", skipped_records)
    _write_json(
        repo_dir / "summary.json",
        {
            "repo": repo,
            "chain_id": chain_id,
            "part_index": chain.part_index,
            "part_count": chain.part_count,
            "instances": len(rows),
            "errors": errors,
            "skipped": len(skipped_records),
            "provider_auth_env": provider_auth_env,
            "agent_flavor": config.agent_flavor,
            "solver_read": config.solver_read,
            "task_tool": config.task_tool,
            "compression_strategy": config.compression_strategy,
            "handoff": config.handoff,
            "handoffs": handoffs_total,
        },
    )
    return {
        "instances": len(rows),
        "errors": errors,
        "skipped": len(skipped_records),
        "provider_auth_env": provider_auth_env,
        "skipped_records": skipped_records,
        "handoffs": handoffs_total,
    }


def _container_suite_name(config: ProRepoExperimentConfig) -> str:
    suffix = "task" if config.task_tool else "bash"
    return f"swebench_pro_repo_{config.agent_flavor}_{suffix}"


def _maybe_write_trace(
    args: argparse.Namespace,
    *,
    path: Path,
    state: Any,
    trace_id: str,
    meta: dict[str, Any],
) -> bool:
    """Write a full trajectory unless explicitly disabled."""

    if getattr(args, "write_trajectories", True) is False:
        return False
    trace = run_trace_from_state(
        state=state,
        trace_id=trace_id,
        producer="swebench_pro_repo_chain",
        meta=meta,
    )
    write_event_stream(path, trace)
    return True


def _load_result_or_error(
    path: Path,
    *,
    instance_id: str,
    chain: RepoChainPart,
    provider_auth_env: str,
    error: str,
) -> dict[str, Any]:
    if path.exists():
        return _read_json(path)
    return {
        "model_patch": "",
        "instance_id": instance_id,
        "repo": chain.repo,
        "chain_id": chain.chain_id,
        "chain_part_index": chain.part_index,
        "chain_part_count": chain.part_count,
        "provider_auth_env": provider_auth_env,
        "status": "error" if error else "ok",
        "error": error,
        "skip_reason": "",
        "invalid_prompt_retries": 0,
        "chain_window_index": 1,
        "baseline_commit": "",
        "compression_metrics": {},
        "chain_event_start": 0,
        "chain_event_end": 0,
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
    lock: threading.Lock | None,
) -> None:
    """Refresh the run-level predictions file after each completed instance."""

    def write() -> None:
        predictions = predictions_from_run_dirs(
            run_root,
            run_id=run_id,
            model_name=model_name,
            dataset_name=dataset_name,
        )
        write_jsonl_atomic(predictions_path, predictions)

    if lock is None:
        write()
        return
    with lock:
        write()


def _chain_display_name(chain: RepoChainPart) -> str:
    if chain.part_count == 1:
        return chain.repo
    return f"{chain.repo} part {chain.part_index}/{chain.part_count}"


def _remaining_turn_budget(events: list[Event], max_turns: int) -> int:
    used = sum(1 for event in events if isinstance(event, TurnStartEvent))
    return max(0, max_turns - used)


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


def _resolve_parallel(value: str, *, chain_count: int) -> int:
    if value == "parts":
        return max(1, chain_count)
    try:
        parsed = int(value)
    except ValueError:
        raise SystemExit("--parallel must be 'parts' or a positive integer") from None
    if parsed <= 0:
        raise SystemExit("--parallel must be positive")
    return parsed


def _ensure_ok(result: Any, label: str) -> None:
    returncode = int(getattr(result, "returncode", 0) or 0)
    if returncode == 0:
        return
    stderr = str(getattr(result, "stderr", "") or "").strip()
    raise RuntimeError(f"{label} failed with exit {returncode}: {stderr}")


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
