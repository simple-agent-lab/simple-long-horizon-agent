"""Run SWE-bench Pro as one compressed agent session per repository.

This is the experimental runner for studying context compression on long
SWE-bench Pro sessions. It groups instances by repository, orders each group by
the timestamp of ``base_commit``, and runs each repository in one persistent
Simple Agent Lab ``State`` while swapping the active Docker container between
instances. The host-side agent has only a bash tool; bash executes via
``docker exec`` in the current instance container.

Example smoke:

    uv run --extra swebench python runs/swebench/run_swebench_pro_repo_sessions.py \
      --repos NodeBB/NodeBB --limit-per-repo 1 --max-turns 5 --skip-official-eval

Formal run shape:

    uv run --extra swebench python runs/swebench/run_swebench_pro_repo_sessions.py \
      --all \
      --parallel parts \
      --provider-auth-envs OPENAI_AUTH_TOKEN:12,OPENAI_AUTH_TOKEN2:11 \
      --api-kind openai-responses \
      --max-turns 250 \
      --run-official-eval

Full trajectories are opt-in for this runner. Repo sessions can accumulate
hundreds of thousands of prompt tokens, and the full trace records every
model request payload; result/prediction/eval artifacts do not need that data.
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
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evals.swebench import harness  # noqa: E402
from evals.swebench.evaluate_predictions import predictions_from_run_dirs  # noqa: E402
from evals.swebench.pro_repo_session import (  # noqa: E402
    CommitTimeResolver,
    CurrentContainer,
    DEFAULT_PRESERVE_KINDS,
    DockerCommandRunner,
    ProRepoExperimentConfig,
    RepoSessionPart,
    append_instance_task,
    extract_container_patch,
    group_instances_by_repo,
    make_container_bash_tool,
    prepare_container_baseline,
    split_repo_session_parts,
    sort_repo_instances,
    start_repo_state,
)
from simple_agent_lab import ContextCompressionEvent, run as run_agent  # noqa: E402
from simple_agent_lab.compression import summarize_compression  # noqa: E402
from simple_agent_lab.evals.runner import (  # noqa: E402
    container_name,
    prepare_run_directory,
)
from simple_agent_lab.evals.suites.swebench.container import (  # noqa: E402
    AGENT_NAME,
    AGENT_ROLE,
    AGENT_SYSTEM_PROMPT,
    build_task,
)
from simple_agent_lab.evals.suites.swebench.patch import (  # noqa: E402
    instance_language,
)
from simple_agent_lab.llm.env import (  # noqa: E402
    OPENAI_ENV,
    OPENAI_MODEL_ENV,
    OPENAI_REASONING_EFFORT_ENV,
    REASONING_EFFORT_ENV,
    provider_from_env,
    request_extra_from_env,
)
from simple_agent_lab.llm.provider import api_kind_defaults  # noqa: E402
from simple_agent_lab.llm_agent import make_llm_agent  # noqa: E402
from simple_agent_lab.messages import (  # noqa: E402
    is_tool_result_message,
    message_tool_calls,
    tool_results_of,
    user_message,
)
from simple_agent_lab.trace import (  # noqa: E402
    run_trace_from_state,
    write_event_stream,
    write_jsonl_atomic,
)
from simple_agent_lab.protocols import Event, TurnStartEvent  # noqa: E402

INVALID_PROMPT_TOOL_REMINDER = (
    "刚刚的工具调用及其输出会触发 invalid_prompt，已从上下文移除。请使用其他命令继续。"
)
INVALID_PROMPT_INSTANCE_END_MESSAGE = (
    "上一道题 {instance_id} 在这里结束；因为工具输出持续触发 invalid_prompt，"
    "已跳过该实例。继续下一道题。"
)
INVALID_PROMPT_TOOL_RETRY_LIMIT = 20
ENCRYPTED_REASONING_INCLUDE = "reasoning.encrypted_content"
InvalidPromptSource = Literal["instance_task", "tool_output", "unknown"]


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
        default=f"pro-repo-summarize-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
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
            "'parts'/'repos' to use one worker per planned session part, "
            "or a positive integer."
        ),
    )
    parser.add_argument(
        "--provider-auth-envs",
        default=None,
        help=(
            "Comma-separated auth env slots, e.g. "
            "OPENAI_AUTH_TOKEN:12,OPENAI_AUTH_TOKEN2:11. "
            "Each planned repo session gets one slot in plan order. "
            "Defaults to OPENAI_AUTH_TOKEN for every session."
        ),
    )
    parser.add_argument("--dotenv", default=str(ROOT / ".env"))
    parser.add_argument("--network-mode", default="host")
    parser.add_argument("--mem-limit", default="8g")
    parser.add_argument(
        "--skip-official-eval",
        action="store_true",
        help="Only run inference and collect predictions.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write the repo/session plan and exit before provider or Docker setup.",
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
        action="store_true",
        help=(
            "Write full per-instance and per-repo trajectory.jsonl files. "
            "Disabled by default because long repo sessions can produce "
            "hundreds of GB of trace data."
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
    run_root = Path(args.run_root)
    rows = _load_rows(args)
    sessions, manifest = _plan_groups(rows, args=args, run_root=run_root, config=config)
    if not sessions:
        raise SystemExit("No SWE-bench Pro instances selected.")
    parallel = _resolve_parallel(args.parallel, session_count=len(sessions))
    manifest["resolved_parallel"] = parallel
    provider_auth_envs = _expand_provider_auth_envs(
        args.provider_auth_envs, session_count=len(sessions)
    )
    session_auth_envs = dict(zip(sessions, provider_auth_envs))
    manifest["provider_auth"] = _provider_auth_manifest(
        session_auth_envs, spec=args.provider_auth_envs
    )

    batch_dir = run_root / args.run_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = batch_dir / f"{args.run_id}_predictions.jsonl"
    instances_json = batch_dir / "instances.jsonl"
    _write_jsonl_records(
        instances_json, [row for session in sessions.values() for row in session.rows]
    )
    _write_json(batch_dir / "experiment.json", manifest)

    print("=== SWE-bench Pro repo-session compression experiment ===")
    print(f"run_id: {args.run_id}")
    print(f"repos: {manifest['repo_count']}")
    print(f"session_parts: {len(sessions)}")
    print(f"instances: {sum(len(session.rows) for session in sessions.values())}")
    print(f"parallel: {parallel}")
    print(
        "provider_auth: "
        + ", ".join(
            f"{entry['auth_env']} x{entry['session_slots']}"
            for entry in manifest["provider_auth"]["slot_summary"]
        )
    )
    print(
        f"model: {config.model} api_kind={config.api_kind} "
        f"reasoning={config.reasoning_effort}"
    )
    print(
        "compression: summarize "
        f"threshold={config.threshold_tokens} keep_recent={config.keep_recent}"
    )
    print(f"trajectories: {'full' if args.write_trajectories else 'disabled'}")
    print("")
    if args.plan_only:
        print(f"plan written: {batch_dir / 'experiment.json'}")
        print(f"instances written: {instances_json}")
        return

    providers = _providers_from_auth_envs(
        provider_auth_envs,
        api_kind=config.api_kind,
    )
    request_extra = _request_extra_for_api_kind(config.api_kind)
    prediction_lock = threading.Lock()

    failures: list[dict[str, str]] = []
    skipped_instances: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(
                _run_repo,
                session,
                args=args,
                config=config,
                provider=providers[session_auth_envs[session.session_id]],
                provider_auth_env=session_auth_envs[session.session_id],
                request_extra=request_extra,
                run_root=run_root,
                predictions_path=predictions_path,
                prediction_lock=prediction_lock,
            ): session.session_id
            for session in sessions.values()
        }
        for future in as_completed(futures):
            session_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failures.append(
                    {"session_id": session_id, "error": f"{type(exc).__name__}: {exc}"}
                )
                print(f"[FAIL] {session_id}: {type(exc).__name__}: {exc}", flush=True)
            else:
                skipped_instances.extend(result.get("skipped_records", []))
                print(
                    f"[DONE] {session_id}: {result['instances']} instance(s), "
                    f"{result['errors']} error(s), {result['skipped']} skipped",
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
        raise SystemExit(f"{len(failures)} repo session(s) failed")


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
        threshold_tokens=args.threshold_tokens,
        keep_recent=args.keep_recent,
        preserve_kinds=DEFAULT_PRESERVE_KINDS,
    )


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
) -> tuple[dict[str, RepoSessionPart], dict[str, Any]]:
    groups = group_instances_by_repo(rows)
    if args.repos:
        requested = set(args.repos)
        groups = {repo: rows for repo, rows in groups.items() if repo in requested}
    groups = dict(sorted(groups.items()))
    if args.max_repos is not None:
        groups = dict(list(groups.items())[: args.max_repos])

    resolver = CommitTimeResolver(cache_root=run_root / "repo-cache")
    sessions: dict[str, RepoSessionPart] = {}
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
        repo_parts = split_repo_session_parts(repo, sorted_rows)
        for part in repo_parts:
            sessions[part.session_id] = part
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
                    "session_id": part.session_id,
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
        "schema": "simple-agent-lab.swebench-pro-repo-session-experiment.v2",
        "run_id": args.run_id,
        "config": config.as_record(),
        "repo_count": len(groups),
        "session_part_count": len(sessions),
        "instance_count": sum(len(session.rows) for session in sessions.values()),
        "parallel": args.parallel,
        "write_trajectories": bool(args.write_trajectories),
        "order": order_manifest,
        "commit_time_warnings": resolver.warnings,
    }
    return sessions, manifest


def _expand_provider_auth_envs(value: str | None, *, session_count: int) -> list[str]:
    """Expand ``ENV:COUNT`` auth slots into one env var per repo session."""

    if session_count < 0:
        raise ValueError("session_count must be non-negative")
    if session_count == 0:
        return []
    spec = (value or "").strip()
    if not spec:
        return [OPENAI_ENV.auth] * session_count

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

    if len(expanded) < session_count:
        raise SystemExit(
            f"--provider-auth-envs provides {len(expanded)} auth slot(s) "
            f"for {session_count} session(s)"
        )
    return expanded[:session_count]


def _is_env_var_name(value: str) -> bool:
    if not value:
        return False
    first = value[0]
    if not (first.isalpha() or first == "_"):
        return False
    return all(char.isalnum() or char == "_" for char in value)


def _provider_auth_manifest(
    session_auth_envs: Mapping[str, str], *, spec: str | None
) -> dict[str, Any]:
    auth_envs = list(session_auth_envs.values())
    return {
        "spec": spec or f"{OPENAI_ENV.auth}:{len(auth_envs)}",
        "slot_summary": _provider_auth_slot_summary(auth_envs),
        "sessions": dict(session_auth_envs),
    }


def _provider_auth_slot_summary(auth_envs: list[str]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for auth_env in auth_envs:
        if summary and summary[-1]["auth_env"] == auth_env:
            summary[-1]["session_slots"] += 1
        else:
            summary.append({"auth_env": auth_env, "session_slots": 1})
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
            label=f"SWE-bench Pro repo-session provider ({auth_env})",
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


def _run_repo(
    session: RepoSessionPart,
    *,
    args: argparse.Namespace,
    config: ProRepoExperimentConfig,
    provider: Any,
    provider_auth_env: str,
    request_extra: dict[str, Any],
    run_root: Path,
    predictions_path: Path,
    prediction_lock: threading.Lock | None,
) -> dict[str, Any]:
    repo = session.repo
    rows = list(session.rows)
    session_id = session.session_id
    docker = DockerCommandRunner()
    current = CurrentContainer()
    bash_tool = make_container_bash_tool(current, docker)
    policy = config.context_policy(provider, request_extra=request_extra)
    agent = make_llm_agent(
        name=AGENT_NAME,
        provider=provider,
        role=AGENT_ROLE,
        system_prompt=AGENT_SYSTEM_PROMPT,
        tools=(bash_tool,),
        target="user",
        context_policy=policy,
        request_extra=request_extra,
    )
    state = start_repo_state(_session_display_name(session), agent_name=AGENT_NAME)
    errors = 0
    skipped_records: list[dict[str, Any]] = []

    for position, instance in enumerate(rows, start=1):
        instance_id = str(instance["instance_id"])
        paths = prepare_run_directory(
            run_root=run_root, run_id=args.run_id, instance_id=instance_id
        )
        sanitized = harness.sanitized_instance(dict(instance))
        _write_json(paths.input_dir / "instance.json", sanitized)
        task = build_task(sanitized, workdir=harness.DEFAULT_PRO_WORKDIR)
        event_start = len(state.events)
        append_instance_task(
            state,
            agent_name=AGENT_NAME,
            instance_id=instance_id,
            task=task,
        )

        container = container_name(
            "swebench_pro_repo",
            instance_id,
            args.run_id,
            namespace=repo,
        )
        image = harness.docker_image_for_instance(
            dict(instance),
            dataset_name=config.dataset_name,
            namespace="swebench",
            instance_image_tag="latest",
            env_image_tag="latest",
        )
        workdir = harness.resolve_workdir(
            "", dict(instance), dataset_name=config.dataset_name
        )
        current.name = container
        current.workdir = workdir
        status = "ok"
        error = ""
        skip_reason = ""
        baseline = ""
        patch = ""
        invalid_prompt_retries = 0
        try:
            if args.force:
                docker.remove_container(container)
            started = docker.start_container(
                container_name=container,
                image=image,
                workdir=workdir,
                network_mode=args.network_mode,
                mem_limit=args.mem_limit,
            )
            _ensure_ok(started, f"start {container}")
            baseline = prepare_container_baseline(
                docker, current, language=instance_language(dict(instance))
            )
            while status == "ok":
                _repair_active_tool_pairs(state, agent_name=AGENT_NAME)
                turn_budget = _remaining_turn_budget(
                    state.events[event_start:], config.max_turns
                )
                if turn_budget <= 0:
                    prompt_source = _invalid_prompt_source(
                        state, instance_id=instance_id
                    )
                    if prompt_source == "tool_output" or invalid_prompt_retries:
                        _end_instance_after_invalid_prompt_tool_retry_limit(
                            state,
                            agent_name=AGENT_NAME,
                            instance_id=instance_id,
                        )
                    elif prompt_source == "instance_task":
                        _drop_instance_task_for_invalid_prompt_skip(
                            state,
                            agent_name=AGENT_NAME,
                            instance_id=instance_id,
                        )
                    status = "skipped"
                    skip_reason = "invalid_prompt_turn_budget_exhausted"
                    error = "invalid_prompt retry exhausted this instance's turn budget"
                    print(
                        f"[{session_id} #{position}/{len(rows)}] {instance_id}: "
                        "skipped after invalid_prompt exhausted turn budget",
                        flush=True,
                    )
                    break
                try:
                    for event in run_agent(agent, state, max_turns=turn_budget):
                        if isinstance(event, ContextCompressionEvent):
                            print(
                                f"[{session_id} #{position}/{len(rows)}] "
                                f"compression {event.before_tokens}"
                                f"->{event.after_tokens}",
                                flush=True,
                            )
                    break
                except Exception as exc:
                    if not _is_invalid_prompt_error(exc):
                        raise
                    prompt_source = _invalid_prompt_source(
                        state, instance_id=instance_id
                    )
                    provider_error = f"{type(exc).__name__}: {exc}"
                    if prompt_source == "instance_task":
                        _drop_instance_task_for_invalid_prompt_skip(
                            state,
                            agent_name=AGENT_NAME,
                            instance_id=instance_id,
                        )
                        status = "skipped"
                        skip_reason = "invalid_prompt_instance_task"
                        error = provider_error
                        print(
                            f"[{session_id} #{position}/{len(rows)}] {instance_id}: "
                            f"skipped after invalid_prompt on instance task",
                            flush=True,
                        )
                        break
                    if prompt_source == "tool_output" or invalid_prompt_retries:
                        if invalid_prompt_retries >= INVALID_PROMPT_TOOL_RETRY_LIMIT:
                            _end_instance_after_invalid_prompt_tool_retry_limit(
                                state,
                                agent_name=AGENT_NAME,
                                instance_id=instance_id,
                            )
                            status = "skipped"
                            skip_reason = "invalid_prompt_tool_output_retry_limit"
                            error = provider_error
                            print(
                                f"[{session_id} #{position}/{len(rows)}] "
                                f"{instance_id}: skipped after "
                                f"{invalid_prompt_retries} invalid_prompt "
                                "tool-output retries",
                                flush=True,
                            )
                            break
                        if not _replace_latest_tool_exchange_for_invalid_prompt(
                            state, agent_name=AGENT_NAME
                        ):
                            _end_instance_after_invalid_prompt_tool_retry_limit(
                                state,
                                agent_name=AGENT_NAME,
                                instance_id=instance_id,
                            )
                            status = "skipped"
                            skip_reason = "invalid_prompt_tool_exchange_not_found"
                            error = provider_error
                            print(
                                f"[{session_id} #{position}/{len(rows)}] "
                                f"{instance_id}: skipped after invalid_prompt "
                                "with no remaining tool exchange to remove",
                                flush=True,
                            )
                            break
                        invalid_prompt_retries += 1
                        print(
                            f"[{session_id} #{position}/{len(rows)}] {instance_id}: "
                            "removed invalid_prompt tool exchange and retrying "
                            f"({invalid_prompt_retries}/"
                            f"{INVALID_PROMPT_TOOL_RETRY_LIMIT})",
                            flush=True,
                        )
                        continue
                    raise
            if status == "ok":
                patch = extract_container_patch(
                    docker,
                    current,
                    language=instance_language(dict(instance)),
                    baseline_commit=baseline,
                )
        except Exception as exc:
            errors += 1
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            print(
                f"[{session_id} #{position}/{len(rows)}] {instance_id}: {error}",
                flush=True,
            )
        finally:
            docker.remove_container(container)
            current.name = ""
            current.workdir = "/app"

        if status == "skipped":
            skipped_record = {
                "repo": repo,
                "session_id": session_id,
                "part_index": session.part_index,
                "part_count": session.part_count,
                "instance_id": instance_id,
                "position": position,
                "reason": skip_reason,
                "error": error,
                "invalid_prompt_retries": invalid_prompt_retries,
            }
            skipped_records.append(skipped_record)

        metrics = summarize_compression(state.events[event_start:])
        result = {
            "model_patch": patch,
            "instance_id": instance_id,
            "repo": repo,
            "session_id": session_id,
            "session_part_index": session.part_index,
            "session_part_count": session.part_count,
            "provider_auth_env": provider_auth_env,
            "status": status,
            "error": error,
            "skip_reason": skip_reason,
            "invalid_prompt_retries": invalid_prompt_retries,
            "baseline_commit": baseline,
            "compression_metrics": metrics.as_dict(),
            "session_event_start": event_start,
            "session_event_end": len(state.events),
        }
        _write_json(paths.output_dir / "result.json", result)
        _write_incremental_predictions(
            predictions_path=predictions_path,
            run_root=run_root,
            run_id=args.run_id,
            model_name=config.model_name,
            dataset_name=config.dataset_name,
            lock=prediction_lock,
        )
        _maybe_write_trace(
            args,
            path=paths.trajectory_jsonl,
            state=state,
            trace_id=f"{args.run_id}.{instance_id}",
            meta={
                "repo": repo,
                "session_id": session_id,
                "part_index": session.part_index,
                "part_count": session.part_count,
                "instance_id": instance_id,
                "position": position,
                "instances_in_session": len(rows),
                "status": status,
                "provider_auth_env": provider_auth_env,
                "compression": config.as_record(),
            },
        )

    repo_dir = run_root / args.run_id / "_repo_sessions" / _safe_path_part(session_id)
    repo_dir.mkdir(parents=True, exist_ok=True)
    _maybe_write_trace(
        args,
        path=repo_dir / "trajectory.jsonl",
        state=state,
        trace_id=f"{args.run_id}.{session_id}",
        meta={
            "repo": repo,
            "session_id": session_id,
            "part_index": session.part_index,
            "part_count": session.part_count,
            "instances": len(rows),
            "errors": errors,
            "skipped": len(skipped_records),
            "provider_auth_env": provider_auth_env,
        },
    )
    if skipped_records:
        _write_jsonl_records(repo_dir / "skipped_instances.jsonl", skipped_records)
    _write_json(
        repo_dir / "summary.json",
        {
            "repo": repo,
            "session_id": session_id,
            "part_index": session.part_index,
            "part_count": session.part_count,
            "instances": len(rows),
            "errors": errors,
            "skipped": len(skipped_records),
            "provider_auth_env": provider_auth_env,
        },
    )
    return {
        "instances": len(rows),
        "errors": errors,
        "skipped": len(skipped_records),
        "provider_auth_env": provider_auth_env,
        "skipped_records": skipped_records,
    }


def _maybe_write_trace(
    args: argparse.Namespace,
    *,
    path: Path,
    state: Any,
    trace_id: str,
    meta: dict[str, Any],
) -> bool:
    """Write a full trajectory only when explicitly requested."""

    if not getattr(args, "write_trajectories", False):
        return False
    trace = run_trace_from_state(
        state=state,
        trace_id=trace_id,
        producer="swebench_pro_repo_session",
        meta=meta,
    )
    write_event_stream(path, trace)
    return True


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


def _session_display_name(session: RepoSessionPart) -> str:
    if session.part_count == 1:
        return session.repo
    return f"{session.repo} part {session.part_index}/{session.part_count}"


def _remaining_turn_budget(events: list[Event], max_turns: int) -> int:
    used = sum(1 for event in events if isinstance(event, TurnStartEvent))
    return max(0, max_turns - used)


def _is_invalid_prompt_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".casefold()
    code = getattr(exc, "code", None)
    status_code = getattr(exc, "status_code", None)
    return (
        "invalid_prompt" in text
        or "invalid prompt" in text
        or "-4321" in text
        or code == -4321
        or status_code == -4321
    )


def _invalid_prompt_source(state: Any, *, instance_id: str) -> InvalidPromptSource:
    """Classify which latest user-visible message caused invalid_prompt."""

    for _, message in reversed(state.active_context_items()):
        if getattr(message, "role", "") != "user":
            continue
        if is_tool_result_message(message):
            return "tool_output"
        if _message_swebench_instance_id(message) == instance_id:
            return "instance_task"
        return "unknown"
    return "unknown"


def _replace_latest_tool_exchange_for_invalid_prompt(
    state: Any, *, agent_name: str
) -> bool:
    """Replace the latest active tool call/result exchange with a safe note."""

    active_items = state.active_context_items()
    tool_result_index: int | None = None
    tool_call_ids: set[str] = set()
    for index, message in reversed(active_items):
        if is_tool_result_message(message):
            tool_result_index = index
            tool_call_ids = {
                block.tool_call_id for block in tool_results_of(message.content)
            }
            break
    if tool_result_index is None:
        return False

    dropped = _tool_exchange_indices(active_items, tool_call_ids)
    if not dropped:
        return False

    replacement = user_message(
        INVALID_PROMPT_TOOL_REMINDER,
        sender="user",
        target=agent_name,
        kind="context",
    )
    state.record(replacement)
    replacement_index = len(state.messages) - 1
    active_context_indices: list[int] = []
    inserted = False
    for index, _ in active_items:
        if index in dropped:
            if not inserted:
                active_context_indices.append(replacement_index)
                inserted = True
            continue
        active_context_indices.append(index)
    state.record_event(
        ContextCompressionEvent(
            agent=agent_name,
            summary_message_index=replacement_index,
            compressed_message_indices=sorted(dropped),
            active_context_indices=active_context_indices,
            before_tokens=0,
            after_tokens=0,
            strategy="invalid-prompt-tool-exchange-replace",
        )
    )
    return True


def _tool_exchange_indices(
    active_items: list[tuple[int, Any]], tool_call_ids: set[str]
) -> set[int]:
    """Return the connected tool-call/tool-result component for call ids."""

    wanted = set(tool_call_ids)
    dropped: set[int] = set()
    changed = True
    while changed:
        changed = False
        for index, message in active_items:
            calls = message_tool_calls(message)
            result_ids = {
                block.tool_call_id for block in tool_results_of(message.content)
            }
            if calls and any(call.id in wanted for call in calls):
                before = len(wanted)
                wanted.update(call.id for call in calls)
                dropped.add(index)
                changed = changed or len(wanted) != before
            if result_ids and result_ids & wanted:
                before = len(wanted)
                wanted.update(result_ids)
                dropped.add(index)
                changed = changed or len(wanted) != before
    return dropped


def _repair_active_tool_pairs(state: Any, *, agent_name: str) -> bool:
    """Drop active tool-call/result orphans before the next provider request."""

    active_items = state.active_context_items()
    kept = _tool_pair_safe_indices(active_items)
    if len(kept) == len(active_items):
        return False

    dropped = [index for index, _ in active_items if index not in set(kept)]
    note = user_message(
        "Removed an incomplete tool call/tool result exchange from context.",
        sender="user",
        target=agent_name,
        kind="context",
    )
    state.record(note)
    note_index = len(state.messages) - 1
    state.record_event(
        ContextCompressionEvent(
            agent=agent_name,
            summary_message_index=note_index,
            compressed_message_indices=dropped,
            active_context_indices=[*kept, note_index],
            before_tokens=0,
            after_tokens=0,
            strategy="tool-pair-orphan-repair",
        )
    )
    return True


def _tool_pair_safe_indices(active_items: list[tuple[int, Any]]) -> list[int]:
    remaining = {index for index, _ in active_items}
    messages = dict(active_items)
    changed = True
    while changed:
        changed = False
        call_ids = {
            call.id
            for index in remaining
            for call in message_tool_calls(messages[index])
        }
        result_ids = {
            block.tool_call_id
            for index in remaining
            for block in tool_results_of(messages[index].content)
        }
        drop: set[int] = set()
        for index in remaining:
            calls = message_tool_calls(messages[index])
            if calls and any(call.id not in result_ids for call in calls):
                drop.add(index)
            results = tool_results_of(messages[index].content)
            if results and any(block.tool_call_id not in call_ids for block in results):
                drop.add(index)
        if drop:
            remaining -= drop
            changed = True
    return [index for index, _ in active_items if index in remaining]


def _drop_instance_task_for_invalid_prompt_skip(
    state: Any, *, agent_name: str, instance_id: str
) -> bool:
    """Drop a skipped problem statement from active context."""

    active_items = state.active_context_items()
    target_index: int | None = None
    for index, message in reversed(active_items):
        if (
            getattr(message, "role", "") == "user"
            and _message_swebench_instance_id(message) == instance_id
        ):
            target_index = index
            break
    if target_index is None:
        return False

    state.record_event(
        ContextCompressionEvent(
            agent=agent_name,
            summary_message_index=target_index,
            compressed_message_indices=[target_index],
            active_context_indices=[
                index for index, _ in active_items if index != target_index
            ],
            before_tokens=0,
            after_tokens=0,
            strategy="invalid-prompt-instance-task-drop",
        )
    )
    return True


def _end_instance_after_invalid_prompt_tool_retry_limit(
    state: Any, *, agent_name: str, instance_id: str
) -> bool:
    """Clear active context after persistent invalid_prompt for one instance."""

    active_items = state.active_context_items()
    if not active_items:
        return False

    end_message = user_message(
        INVALID_PROMPT_INSTANCE_END_MESSAGE.format(instance_id=instance_id),
        sender="user",
        target=agent_name,
        kind="context",
    )
    state.record(end_message)
    end_message_index = len(state.messages) - 1
    state.record_event(
        ContextCompressionEvent(
            agent=agent_name,
            summary_message_index=end_message_index,
            compressed_message_indices=[index for index, _ in active_items],
            active_context_indices=[],
            before_tokens=0,
            after_tokens=0,
            strategy="invalid-prompt-clear-context",
        )
    )
    return True


def _message_swebench_instance_id(message: Any) -> str:
    details = getattr(message, "sidecar", {}).get("details", {})
    if not isinstance(details, Mapping):
        return ""
    swebench = details.get("swebench", {})
    if not isinstance(swebench, Mapping):
        return ""
    return str(swebench.get("instance_id") or "")


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


def _resolve_parallel(value: str, *, session_count: int) -> int:
    if value in {"parts", "repos"}:
        return max(1, session_count)
    try:
        parsed = int(value)
    except ValueError:
        raise SystemExit(
            "--parallel must be 'parts', 'repos', or a positive integer"
        ) from None
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
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _safe_path_part(value: str) -> str:
    return "".join(char if char.isalnum() or char in "_.-" else "_" for char in value)


if __name__ == "__main__":
    main()
