"""Run GDPVal GSB judge from an existing solver run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evals.gdpval.load_instances import (  # noqa: E402
    DEFAULT_HF_DATASET,
    DEFAULT_HF_SPLIT,
    load_instances,
)
from evals.gdpval.judge_suite import GdpvalGsbJudgeSuite  # noqa: E402
from runs import run_gdpval  # noqa: E402
from simple_agent_lab.evals import (  # noqa: E402
    RESULT_KEY,
    InstanceResult,
    RunArtifacts,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-run-id", required=True)
    parser.add_argument(
        "--judge-run-id",
        default=f"gdpval-judge-existing-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    parser.add_argument("--run-root", default=str(ROOT / "evals/out/gdpval"))
    parser.add_argument(
        "--image", default="hub.byted.org/boyuan/gdpval-agent-base:latest"
    )
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--prepare-concurrency", type=int, default=None)
    parser.add_argument("--prepare-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument(
        "--task-ids-file",
        default=None,
        help=(
            "Optional text file of GDPVal task ids. When set, only matching "
            "solver artifacts are judged, in file order."
        ),
    )
    parser.add_argument("--judge-max-turns", type=int, default=50)
    parser.add_argument("--judge-semantic-max-attempts", type=int, default=2)
    parser.add_argument(
        "--judge-api-kind",
        choices=["openai-chat", "openai-responses"],
        default="openai-chat",
    )
    parser.add_argument(
        "--judge-tool-mode", choices=["local", "mcp", "hybrid"], default="hybrid"
    )
    parser.add_argument(
        "--pull", choices=["missing", "always", "never"], default="never"
    )
    parser.add_argument("--network-mode", default="host")
    parser.add_argument("--platform", default=None)
    parser.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET)
    parser.add_argument("--hf-split", default=DEFAULT_HF_SPLIT)
    parser.add_argument(
        "--hf-cache-dir", default=str(ROOT / "evals/out/gdpval/hf_cache")
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_root = Path(args.run_root).resolve()
    solver_dir = run_root / args.solver_run_id
    if not solver_dir.is_dir():
        raise SystemExit(f"solver output dir not found: {solver_dir}")

    solver_results = _load_solver_results(solver_dir)
    requested_task_ids = run_gdpval._task_ids_from_args(args)
    if requested_task_ids:
        solver_results, missing_task_ids = _filter_solver_results_by_task_ids(
            solver_results,
            requested_task_ids,
        )
        print(
            f"==> task ids selected from file/cli: {len(solver_results)}",
            flush=True,
        )
        if missing_task_ids:
            print(
                f"==> task ids without solver artifacts: {len(missing_task_ids)}",
                flush=True,
            )
        if not solver_results:
            raise SystemExit("No solver artifacts matched the requested task ids.")
    task_ids = [result.instance_id for result in solver_results]
    print(f"==> discovered solver artifacts: {len(solver_results)}", flush=True)

    runner_args = _runner_args(args, run_root=run_root, task_ids=task_ids)
    wheelhouse = run_gdpval._wheelhouse_for(runner_args, run_root=run_root)
    print(f"==> refreshing project wheelhouse: {wheelhouse}", flush=True)
    run_gdpval._prepare_wheelhouse_for_run(runner_args, wheelhouse)

    print("==> loading GDPVal source rows from Hugging Face/cache", flush=True)
    source_instances = load_instances(
        None,
        task_ids=task_ids,
        require_deliverables=True,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        hf_cache_dir=args.hf_cache_dir,
    )
    source_by_id = {str(item["instance_id"]): item for item in source_instances}
    print(f"==> loaded source rows: {len(source_by_id)}", flush=True)

    suite = GdpvalGsbJudgeSuite(
        image=args.image,
        reference_root=None,
        deliverable_root=None,
        network_mode=args.network_mode or None,
        platform=args.platform,
        judge_tool_mode=args.judge_tool_mode,
    )
    judge_instances, skipped = _build_judge_instances(
        suite=suite,
        solver_results=solver_results,
        source_by_id=source_by_id,
        concurrency=args.prepare_concurrency or args.concurrency,
        timeout_seconds=args.prepare_timeout_seconds,
    )

    provider_env = _judge_provider_env(judge_api_kind=args.judge_api_kind)
    print("", flush=True)
    print("==> Running GDPVal judge from existing artifacts", flush=True)
    print(f"    selected:    {len(judge_instances)}", flush=True)
    print(f"    skipped:     {len(skipped)}", flush=True)
    print(f"    run-id:      {args.judge_run_id}", flush=True)
    print("    mode:        gsb", flush=True)
    print(f"    tools:       {args.judge_tool_mode}", flush=True)
    print("    provider:    openai", flush=True)
    print(f"    api-kind:    {args.judge_api_kind}", flush=True)
    print(f"    model:       {provider_env.get('OPENAI_MODEL', '<unset>')}", flush=True)
    print(f"    concurrency: {args.concurrency}", flush=True)
    print("", flush=True)

    def on_judge_result(result: InstanceResult) -> None:
        status = "ok" if result.ok else "error"
        if result.artifacts is not None:
            status = f"status={result.artifacts.status_code}"
        print(f"[judge {status}] {result.instance_id}", flush=True)
        run_gdpval._persist_backend_log(result.artifacts)

    judge_results, judge_run_ids, attempt_counts, semantic_histories = (
        run_gdpval._run_judge_with_semantic_retries(
            args=runner_args,
            suite=suite,
            judge_instances=judge_instances,
            run_root=run_root,
            base_judge_run_id=args.judge_run_id,
            wheelhouse=wheelhouse,
            judge_provider="openai",
            judge_api_kind=args.judge_api_kind,
            provider_env=provider_env,
            on_judge_result=on_judge_result,
        )
    )
    summary = run_gdpval._write_judge_summary(
        run_root=run_root,
        solver_run_id=args.solver_run_id,
        judge_run_id=args.judge_run_id,
        judge_run_ids=judge_run_ids,
        judge_mode="gsb",
        results=judge_results,
        skipped=skipped,
        attempt_counts=attempt_counts,
        semantic_histories=semantic_histories,
    )
    print("", flush=True)
    print(f"==> judge summary: {summary}", flush=True)


def _load_solver_results(solver_dir: Path) -> list[InstanceResult]:
    results: list[InstanceResult] = []
    for result_path in sorted(solver_dir.glob("*/out/result.json")):
        run_dir = result_path.parents[1]
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        task_id = run_dir.name
        results.append(
            InstanceResult(
                instance_id=task_id,
                artifacts=RunArtifacts(
                    instance_id=task_id,
                    run_dir=run_dir,
                    trajectory_path=run_dir / "out" / "trajectory.jsonl",
                    status_code=0,
                    logs="",
                ),
                error=None,
                attempts=int(payload.get("attempts") or 1),
            )
        )
    if not results:
        raise SystemExit(f"no solver result.json files found under {solver_dir}")
    return results


def _filter_solver_results_by_task_ids(
    solver_results: list[InstanceResult],
    task_ids: list[str],
) -> tuple[list[InstanceResult], list[str]]:
    by_id = {str(result.instance_id): result for result in solver_results}
    selected: list[InstanceResult] = []
    missing: list[str] = []
    for task_id in task_ids:
        result = by_id.get(str(task_id))
        if result is None:
            missing.append(str(task_id))
            continue
        selected.append(result)
    return selected, missing


def _build_judge_instances(
    *,
    suite: GdpvalGsbJudgeSuite,
    solver_results: list[InstanceResult],
    source_by_id: dict[str, dict[str, Any]],
    concurrency: int,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    skipped: list[tuple[str, str]] = []

    def build(result: InstanceResult) -> tuple[str, dict[str, Any] | None, str]:
        artifacts = result.artifacts
        if artifacts is None or artifacts.status_code != 0:
            return result.instance_id, None, "solver_status_nonzero"
        result_path = artifacts.run_dir / RESULT_KEY
        if not result_path.is_file():
            return result.instance_id, None, "solver_result_missing"
        source = source_by_id.get(result.instance_id)
        if source is None:
            return result.instance_id, None, "source_instance_missing"
        candidate_result = json.loads(result_path.read_text(encoding="utf-8"))
        return (
            result.instance_id,
            suite.build_instance(
                source,
                candidate_result=candidate_result,
                candidate_artifacts=artifacts,
            ),
            "",
        )

    built_by_id: dict[str, dict[str, Any]] = {}
    pool = ThreadPoolExecutor(max_workers=max(1, concurrency))
    futures = {pool.submit(build, result): result for result in solver_results}
    completed: set[Any] = set()
    try:
        for index, future in enumerate(
            as_completed(futures, timeout=max(1.0, timeout_seconds)),
            start=1,
        ):
            completed.add(future)
            task_id, instance, reason = future.result()
            if instance is None:
                skipped.append((task_id, reason))
            else:
                built_by_id[task_id] = instance
            if index % 10 == 0 or index == len(futures):
                print(
                    f"==> prepared judge inputs: {index}/{len(futures)}",
                    flush=True,
                )
    except TimeoutError:
        for future, result in futures.items():
            if future in completed:
                continue
            future.cancel()
            skipped.append((result.instance_id, "prepare_timeout"))
        print(
            "==> prepared judge inputs timed out; "
            f"completed={len(completed)}/{len(futures)} "
            f"timeout_seconds={timeout_seconds:g}",
            flush=True,
        )
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return (
        [
            built_by_id[result.instance_id]
            for result in solver_results
            if result.instance_id in built_by_id
        ],
        skipped,
    )


def _runner_args(args: argparse.Namespace, *, run_root: Path, task_ids: list[str]):
    return SimpleNamespace(
        backend="local-docker",
        run_root=str(run_root),
        wheelhouse=None,
        wheelhouse_mount=run_gdpval.DEFAULT_WHEELHOUSE_MOUNT,
        prepare_wheelhouse=False,
        uv_binary=None,
        run_id=args.solver_run_id,
        pull=args.pull,
        keep_container=False,
        workspace_root=None,
        image=args.image,
        reference_root=None,
        deliverable_root=None,
        network_mode=args.network_mode,
        platform=args.platform,
        judge_tool_mode=args.judge_tool_mode,
        judge_mode="gsb",
        judge_run_id=args.judge_run_id,
        judge_concurrency=args.concurrency,
        concurrency=args.concurrency,
        judge_max_attempts=1,
        judge_semantic_max_attempts=args.judge_semantic_max_attempts,
        judge_max_turns=args.judge_max_turns,
        judge_provider="openai",
        judge_api_kind=args.judge_api_kind,
        provider="openai",
        api_kind="openai-responses",
        max_turns=100,
        max_attempts=1,
        input=None,
        task_ids=task_ids,
        task_ids_file=None,
        limit=None,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        hf_cache_dir=args.hf_cache_dir,
        include_empty_deliverables=False,
        dotenv="",
        solver_model=None,
        solver_api_key=None,
        solver_base_url=None,
        solver_session_id=None,
        solver_log_id=None,
        solver_azure_endpoint=None,
        solver_azure_api_version=None,
        solver_azure_logid=None,
        judge_model=None,
        judge_api_key=None,
        judge_base_url=None,
        judge_session_id=None,
        judge_log_id=None,
        judge_azure_endpoint=None,
        judge_azure_api_version=None,
        judge_azure_logid=None,
    )


def _judge_provider_env(*, judge_api_kind: str = "openai-chat") -> dict[str, str]:
    names = (
        "OPENAI_MODEL",
        "OPENAI_AUTH_TOKEN",
        "OPENAI_BASE_URL",
        "OPENAI_SESSION_ID",
        "OPENAI_LOG_ID",
        "OPENAI_REASONING_EFFORT",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_LOGID",
        "SAL_LLM_TIMEOUT_SECONDS",
        "GDPVAL_GSB_DIRECTION_ATTEMPTS",
    )
    env = {name: os.environ[name] for name in names if os.environ.get(name)}
    missing = [
        name for name in ("OPENAI_MODEL", "OPENAI_AUTH_TOKEN") if name not in env
    ]
    if missing:
        raise SystemExit(f"missing judge provider env keys: {', '.join(missing)}")
    if judge_api_kind == "openai-responses" and env.get("OPENAI_BASE_URL"):
        env["OPENAI_BASE_URL"] = _responses_base_url_for_sdk(env["OPENAI_BASE_URL"])
    return env


def _responses_base_url_for_sdk(base_url: str) -> str:
    """Accept either a Responses endpoint or the SDK base URL.

    ``OpenAI(...).responses.create(...)`` appends ``/responses`` internally.
    Some provider docs publish the full endpoint ending in ``/responses``; trim
    that suffix before passing it as ``base_url`` to avoid
    ``/responses/responses``.
    """

    stripped = base_url.rstrip("/")
    suffix = "/responses"
    if stripped.endswith(suffix):
        return stripped[: -len(suffix)]
    return base_url


if __name__ == "__main__":
    main()
