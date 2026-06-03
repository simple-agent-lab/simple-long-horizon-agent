"""Run GDPVal instances through the Simple Agent Lab suite framework."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

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
from evals.gdpval.judge_suite import GdpvalGsbJudgeSuite, GdpvalJudgeSuite  # noqa: E402
from evals.gdpval.suite import GdpvalSuite  # noqa: E402
from simple_agent_lab.evals import (  # noqa: E402
    RESULT_KEY,
    LocalDirStore,
    LocalDockerBackend,
    LocalProcessBackend,
    RunArtifacts,
    run_dataset,
)

DEFAULT_WHEELHOUSE_MOUNT = "/agent/wheelhouse"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Optional GDPVal JSONL, JSON, or Parquet file. Defaults to Hugging Face.",
    )
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET)
    parser.add_argument("--hf-split", default=DEFAULT_HF_SPLIT)
    parser.add_argument(
        "--hf-cache-dir", default=str(ROOT / "evals/out/gdpval/hf_cache")
    )
    parser.add_argument(
        "--include-empty-deliverables",
        action="store_true",
        help="Do not filter out rows whose deliverable_files field is empty.",
    )
    parser.add_argument("--reference-root", default=None)
    parser.add_argument("--image", default="hub.byted.org/apihub/gdpeval:1.0.0")
    parser.add_argument("--network-mode", default="host")
    parser.add_argument("--platform", default=None)
    parser.add_argument("--run-root", default=str(ROOT / "evals/out/gdpval"))
    parser.add_argument(
        "--wheelhouse",
        default=None,
        help=(
            "Host wheelhouse for Docker bootstrap. Defaults to "
            "<run-root>/wheelhouse/cp311-manylinux for local-docker."
        ),
    )
    parser.add_argument(
        "--wheelhouse-mount",
        default=DEFAULT_WHEELHOUSE_MOUNT,
        help="In-container path where --wheelhouse is mounted.",
    )
    parser.add_argument(
        "--prepare-wheelhouse",
        action="store_true",
        help="Download the full provider dependency wheelhouse before the run.",
    )
    parser.add_argument(
        "--uv-binary",
        default=None,
        help="Optional host uv binary to mount at /tmp/uv for Docker runs.",
    )
    parser.add_argument(
        "--run-id", default=f"gdpval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Run a second-stage judge over successful solver outputs.",
    )
    parser.add_argument(
        "--judge-mode",
        choices=["gsb", "rubric"],
        default="gsb",
        help=(
            "Judge style. 'gsb' compares candidate deliverables against "
            "deliverable_files; 'rubric' keeps the legacy direct rubric score."
        ),
    )
    parser.add_argument("--judge-run-id", default=None)
    parser.add_argument("--judge-max-turns", type=int, default=50)
    parser.add_argument("--judge-concurrency", type=int, default=None)
    parser.add_argument("--judge-max-attempts", type=int, default=1)
    parser.add_argument(
        "--judge-provider",
        choices=["fake", "openai", "oracle"],
        default=None,
        help="Judge provider. Defaults to --provider.",
    )
    parser.add_argument(
        "--judge-api-kind",
        choices=["openai-chat", "openai-responses"],
        default=None,
        help="Judge API kind. Defaults to --api-kind.",
    )
    parser.add_argument(
        "--deliverable-root",
        default=None,
        help="Optional local root for GDPVal gold deliverable files.",
    )
    parser.add_argument("--provider", choices=["fake", "openai"], default="openai")
    parser.add_argument(
        "--api-kind",
        choices=["openai-chat", "openai-responses"],
        default=os.environ.get("API_KIND", "openai-responses"),
    )
    parser.add_argument("--dotenv", default=str(ROOT / ".env"))
    parser.add_argument(
        "--backend",
        choices=["local-docker", "local-process"],
        default="local-docker",
        help="Use local-process only for unit/dev smoke runs.",
    )
    parser.add_argument(
        "--workspace-root",
        default=None,
        help="Base directory for local-process workspaces.",
    )
    parser.add_argument(
        "--pull",
        choices=["missing", "always", "never"],
        default="missing",
    )
    parser.add_argument("--keep-container", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.provider == "openai":
        _load_dotenv(Path(args.dotenv))

    instances = load_instances(
        args.input,
        task_ids=args.task_ids,
        limit=args.limit,
        require_deliverables=not args.include_empty_deliverables,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        hf_cache_dir=args.hf_cache_dir,
    )
    if not instances:
        raise SystemExit("No GDPVal instances selected.")

    run_root = Path(args.run_root).resolve()
    wheelhouse = _wheelhouse_for(args, run_root=run_root)
    _prepare_wheelhouse_for_run(args, wheelhouse)
    suite = GdpvalSuite(
        image=args.image,
        reference_root=args.reference_root,
        network_mode=args.network_mode or None,
        platform=args.platform,
    )
    backend = _backend_for(args, run_id=args.run_id, wheelhouse=wheelhouse)
    provider_env = _provider_env()

    print("==> Running GDPVal solver")
    source = args.input or f"huggingface:{args.hf_dataset}/{args.hf_split}"
    print(f"    input:       {source}")
    if args.input is None:
        print(f"    hf-cache:    {args.hf_cache_dir}")
    print(f"    deliverable filter: {not args.include_empty_deliverables}")
    print(f"    selected:    {len(instances)}")
    print(f"    run-id:      {args.run_id}")
    print(f"    backend:     {args.backend}")
    print(f"    provider:    {args.provider}")
    print(f"    api-kind:    {args.api_kind}")
    print(f"    max-turns:   {args.max_turns}")
    print(f"    output root: {run_root}")
    print("")

    def on_result(result) -> None:
        status = "ok" if result.ok else "error"
        if result.artifacts is not None:
            status = f"status={result.artifacts.status_code}"
        print(f"[{status}] {result.instance_id}")
        _persist_backend_log(result.artifacts)

    report = run_dataset(
        suite=suite,
        instances=instances,
        backend=backend,
        store=LocalDirStore(run_root),
        run_root=run_root,
        run_id=args.run_id,
        concurrency=args.concurrency,
        max_attempts=args.max_attempts,
        on_result=on_result,
        provider=args.provider,
        api_kind=args.api_kind,
        max_turns=args.max_turns,
        provider_env=provider_env,
        wheelhouse_mount=args.wheelhouse_mount if wheelhouse else None,
    )
    print("")
    print(f"==> summary: {report.summary()}")
    failures = [item for item in report.results if not item.ok]
    if failures:
        for item in failures:
            print(f"    failed {item.instance_id}: {item.error}")
        raise SystemExit(1)
    if args.judge:
        _run_judge_phase(args, instances, report.results, run_root, provider_env)


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _provider_env() -> dict[str, str]:
    names = (
        "OPENAI_MODEL",
        "OPENAI_AUTH_TOKEN",
        "OPENAI_BASE_URL",
        "OPENAI_SESSION_ID",
        "OPENAI_LOG_ID",
    )
    return {name: os.environ[name] for name in names if os.environ.get(name)}


def _workspace_factory(args: argparse.Namespace, *, run_id: str):
    base_root = (
        Path(args.workspace_root) if args.workspace_root else Path(args.run_root)
    )
    base = base_root / "_local_workspaces" / run_id

    def make_workspace(spec) -> Path:
        return base / spec.instance_id / "workdir"

    return make_workspace


def _wheelhouse_for(args: argparse.Namespace, *, run_root: Path) -> Path | None:
    if args.backend != "local-docker":
        return None
    value = args.wheelhouse or str(run_root / "wheelhouse/cp311-manylinux")
    return Path(value).resolve()


def _prepare_wheelhouse_for_run(
    args: argparse.Namespace, wheelhouse: Path | None
) -> None:
    if wheelhouse is None:
        return
    prepare_all = (
        args.prepare_wheelhouse
        or not wheelhouse.exists()
        or not any(wheelhouse.iterdir())
    )
    from evals.swebench.harness import prepare_wheelhouse_for_run

    prepare_wheelhouse_for_run(wheelhouse, prepare_all=prepare_all)


def _backend_for(
    args: argparse.Namespace,
    *,
    run_id: str,
    wheelhouse: Path | None,
):
    if args.backend == "local-process":
        return LocalProcessBackend(workspace=_workspace_factory(args, run_id=run_id))
    return LocalDockerBackend(
        pull=args.pull,
        keep_container=args.keep_container,
        wheelhouse=wheelhouse,
        uv_binary=args.uv_binary or None,
    )


def _run_judge_phase(
    args: argparse.Namespace,
    source_instances: list[dict],
    solver_results: list,
    run_root: Path,
    provider_env: dict[str, str],
) -> None:
    source_by_id = {str(item["instance_id"]): item for item in source_instances}
    judge_run_id = args.judge_run_id or f"{args.run_id}-judge"
    judge_provider = args.judge_provider or args.provider
    judge_api_kind = args.judge_api_kind or args.api_kind
    suite_cls = GdpvalGsbJudgeSuite if args.judge_mode == "gsb" else GdpvalJudgeSuite
    suite = suite_cls(
        image=args.image,
        reference_root=args.reference_root,
        deliverable_root=args.deliverable_root,
        network_mode=args.network_mode or None,
        platform=args.platform,
    )
    judge_instances = []
    skipped = []
    for result in solver_results:
        artifacts = result.artifacts
        if artifacts is None or artifacts.status_code != 0:
            skipped.append((result.instance_id, "solver_status_nonzero"))
            continue
        result_path = artifacts.run_dir / RESULT_KEY
        if not result_path.is_file():
            skipped.append((result.instance_id, "solver_result_missing"))
            continue
        candidate_result = json.loads(result_path.read_text(encoding="utf-8"))
        source = source_by_id.get(str(result.instance_id))
        if source is None:
            skipped.append((result.instance_id, "source_instance_missing"))
            continue
        judge_instances.append(
            suite.build_instance(
                source,
                candidate_result=candidate_result,
                candidate_artifacts=artifacts,
            )
        )

    print("")
    print("==> Running GDPVal judge")
    print(f"    selected:    {len(judge_instances)}")
    print(f"    skipped:     {len(skipped)}")
    print(f"    run-id:      {judge_run_id}")
    print(f"    mode:        {args.judge_mode}")
    print(f"    provider:    {judge_provider}")
    print(f"    api-kind:    {judge_api_kind}")
    print(f"    max-turns:   {args.judge_max_turns}")
    print("")
    if skipped:
        for task_id, reason in skipped:
            print(f"    skipped {task_id}: {reason}")
    if not judge_instances:
        print("==> judge summary: no judge instances selected")
        return

    wheelhouse = _wheelhouse_for(args, run_root=run_root)
    backend = _backend_for(args, run_id=judge_run_id, wheelhouse=wheelhouse)

    def on_judge_result(result) -> None:
        status = "ok" if result.ok else "error"
        if result.artifacts is not None:
            status = f"status={result.artifacts.status_code}"
        print(f"[judge {status}] {result.instance_id}")
        _persist_backend_log(result.artifacts)

    judge_report = run_dataset(
        suite=suite,
        instances=judge_instances,
        backend=backend,
        store=LocalDirStore(run_root),
        run_root=run_root,
        run_id=judge_run_id,
        concurrency=args.judge_concurrency or args.concurrency,
        max_attempts=args.judge_max_attempts,
        on_result=on_judge_result,
        provider=judge_provider,
        api_kind=judge_api_kind,
        max_turns=args.judge_max_turns,
        provider_env=provider_env,
        wheelhouse_mount=args.wheelhouse_mount if wheelhouse else None,
    )
    summary = _write_judge_summary(
        run_root=run_root,
        solver_run_id=args.run_id,
        judge_run_id=judge_run_id,
        judge_mode=args.judge_mode,
        results=judge_report.results,
        skipped=skipped,
    )
    print("")
    print(f"==> judge summary: {summary}")


def _write_judge_summary(
    *,
    run_root: Path,
    solver_run_id: str,
    judge_run_id: str,
    judge_mode: str,
    results: list,
    skipped: list[tuple[str, str]],
) -> dict[str, object]:
    rows: list[dict[str, object]] = [
        {"task_id": task_id, "status": "skipped", "reason": reason}
        for task_id, reason in skipped
    ]
    for item in results:
        row: dict[str, object] = {
            "task_id": item.instance_id,
            "attempts": item.attempts,
        }
        if item.artifacts is None:
            row.update({"status": "error", "error": item.error or ""})
        else:
            row["judge_status_code"] = item.artifacts.status_code
            result_path = item.artifacts.run_dir / RESULT_KEY
            row["result_path"] = str(result_path)
            if result_path.is_file():
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                row.update(
                    {
                        "status": payload.get("status", "unknown"),
                        "score": payload.get("score", 0.0),
                        "earned_score": payload.get("earned_score", 0.0),
                        "max_score": payload.get("max_score", 0.0),
                    }
                )
                for key in (
                    "combined_weighted_score",
                    "llm_score",
                    "score_process",
                    "dcg_winrate",
                    "rubrics_weighted_score_reverse",
                    "rubrics_weighted_score_forward",
                    "llm_gsb_score_reverse",
                    "llm_gsb_score_forward",
                ):
                    if key in payload:
                        row[key] = payload[key]
            else:
                row["status"] = "judge_result_missing"
        rows.append(row)

    scored = [
        row
        for row in rows
        if row.get("status") in {"judged", "gsb_judged", "no_rubrics"}
        and isinstance(row.get("score"), (int, float))
    ]
    aggregate = {
        "total": len(rows),
        "judged": len(scored),
        "skipped": len(skipped),
        "mean_score": (
            sum(float(row["score"]) for row in scored) / len(scored) if scored else 0.0
        ),
        "judge_run_id": judge_run_id,
        "judge_mode": judge_mode,
    }
    summary_dir = run_root / solver_run_id
    summary_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = summary_dir / "judge_summary.jsonl"
    json_path = summary_dir / "judge_summary.json"
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps({"summary": aggregate, "rows": rows}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return {**aggregate, "summary_path": str(json_path), "jsonl_path": str(jsonl_path)}


def _persist_backend_log(artifacts: RunArtifacts | None) -> None:
    if artifacts is None or artifacts.status_code == 0 or not artifacts.logs:
        return
    log_path = artifacts.run_dir / "out" / "backend.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(artifacts.logs + "\n", encoding="utf-8")
    print(f"    backend-log: {log_path}")


if __name__ == "__main__":
    main()
