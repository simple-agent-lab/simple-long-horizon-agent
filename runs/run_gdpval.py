"""Run GDPVal instances through the Simple Agent Lab suite framework."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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
    KNOWN_BAD_TASK_IDS,
    load_instances,
)
from evals.gdpval.judge_suite import GdpvalGsbJudgeSuite, GdpvalJudgeSuite  # noqa: E402
from evals.gdpval.suite import GdpvalSuite  # noqa: E402
from simple_agent_lab.evals import (  # noqa: E402
    RESULT_KEY,
    LocalDirStore,
    LocalDockerBackend,
    LocalProcessBackend,
    InstanceResult,
    RunArtifacts,
    run_dataset,
    run_suite_instance,
)

DEFAULT_WHEELHOUSE_MOUNT = "/agent/wheelhouse"
DEFAULT_GDPVAL_IMAGE = "hub.byted.org/boyuan/gdpval-agent-base:latest"
JUDGE_SUCCESS_STATUSES = {
    "candidate_deliverables_missing",
    "judged",
    "gsb_judged",
    "no_rubrics",
}
PROVIDER_ENV_FIELDS = (
    ("OPENAI_MODEL", "model"),
    ("OPENAI_AUTH_TOKEN", "api_key"),
    ("OPENAI_BASE_URL", "base_url"),
    ("OPENAI_SESSION_ID", "session_id"),
    ("OPENAI_LOG_ID", "log_id"),
    ("OPENAI_REASONING_EFFORT", "reasoning_effort"),
    ("AZURE_OPENAI_ENDPOINT", "azure_endpoint"),
    ("AZURE_OPENAI_API_VERSION", "azure_api_version"),
    ("AZURE_OPENAI_LOGID", "azure_logid"),
)
WEB_TOOL_ENV_FIELDS = (
    "SERPER_API_KEY",
    "SERPER_ENDPOINT",
    "JINA_API_KEY",
    "JINA_ENDPOINT",
)


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
    parser.add_argument(
        "--include-known-bad-tasks",
        action="store_true",
        help=(
            "Include GDPVal rows known to have unreadable or corrupted gold "
            "deliverables. They are skipped by default."
        ),
    )
    parser.add_argument("--reference-root", default=None)
    parser.add_argument("--image", default=DEFAULT_GDPVAL_IMAGE)
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
            "deliverable_files; 'rubric' uses the direct rubric score."
        ),
    )
    parser.add_argument(
        "--judge-tool-mode",
        choices=["local", "mcp", "hybrid"],
        default="hybrid",
        help=(
            "Tool surface for the judge. 'hybrid' keeps local GDPVal tools and "
            "adds local stdio MCP servers when available."
        ),
    )
    parser.add_argument("--judge-run-id", default=None)
    parser.add_argument("--judge-max-turns", type=int, default=50)
    parser.add_argument("--judge-concurrency", type=int, default=None)
    parser.add_argument("--judge-max-attempts", type=int, default=1)
    parser.add_argument(
        "--judge-semantic-max-attempts",
        type=int,
        default=2,
        help=(
            "Maximum GDPVal judge semantic attempts per instance. Retries judge "
            "runs whose result status is not a scored judge status."
        ),
    )
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
    web_tools_group = parser.add_mutually_exclusive_group()
    web_tools_group.add_argument(
        "--enable-web-tools",
        dest="enable_web_tools",
        action="store_true",
        default=True,
        help="Expose Serper WebSearch and Jina WebFetch to the solver. This is the default.",
    )
    web_tools_group.add_argument(
        "--disable-web-tools",
        dest="enable_web_tools",
        action="store_false",
        help="Do not expose WebSearch/WebFetch to the solver.",
    )
    solver_group = parser.add_argument_group("solver provider overrides")
    solver_group.add_argument(
        "--solver-model",
        "--model",
        dest="solver_model",
        default=None,
        help="Solver model. Overrides OPENAI_MODEL.",
    )
    solver_group.add_argument(
        "--solver-api-key",
        "--api-key",
        dest="solver_api_key",
        default=None,
        help="Solver API key/token. Overrides OPENAI_AUTH_TOKEN.",
    )
    solver_group.add_argument(
        "--solver-base-url",
        "--base-url",
        dest="solver_base_url",
        default=None,
        help="Solver OpenAI-compatible base URL. Overrides OPENAI_BASE_URL.",
    )
    solver_group.add_argument(
        "--solver-session-id",
        default=None,
        help="Solver session id. Overrides OPENAI_SESSION_ID.",
    )
    solver_group.add_argument(
        "--solver-log-id",
        default=None,
        help="Solver log id. Overrides OPENAI_LOG_ID.",
    )
    solver_group.add_argument(
        "--solver-azure-endpoint",
        default=None,
        help="Solver Azure OpenAI endpoint. Overrides AZURE_OPENAI_ENDPOINT.",
    )
    solver_group.add_argument(
        "--solver-azure-api-version",
        default=None,
        help="Solver Azure OpenAI API version. Overrides AZURE_OPENAI_API_VERSION.",
    )
    solver_group.add_argument(
        "--solver-azure-logid",
        default=None,
        help="Solver Azure OpenAI log id. Overrides AZURE_OPENAI_LOGID.",
    )
    judge_group = parser.add_argument_group("judge provider overrides")
    judge_group.add_argument(
        "--judge-model",
        default=None,
        help="Judge model. Defaults to the solver model.",
    )
    judge_group.add_argument(
        "--judge-api-key",
        default=None,
        help="Judge API key/token. Defaults to the solver API key/token.",
    )
    judge_group.add_argument(
        "--judge-base-url",
        default=None,
        help="Judge OpenAI-compatible base URL. Defaults to the solver base URL.",
    )
    judge_group.add_argument(
        "--judge-session-id",
        default=None,
        help="Judge session id. Defaults to the solver session id.",
    )
    judge_group.add_argument(
        "--judge-log-id",
        default=None,
        help="Judge log id. Defaults to the solver log id.",
    )
    judge_group.add_argument(
        "--judge-azure-endpoint",
        default=None,
        help="Judge Azure OpenAI endpoint. Defaults to the solver endpoint.",
    )
    judge_group.add_argument(
        "--judge-azure-api-version",
        default=None,
        help="Judge Azure OpenAI API version. Defaults to the solver API version.",
    )
    judge_group.add_argument(
        "--judge-azure-logid",
        default=None,
        help="Judge Azure OpenAI log id. Defaults to the solver log id.",
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
        include_known_bad=args.include_known_bad_tasks,
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
        enable_web_tools=args.enable_web_tools,
    )
    backend = _backend_for(args, run_id=args.run_id, wheelhouse=wheelhouse)
    base_solver_provider_env = _provider_env(args, stage="solver")
    solver_provider_env = {**base_solver_provider_env, **_web_tool_env()}
    judge_provider_env = _provider_env(
        args,
        stage="judge",
        base=base_solver_provider_env,
    )

    print("==> Running GDPVal solver")
    source = args.input or f"huggingface:{args.hf_dataset}/{args.hf_split}"
    print(f"    input:       {source}")
    if args.input is None:
        print(f"    hf-cache:    {args.hf_cache_dir}")
    print(f"    deliverable filter: {not args.include_empty_deliverables}")
    print(f"    known-bad filter:   {not args.include_known_bad_tasks}")
    if not args.include_known_bad_tasks:
        print(f"    known-bad skipped:  {len(KNOWN_BAD_TASK_IDS)}")
    print(f"    selected:    {len(instances)}")
    print(f"    run-id:      {args.run_id}")
    print(f"    backend:     {args.backend}")
    print(f"    provider:    {args.provider}")
    print(f"    api-kind:    {args.api_kind}")
    print(f"    model:       {_provider_model_label(solver_provider_env)}")
    print(f"    web tools:   {args.enable_web_tools}")
    print(f"    max-turns:   {args.max_turns}")
    print(f"    output root: {run_root}")
    print("")

    streaming_judge = None
    if args.judge:
        streaming_judge = _start_streaming_judge_phase(
            args=args,
            source_instances=instances,
            run_root=run_root,
            provider_env=judge_provider_env,
        )

    def on_result(result) -> None:
        status = "ok" if result.ok else "error"
        if result.artifacts is not None:
            status = f"status={result.artifacts.status_code}"
        print(f"[{status}] {result.instance_id}")
        _persist_backend_log(result.artifacts)
        if streaming_judge is not None:
            streaming_judge.submit_solver_result(result)

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
        provider_env=solver_provider_env,
        wheelhouse_mount=args.wheelhouse_mount if wheelhouse else None,
    )
    print("")
    print(f"==> summary: {report.summary()}")
    if streaming_judge is not None:
        streaming_judge.finish()
    failures = [item for item in report.results if not item.ok]
    if failures:
        for item in failures:
            print(f"    failed {item.instance_id}: {item.error}")
        raise SystemExit(1)


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _provider_env(
    args: argparse.Namespace | None = None,
    *,
    stage: str = "solver",
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    env = (
        {
            name: os.environ[name]
            for name, _ in PROVIDER_ENV_FIELDS
            if os.environ.get(name)
        }
        if base is None
        else dict(base)
    )
    if args is None:
        return env
    overrides = _provider_override_attrs(args, stage=stage)
    stage_prefix = stage.upper()
    for name, attr in PROVIDER_ENV_FIELDS:
        prefixed = os.environ.get(f"{stage_prefix}_{name}")
        if prefixed:
            env[name] = prefixed
        value = getattr(args, f"{stage}_{attr}", None)
        if value:
            env[name] = value
    _normalize_provider_env(env, overrides=overrides)
    return env


def _web_tool_env() -> dict[str, str]:
    return {
        name: os.environ[name] for name in WEB_TOOL_ENV_FIELDS if os.environ.get(name)
    }


def _provider_override_attrs(args: argparse.Namespace, *, stage: str) -> set[str]:
    attrs: set[str] = set()
    stage_prefix = stage.upper()
    for name, attr in PROVIDER_ENV_FIELDS:
        if os.environ.get(f"{stage_prefix}_{name}"):
            attrs.add(attr)
        if getattr(args, f"{stage}_{attr}", None):
            attrs.add(attr)
    return attrs


def _normalize_provider_env(env: dict[str, str], *, overrides: set[str]) -> None:
    azure_attrs = {"azure_endpoint", "azure_api_version", "azure_logid"}
    if overrides & azure_attrs and "base_url" not in overrides:
        env.pop("OPENAI_BASE_URL", None)
    if "base_url" in overrides and not (overrides & azure_attrs):
        env.pop("AZURE_OPENAI_ENDPOINT", None)
        env.pop("AZURE_OPENAI_API_VERSION", None)
        env.pop("AZURE_OPENAI_LOGID", None)


def _provider_model_label(provider_env: dict[str, str]) -> str:
    return provider_env.get("OPENAI_MODEL", "") or "<unset>"


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
        judge_tool_mode=args.judge_tool_mode,
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
    print(f"    tools:       {args.judge_tool_mode}")
    print(f"    provider:    {judge_provider}")
    print(f"    api-kind:    {judge_api_kind}")
    print(f"    model:       {_provider_model_label(provider_env)}")
    print(f"    max-turns:   {args.judge_max_turns}")
    print("")
    if skipped:
        for task_id, reason in skipped:
            print(f"    skipped {task_id}: {reason}")
    if not judge_instances:
        print("==> judge summary: no judge instances selected")
        return

    wheelhouse = _wheelhouse_for(args, run_root=run_root)

    def on_judge_result(result) -> None:
        status = "ok" if result.ok else "error"
        if result.artifacts is not None:
            status = f"status={result.artifacts.status_code}"
        print(f"[judge {status}] {result.instance_id}")
        _persist_backend_log(result.artifacts)

    judge_results, judge_run_ids, attempt_counts, semantic_histories = (
        _run_judge_with_semantic_retries(
            args=args,
            suite=suite,
            judge_instances=judge_instances,
            run_root=run_root,
            base_judge_run_id=judge_run_id,
            wheelhouse=wheelhouse,
            judge_provider=judge_provider,
            judge_api_kind=judge_api_kind,
            provider_env=provider_env,
            on_judge_result=on_judge_result,
        )
    )
    summary = _write_judge_summary(
        run_root=run_root,
        solver_run_id=args.run_id,
        judge_run_id=judge_run_id,
        judge_run_ids=judge_run_ids,
        judge_mode=args.judge_mode,
        results=judge_results,
        skipped=skipped,
        attempt_counts=attempt_counts,
        semantic_histories=semantic_histories,
    )
    print("")
    print(f"==> judge summary: {summary}")


def _start_streaming_judge_phase(
    *,
    args: argparse.Namespace,
    source_instances: list[dict],
    run_root: Path,
    provider_env: dict[str, str],
) -> "_StreamingJudgePhase":
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
        judge_tool_mode=args.judge_tool_mode,
    )
    phase = _StreamingJudgePhase(
        args=args,
        suite=suite,
        source_by_id={str(item["instance_id"]): item for item in source_instances},
        run_root=run_root,
        base_judge_run_id=judge_run_id,
        wheelhouse=_wheelhouse_for(args, run_root=run_root),
        judge_provider=judge_provider,
        judge_api_kind=judge_api_kind,
        provider_env=provider_env,
    )
    print("")
    print("==> Running GDPVal judge")
    print("    schedule:    streaming")
    print("    selected:    as solver results finish")
    print(f"    run-id:      {judge_run_id}")
    print(f"    mode:        {args.judge_mode}")
    print(f"    tools:       {args.judge_tool_mode}")
    print(f"    provider:    {judge_provider}")
    print(f"    api-kind:    {judge_api_kind}")
    print(f"    model:       {_provider_model_label(provider_env)}")
    print(f"    max-turns:   {args.judge_max_turns}")
    print(f"    concurrency: {phase.concurrency}")
    print("")
    return phase


class _StreamingJudgePhase:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        suite,
        source_by_id: dict[str, dict],
        run_root: Path,
        base_judge_run_id: str,
        wheelhouse: Path | None,
        judge_provider: str,
        judge_api_kind: str,
        provider_env: dict[str, str],
    ) -> None:
        self.args = args
        self.suite = suite
        self.source_by_id = source_by_id
        self.run_root = run_root
        self.base_judge_run_id = base_judge_run_id
        self.wheelhouse = wheelhouse
        self.judge_provider = judge_provider
        self.judge_api_kind = judge_api_kind
        self.provider_env = provider_env
        self.concurrency = max(1, int(args.judge_concurrency or args.concurrency or 1))
        self.pool = ThreadPoolExecutor(max_workers=self.concurrency)
        self.futures = {}
        self.results: list[InstanceResult] = []
        self.skipped: list[tuple[str, str]] = []
        self.attempt_counts: dict[str, int] = {}
        self.semantic_histories: dict[str, list[str]] = {}
        self.run_ids: list[str] = []
        self.seen_run_ids: set[str] = set()

    def submit_solver_result(self, result: InstanceResult) -> None:
        instance, skipped = _build_judge_instance_from_solver_result(
            suite=self.suite,
            source_by_id=self.source_by_id,
            result=result,
        )
        if skipped is not None:
            task_id, reason = skipped
            self.skipped.append(skipped)
            print(f"    judge skipped {task_id}: {reason}", flush=True)
            return
        assert instance is not None
        future = self.pool.submit(
            _run_streaming_judge_instance_with_semantic_retries,
            args=self.args,
            suite=self.suite,
            instance=instance,
            run_root=self.run_root,
            base_judge_run_id=self.base_judge_run_id,
            wheelhouse=self.wheelhouse,
            judge_provider=self.judge_provider,
            judge_api_kind=self.judge_api_kind,
            provider_env=self.provider_env,
            on_judge_result=_print_judge_result,
        )
        self.futures[future] = str(instance["instance_id"])

    def finish(self) -> dict[str, object] | None:
        pending = dict(self.futures)
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                task_id = pending.pop(future)
                try:
                    item, run_ids, attempts, history = future.result()
                except Exception as exc:
                    item = InstanceResult(
                        task_id,
                        None,
                        f"{type(exc).__name__}: {exc}",
                        attempts=1,
                    )
                    run_ids = [self.base_judge_run_id]
                    attempts = 1
                    history = ["error"]
                self.results.append(item)
                self.attempt_counts[task_id] = attempts
                self.semantic_histories[task_id] = history
                for run_id in run_ids:
                    if run_id not in self.seen_run_ids:
                        self.seen_run_ids.add(run_id)
                        self.run_ids.append(run_id)
        self.pool.shutdown(wait=True)
        if not self.results and not self.skipped:
            print("==> judge summary: no judge instances selected")
            return None
        summary = _write_judge_summary(
            run_root=self.run_root,
            solver_run_id=self.args.run_id,
            judge_run_id=self.base_judge_run_id,
            judge_run_ids=self.run_ids or [self.base_judge_run_id],
            judge_mode=self.args.judge_mode,
            results=self._ordered_results(),
            skipped=self.skipped,
            attempt_counts=self.attempt_counts,
            semantic_histories=self.semantic_histories,
        )
        print("")
        print(f"==> judge summary: {summary}")
        return summary

    def _ordered_results(self) -> list[InstanceResult]:
        order = {task_id: index for index, task_id in enumerate(self.source_by_id)}
        return sorted(
            self.results,
            key=lambda item: order.get(str(item.instance_id), len(order)),
        )


def _print_judge_result(result: InstanceResult) -> None:
    status = "ok" if result.ok else "error"
    if result.artifacts is not None:
        status = f"status={result.artifacts.status_code}"
    print(f"[judge {status}] {result.instance_id}", flush=True)
    _persist_backend_log(result.artifacts)


def _build_judge_instance_from_solver_result(
    *,
    suite,
    source_by_id: dict[str, dict],
    result: InstanceResult,
) -> tuple[dict | None, tuple[str, str] | None]:
    artifacts = result.artifacts
    if artifacts is None or artifacts.status_code != 0:
        return None, (result.instance_id, "solver_status_nonzero")
    result_path = artifacts.run_dir / RESULT_KEY
    if not result_path.is_file():
        return None, (result.instance_id, "solver_result_missing")
    source = source_by_id.get(str(result.instance_id))
    if source is None:
        return None, (result.instance_id, "source_instance_missing")
    try:
        candidate_result = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, (result.instance_id, "solver_result_invalid_json")
    return (
        suite.build_instance(
            source,
            candidate_result=candidate_result,
            candidate_artifacts=artifacts,
        ),
        None,
    )


def _run_streaming_judge_instance_with_semantic_retries(
    *,
    args: argparse.Namespace,
    suite,
    instance: dict,
    run_root: Path,
    base_judge_run_id: str,
    wheelhouse: Path | None,
    judge_provider: str,
    judge_api_kind: str,
    provider_env: dict[str, str],
    on_judge_result,
) -> tuple[InstanceResult, list[str], int, list[str]]:
    max_semantic_attempts = max(1, int(args.judge_semantic_max_attempts or 1))
    run_ids: list[str] = []
    histories: list[str] = []
    total_attempts = 0
    latest: InstanceResult | None = None
    for semantic_attempt in range(1, max_semantic_attempts + 1):
        run_id = _judge_attempt_run_id(base_judge_run_id, semantic_attempt)
        if run_id not in run_ids:
            run_ids.append(run_id)
        if semantic_attempt > 1:
            print(
                "    semantic retry queued "
                f"{instance['instance_id']}: attempt={semantic_attempt}/"
                f"{max_semantic_attempts}",
                flush=True,
            )
        backend = _backend_for(args, run_id=run_id, wheelhouse=wheelhouse)
        item = _run_one_judge_attempt(
            args=args,
            suite=suite,
            instance=instance,
            backend=backend,
            run_root=run_root,
            run_id=run_id,
            wheelhouse=wheelhouse,
            judge_provider=judge_provider,
            judge_api_kind=judge_api_kind,
            provider_env=provider_env,
        )
        latest = item
        total_attempts += item.attempts
        status = _judge_result_status(item)
        histories.append(status)
        on_judge_result(item)
        if _judge_result_is_semantic_success(item):
            break
    assert latest is not None
    return latest, run_ids, total_attempts, histories


def _run_judge_with_semantic_retries(
    *,
    args: argparse.Namespace,
    suite,
    judge_instances: list[dict],
    run_root: Path,
    base_judge_run_id: str,
    wheelhouse: Path | None,
    judge_provider: str,
    judge_api_kind: str,
    provider_env: dict[str, str],
    on_judge_result,
) -> tuple[list, list[str], dict[str, int], dict[str, list[str]]]:
    pending = deque((instance, 1) for instance in judge_instances)
    latest_by_id: dict[str, InstanceResult] = {}
    attempt_counts: dict[str, int] = {}
    semantic_histories: dict[str, list[str]] = {}
    run_ids: list[str] = []
    seen_run_ids: set[str] = set()
    max_semantic_attempts = max(1, int(args.judge_semantic_max_attempts or 1))
    concurrency = max(1, int(args.judge_concurrency or args.concurrency or 1))

    def note_run_id(semantic_attempt: int) -> str:
        run_id = _judge_attempt_run_id(base_judge_run_id, semantic_attempt)
        if run_id not in seen_run_ids:
            seen_run_ids.add(run_id)
            run_ids.append(run_id)
            if semantic_attempt > 1:
                print("")
                print(
                    "==> Retrying GDPVal judge semantic failures "
                    f"(attempt {semantic_attempt}/{max_semantic_attempts})"
                )
                print(f"    run-id:      {run_id}")
                print("")
        return run_id

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        in_flight = {}

        def submit_ready() -> None:
            while pending and len(in_flight) < concurrency:
                instance, semantic_attempt = pending.popleft()
                attempt_run_id = note_run_id(semantic_attempt)
                backend = _backend_for(
                    args,
                    run_id=attempt_run_id,
                    wheelhouse=wheelhouse,
                )
                future = pool.submit(
                    _run_one_judge_attempt,
                    args=args,
                    suite=suite,
                    instance=instance,
                    backend=backend,
                    run_root=run_root,
                    run_id=attempt_run_id,
                    wheelhouse=wheelhouse,
                    judge_provider=judge_provider,
                    judge_api_kind=judge_api_kind,
                    provider_env=provider_env,
                )
                in_flight[future] = (instance, semantic_attempt)

        submit_ready()
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                instance, semantic_attempt = in_flight.pop(future)
                item = future.result()
                latest_by_id[str(item.instance_id)] = item

                task_id = str(item.instance_id)
                attempt_counts[task_id] = attempt_counts.get(task_id, 0) + item.attempts
                status = _judge_result_status(item)
                semantic_histories.setdefault(task_id, []).append(status)
                on_judge_result(item)

                if (
                    not _judge_result_is_semantic_success(item)
                    and semantic_attempt < max_semantic_attempts
                ):
                    next_attempt = semantic_attempt + 1
                    print(
                        f"    semantic retry queued {item.instance_id}: status={status}"
                    )
                    # Retry before starting more first-pass work so failures do
                    # not wait behind the rest of a long dataset.
                    pending.appendleft((instance, next_attempt))
            submit_ready()

    ordered = [
        latest_by_id[str(instance["instance_id"])]
        for instance in judge_instances
        if str(instance["instance_id"]) in latest_by_id
    ]
    return ordered, run_ids, attempt_counts, semantic_histories


def _run_one_judge_attempt(
    *,
    args: argparse.Namespace,
    suite,
    instance: dict,
    backend,
    run_root: Path,
    run_id: str,
    wheelhouse: Path | None,
    judge_provider: str,
    judge_api_kind: str,
    provider_env: dict[str, str],
) -> InstanceResult:
    instance_id = str(instance["instance_id"])
    last_error: str | None = None
    max_attempts = max(1, int(args.judge_max_attempts or 1))
    for attempt in range(1, max_attempts + 1):
        try:
            artifacts = run_suite_instance(
                suite=suite,
                instance=instance,
                backend=backend,
                store=LocalDirStore(run_root),
                run_root=run_root,
                run_id=run_id,
                provider=judge_provider,
                api_kind=judge_api_kind,
                max_turns=args.judge_max_turns,
                provider_env=provider_env,
                wheelhouse_mount=args.wheelhouse_mount if wheelhouse else None,
            )
            return InstanceResult(instance_id, artifacts, None, attempt)
        except Exception as exc:  # transient/infra failure
            last_error = f"{type(exc).__name__}: {exc}"
    return InstanceResult(instance_id, None, last_error, max_attempts)


def _judge_attempt_run_id(base_judge_run_id: str, semantic_attempt: int) -> str:
    if semantic_attempt <= 1:
        return base_judge_run_id
    return f"{base_judge_run_id}-semantic-retry-{semantic_attempt}"


def _judge_result_status(item) -> str:
    if item.artifacts is None:
        return "error"
    result_path = item.artifacts.run_dir / RESULT_KEY
    if not result_path.is_file():
        return "judge_result_missing"
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "judge_result_invalid_json"
    return str(payload.get("status", "unknown"))


def _judge_result_is_semantic_success(item) -> bool:
    return _judge_result_status(item) in JUDGE_SUCCESS_STATUSES


def _write_judge_summary(
    *,
    run_root: Path,
    solver_run_id: str,
    judge_run_id: str,
    judge_run_ids: list[str] | None = None,
    judge_mode: str,
    results: list,
    skipped: list[tuple[str, str]],
    attempt_counts: dict[str, int] | None = None,
    semantic_histories: dict[str, list[str]] | None = None,
) -> dict[str, object]:
    attempt_counts = attempt_counts or {}
    semantic_histories = semantic_histories or {}
    rows: list[dict[str, object]] = [
        {"task_id": task_id, "status": "skipped", "reason": reason}
        for task_id, reason in skipped
    ]
    for item in results:
        task_id = str(item.instance_id)
        row: dict[str, object] = {
            "task_id": item.instance_id,
            "attempts": attempt_counts.get(task_id, item.attempts),
        }
        history = semantic_histories.get(task_id, [])
        if history:
            row["semantic_attempts"] = len(history)
            row["semantic_status_history"] = history
        if item.artifacts is None:
            row.update({"status": "error", "error": item.error or ""})
        else:
            row["judge_status_code"] = item.artifacts.status_code
            result_path = item.artifacts.run_dir / RESULT_KEY
            row["result_path"] = str(result_path)
            if result_path.is_file():
                try:
                    payload = json.loads(result_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    row.update(
                        {
                            "status": "judge_result_invalid_json",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                else:
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
                        "final_gsb_reverse",
                        "final_gsb_forward",
                    ):
                        if key in payload:
                            row[key] = payload[key]
                    attempts = payload.get("judge_attempts")
                    if isinstance(attempts, list):
                        tool_message_count = 0
                        mcp_tool_count = 0
                        for attempt in attempts:
                            if not isinstance(attempt, dict):
                                continue
                            tool_message_count += int(
                                attempt.get("tool_message_count") or 0
                            )
                            mcp_tool_count += int(attempt.get("mcp_tool_count") or 0)
                        row["tool_message_count"] = tool_message_count
                        row["mcp_tool_count"] = mcp_tool_count
                    if "judge_retry_summary" in payload:
                        row["judge_retry_summary"] = payload["judge_retry_summary"]
            else:
                row["status"] = "judge_result_missing"
        rows.append(row)

    scored = [
        row
        for row in rows
        if row.get("status") in JUDGE_SUCCESS_STATUSES
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
        "judge_run_ids": judge_run_ids or [judge_run_id],
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
