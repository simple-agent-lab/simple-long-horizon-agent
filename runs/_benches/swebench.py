"""Run SWE-bench Verified, Multilingual, or Pro through the generic Suite API.

Use ``run_bench.py swebench ID`` for one prepared record or
``run_bench.py batch swebench`` for dataset selection and parallelism.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import namedtuple
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from evals.swebench import harness
from evals.swebench.suite import SwebenchSuite
from runs.lib.container_batch import run_container_batch
from runs.lib import docker_cli
from simple_agent_lab.agent_flavors import (
    AGENT_FLAVOR_ENV,
    WORKFLOW_AGENT_FLAVORS,
)
import simple_agent_lab.config as config
from simple_agent_lab.evals import (
    LocalDirStore,
    run_suite_instance,
)
from simple_agent_lab.evals.runner import (
    canonical_run_id,
    clear_run_outputs,
    container_name,
    prepare_run_directory,
)
from simple_agent_lab.evals.suites.swebench.patch import instance_language
from simple_agent_lab.trace import write_jsonl

ROOT = Path(__file__).resolve().parents[2]
NAME = "swebench"
DESCRIPTION = "SWE-bench instance in a Docker container (single instance per run)."
SCORER = ("-m", "evals.swebench.evaluate_predictions")
DEFAULT_SWEBENCH_DOCKER_TIMEOUT_S = 1800.0


Variant = namedtuple(
    "Variant", "dataset default_instance_id run_root wheelhouse max_turns"
)


VARIANTS = {
    "verified": Variant(
        harness.DEFAULT_DATASET,
        "sympy__sympy-23824",
        harness.DEFAULT_RUN_ROOT,
        harness.DEFAULT_WHEELHOUSE,
        150,
    ),
    "multilingual": Variant(
        harness.DEFAULT_MULTILINGUAL_DATASET,
        "",
        harness.DEFAULT_MULTILINGUAL_RUN_ROOT,
        harness.DEFAULT_MULTILINGUAL_WHEELHOUSE,
        150,
    ),
    "pro": Variant(
        "ScaleAI/SWE-bench_Pro",
        "instance_navidrome__navidrome-8e640bb8580affb7e0ea6225c0bbe240186b6b08",
        harness.DEFAULT_PRO_RUN_ROOT,
        harness.DEFAULT_PRO_WHEELHOUSE,
        250,
    ),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance_id")
    parser.add_argument("--profile", help="JSON run-profile path.")
    parser.add_argument(
        "--instance-json",
        help="Defaults to evals/out/swebench/instance_<id>.jsonl",
    )
    parser.add_argument("--dataset-name", default=harness.DEFAULT_DATASET)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument(
        "--run-id", default=f"swebench-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    parser.add_argument(
        "--agent-flavor",
        choices=harness.AGENT_FLAVOR_CHOICES,
        default=harness.DEFAULT_AGENT_FLAVOR,
        help="Agent flavor; workflow arms treat --max-turns as a worker budget.",
    )
    parser.add_argument("--pdr-rounds", type=int)
    parser.add_argument("--pdr-width", type=int)
    parser.add_argument("--loop-max-turns", type=int)
    parser.add_argument(
        "--provider", choices=["fake", "openai", "oracle"], default="openai"
    )
    parser.add_argument("--api-kind")
    parser.add_argument("--namespace", default="swebench")
    docker_cli.add_arguments(
        parser,
        default_uv_binary=harness.DEFAULT_UV_BINARY,
        default_timeout_seconds=DEFAULT_SWEBENCH_DOCKER_TIMEOUT_S,
    )
    parser.add_argument("--dotenv", default=str(ROOT / ".env"))
    parser.add_argument(
        "--reuse-prepared-wheelhouse",
        action="store_true",
        help="Skip wheelhouse rebuild after the batch parent prepared it.",
    )
    parser.add_argument("--in-env-scoring", action="store_true")
    return parser


def _build_batch_parser() -> argparse.ArgumentParser:
    """The single-run options plus dataset selection and concurrency."""

    parser = _build_parser()
    parser.description = (
        "Run one or many SWE-bench-family instances; prepare shared assets once."
    )
    docker_cli.enable_batch(parser, instance_nargs="?")
    parser.set_defaults(
        dataset_name=None,
        max_turns=None,
        run_id=None,
        run_root=None,
        wheelhouse=None,
        uv_binary=None,
        agent_flavor=None,
        force=True,
    )
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="verified")
    parser.add_argument("--all", action="store_true", help="Run the full test split.")
    parser.add_argument(
        "--ids-file", help="Run instance ids listed in this file, in file order."
    )
    parser.add_argument("--parallel", type=int, default=1)
    return parser


def _resolve_paths(
    args: argparse.Namespace, instance: dict
) -> tuple[Path, Path | None]:
    """Pick the run root + wheelhouse for the SWE-bench family."""

    pro = harness.is_swebench_pro_instance(instance, dataset_name=args.dataset_name)
    multilingual = harness.is_swebench_multilingual(dataset_name=args.dataset_name)
    default_run_root = (
        harness.DEFAULT_PRO_RUN_ROOT
        if pro
        else harness.DEFAULT_MULTILINGUAL_RUN_ROOT
        if multilingual
        else harness.DEFAULT_RUN_ROOT
    )
    default_wheelhouse = (
        harness.DEFAULT_PRO_WHEELHOUSE
        if pro
        else harness.DEFAULT_MULTILINGUAL_WHEELHOUSE
        if multilingual
        else harness.DEFAULT_WHEELHOUSE
    )
    run_root = Path(args.run_root) if args.run_root else default_run_root
    wheelhouse = (
        Path(args.wheelhouse).resolve() if args.wheelhouse else default_wheelhouse
    )
    return run_root, wheelhouse


def _provider_environment(args: argparse.Namespace) -> dict[str, str]:
    if args.provider == "openai":
        harness.load_dotenv(args.dotenv)
    provider_env = harness._container_environment(args.provider)
    provider_env[harness.API_KIND_ENV] = harness.resolve_api_kind(args.api_kind)
    provider_env[AGENT_FLAVOR_ENV] = args.agent_flavor
    return provider_env


def _instance_run_kwargs(
    args: argparse.Namespace,
    instance: Mapping[str, Any],
    provider_env: dict[str, str],
) -> dict[str, Any]:
    """Resolve workflow budgets without mutating the process environment."""

    is_arm = args.agent_flavor in WORKFLOW_AGENT_FLAVORS
    outer_max_turns = args.max_turns
    env = dict(provider_env)
    if is_arm:
        env[config.WORKER_MAX_TURNS.name] = str(args.max_turns)
        env[config.REPO_LANGUAGE.name] = instance_language(dict(instance))
        for value, env_name in (
            (args.pdr_rounds, config.PDR_ROUNDS.name),
            (args.pdr_width, config.PDR_WIDTH.name),
            (args.loop_max_turns, config.LOOP_MAX_TURNS.name),
        ):
            if value is not None:
                env[env_name] = str(value)
        outer_max_turns = 1
    return {"provider_env": env, "max_turns": outer_max_turns}


def _suite(args: argparse.Namespace) -> SwebenchSuite:
    return SwebenchSuite(
        dataset_name=args.dataset_name,
        namespace=args.namespace,
        platform=args.platform,
        network_mode=args.network_mode,
        security_opt=docker_cli.security_options(args.security_opt),
        in_env_scoring=args.in_env_scoring,
    )


def run(args: argparse.Namespace) -> dict:
    instance_json = args.instance_json or str(
        ROOT / f"evals/out/swebench/instance_{args.instance_id}.jsonl"
    )
    instance = harness.load_instance(instance_json, args.instance_id)
    provider_env = _provider_environment(args)
    instance_kwargs = _instance_run_kwargs(args, instance, provider_env)
    is_arm = args.agent_flavor in WORKFLOW_AGENT_FLAVORS

    run_root, wheelhouse = _resolve_paths(args, instance)
    clear_run_outputs(
        prepare_run_directory(
            run_root=run_root,
            run_id=args.run_id,
            instance_id=args.instance_id,
        )
    )
    package_extras: tuple[str, ...] = ()
    harness.prepare_wheelhouse_for_run(
        wheelhouse,
        prepare_all=args.prepare_wheelhouse,
        reuse_prepared=args.reuse_prepared_wheelhouse,
        extras=package_extras,
    )

    suite = _suite(args)
    backend = docker_cli.backend(args, wheelhouse)

    name = container_name(suite.name, args.instance_id, args.run_id)

    print(
        f"==> SWE-bench {args.instance_id} [{args.agent_flavor}"
        f"{' arm' if is_arm else ''}] run={args.run_id} container={name}"
    )
    docker_cli.warn_if_unconfined(suite.security_opt)
    result = run_suite_instance(
        suite=suite,
        instance=instance,
        backend=backend,
        store=LocalDirStore(run_root),
        run_root=run_root,
        run_id=args.run_id,
        provider=args.provider,
        api_kind=provider_env[harness.API_KIND_ENV],
        **instance_kwargs,
        package_extras=package_extras,
        wheelhouse_mount=harness.DEFAULT_WHEELHOUSE_MOUNT,
        name=name,
    )

    if result.logs:
        print(result.logs, end="" if result.logs.endswith("\n") else "\n")
    print(f"==> status={result.status_code} run_dir={result.run_dir}")
    return docker_cli.result_record(NAME, result)


def _dataset_rows(dataset_name: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Fetching SWE-bench records requires the 'swebench' extra: "
            "uv sync --extra swebench"
        ) from exc
    return [dict(row) for row in load_dataset(dataset_name, split="test")]


def _ids_from_file(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"--ids-file does not exist: {path}")
    ids: list[str] = []
    seen: set[str] = set()
    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        instance_id = raw_line.split("#", 1)[0].strip()
        if not instance_id:
            continue
        if instance_id in seen:
            raise SystemExit(
                f"Duplicate instance id in {path} on line {lineno}: {instance_id}"
            )
        seen.add(instance_id)
        ids.append(instance_id)
    if not ids:
        raise SystemExit(f"--ids-file {path} did not contain any instance ids")
    return ids


def _select_instances(
    args: argparse.Namespace, config: Variant
) -> list[dict[str, Any]]:
    selectors = int(args.all) + int(bool(args.ids_file)) + int(bool(args.instance_id))
    if selectors > 1:
        raise SystemExit("Pass only one of --all, --ids-file, or one INSTANCE_ID.")
    if args.instance_json:
        if args.all or args.ids_file:
            raise SystemExit("--instance-json only applies to one INSTANCE_ID")
        return [harness.load_instance(args.instance_json, args.instance_id)]

    rows = _dataset_rows(args.dataset_name)
    if not rows:
        raise SystemExit(f"{args.dataset_name} test returned no instances")
    if args.all:
        return rows
    ids = (
        _ids_from_file(Path(args.ids_file))
        if args.ids_file
        else [
            args.instance_id
            or config.default_instance_id
            or os.environ.get("SWE_BENCH_MULTILINGUAL_DEFAULT_INSTANCE_ID")
            or str(rows[0]["instance_id"])
        ]
    )
    by_id = {str(row["instance_id"]): row for row in rows}
    missing = [instance_id for instance_id in ids if instance_id not in by_id]
    if missing:
        raise SystemExit(
            f"Instance id(s) not found in {args.dataset_name}: {', '.join(missing[:10])}"
        )
    return [by_id[instance_id] for instance_id in ids]


def run_batch(args: argparse.Namespace) -> dict:
    config = VARIANTS[args.variant]
    args.dataset_name = args.dataset_name or config.dataset
    args.run_id = canonical_run_id(
        args.run_id or f"{args.variant}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    args.run_root = args.run_root or str(config.run_root)
    args.wheelhouse = args.wheelhouse or str(config.wheelhouse)
    args.agent_flavor = (
        args.agent_flavor
        or os.environ.get(AGENT_FLAVOR_ENV)
        or harness.DEFAULT_AGENT_FLAVOR
    )
    env_turns = os.environ.get("SWEBENCH_MAX_TURNS")
    try:
        if args.max_turns is None:
            args.max_turns = int(env_turns) if env_turns else config.max_turns
    except ValueError as exc:
        raise SystemExit("SWEBENCH_MAX_TURNS must be a positive integer") from exc
    if args.parallel < 1:
        raise SystemExit("--parallel must be a positive integer")
    if args.max_turns < 1:
        raise SystemExit("SWEBENCH_MAX_TURNS/--max-turns must be positive")

    instances = _select_instances(args, config)
    provider_env = _provider_environment(args)
    secondary_token = (
        os.environ.get("OPENAI_AUTH_TOKEN2", "") if args.provider == "openai" else ""
    )
    run_root = Path(args.run_root)
    wheelhouse = Path(args.wheelhouse).resolve()
    args.uv_binary = args.uv_binary or str(harness.ensure_linux_uv())
    print("Preparing wheelhouse and Linux uv once before launching batch...")
    harness.prepare_wheelhouse(wheelhouse)

    suite = _suite(args)
    backend = docker_cli.backend(args, wheelhouse)
    instance_ids = [str(instance["instance_id"]) for instance in instances]
    index_by_id = {instance_id: index for index, instance_id in enumerate(instance_ids)}
    expected_ids = run_root / args.run_id / "expected_instance_ids.txt"
    expected_ids.parent.mkdir(parents=True, exist_ok=True)
    expected_ids.write_text(
        "".join(f"{instance_id}\n" for instance_id in instance_ids),
        encoding="utf-8",
    )

    def per_instance(instance: Mapping[str, Any]) -> dict[str, Any]:
        instance_id = str(instance["instance_id"])
        env = dict(provider_env)
        if secondary_token and index_by_id[instance_id] % 2:
            env[harness.OPENAI_AUTH_ENV] = secondary_token
        return {
            **_instance_run_kwargs(args, instance, env),
            "name": container_name(suite.name, instance_id, args.run_id),
        }

    print(f"=== SWE-bench {args.variant.title()} batch ===")
    print(f"Run ID: {args.run_id}")
    print(f"Instances: {len(instances)}")
    print(f"Parallel: {args.parallel}")
    print(f"OpenAI auth tokens: {2 if secondary_token else 1}")
    report, failed = run_container_batch(
        suite=suite,
        instances=instances,
        backend=backend,
        store=LocalDirStore(run_root),
        run_root=run_root,
        run_id=args.run_id,
        parallel=args.parallel,
        per_instance_kwargs=per_instance,
        provider=args.provider,
        api_kind=provider_env[harness.API_KIND_ENV],
        max_turns=args.max_turns,
        provider_env=provider_env,
        package_extras=(),
        wheelhouse_mount=harness.DEFAULT_WHEELHOUSE_MOUNT,
    )

    from evals.swebench.evaluate_predictions import predictions_from_run_dirs

    predictions_path = run_root / f"{args.run_id}_predictions.jsonl"
    write_jsonl(
        predictions_path,
        predictions_from_run_dirs(
            run_root,
            run_id=args.run_id,
            model_name=f"simple-agent-lab-{args.variant}",
            dataset_name=args.dataset_name,
            expected_instance_ids=instance_ids,
        ),
    )
    print(f"Outputs: {run_root / args.run_id}/")
    if failed:
        print(f"Failed runs: {failed}", file=sys.stderr)
    return {
        "bench": NAME,
        "status_code": int(bool(failed)),
        "run_dir": str(run_root / args.run_id),
        "result_path": str(predictions_path),
        "summary": {**report.summary(), "nonzero": failed},
    }
