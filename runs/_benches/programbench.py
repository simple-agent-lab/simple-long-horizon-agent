"""Run one or many ProgramBench tasks through the generic Suite API.

The container can reach the model, while agent bash commands run in an isolated
network namespace unless ``--no-network-isolation`` is explicit.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from evals.programbench import harness
from evals.programbench.suite import ProgrambenchSuite
from evals.swebench.harness import ensure_linux_uv
from runs.lib.container_batch import run_container_batch
from runs.lib import docker_cli
from simple_long_horizon_agent.evals import (
    LocalDirStore,
    run_suite_instance,
)
from simple_long_horizon_agent.evals.suites.programbench import container
from simple_long_horizon_agent.evals.backends.docker_local import (
    DEFAULT_DOCKER_TIMEOUT_S,
)
from simple_long_horizon_agent.evals.runner import canonical_run_id, container_name

ROOT = Path(__file__).resolve().parents[2]
NAME = "programbench"
DESCRIPTION = (
    "ProgramBench reverse-engineering instance in a Docker container "
    "(single instance per run; per-command network isolation)."
)
SCORER = ("-m", "evals.programbench.evaluate_submissions")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance_id")
    parser.add_argument("--profile", help="JSON run-profile path.")
    parser.add_argument("--max-turns", type=int, default=1000)
    parser.add_argument(
        "--wall-time-seconds",
        type=float,
        default=21600,
        help="Agent wall-clock limit (default: 6h).",
    )
    parser.add_argument(
        "--run-id", default=f"programbench-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    parser.add_argument("--provider", choices=["fake", "openai"], default="openai")
    parser.add_argument("--api-kind")
    parser.add_argument(
        "--image-tag",
        default=harness.DEFAULT_IMAGE_TAG,
        help="Inference image tag.",
    )
    docker_cli.add_arguments(
        parser,
        default_uv_binary=harness.DEFAULT_UV_BINARY,
        default_timeout_seconds=DEFAULT_DOCKER_TIMEOUT_S,
    )
    parser.add_argument("--dotenv", default=str(ROOT / ".env"))
    parser.add_argument(
        "--cpus",
        type=int,
        default=20,
        help="Docker CPU limit.",
    )
    parser.add_argument(
        "--mem-limit",
        default="60g",
        help="Docker memory and memory-swap limit.",
    )
    parser.add_argument(
        "--no-network-isolation",
        action="store_true",
        help="Allow agent bash commands network access.",
    )
    return parser


def _build_batch_parser() -> argparse.ArgumentParser:
    parser = _build_parser()
    parser.description = (
        "Run one or many ProgramBench instances; prepare shared assets once."
    )
    docker_cli.enable_batch(parser, instance_nargs="*")
    parser.set_defaults(run_id=None, uv_binary=None, force=True)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--filter", default="", help="Regex on instance_id.")
    parser.add_argument("--slice", default="", help="Task slice, e.g. 0:5.")
    return parser


def _provider_environment(args: argparse.Namespace) -> dict[str, str]:
    if args.provider == "openai":
        harness.load_dotenv(args.dotenv)
    provider_env = harness.container_environment(args.provider)
    provider_env[harness.API_KIND_ENV] = harness.resolve_api_kind(args.api_kind)
    provider_env[container.REQUIRE_ISOLATION_ENV] = (
        "0" if args.no_network_isolation else "1"
    )
    return provider_env


def _suite(args: argparse.Namespace) -> ProgrambenchSuite:
    return ProgrambenchSuite(
        image_tag=args.image_tag,
        platform=args.platform,
        network_mode=args.network_mode,
        cap_add=() if args.no_network_isolation else ("SYS_ADMIN",),
        security_opt=docker_cli.security_options(args.security_opt),
        cpus=args.cpus,
        mem_limit=args.mem_limit or None,
    )


def run(args: argparse.Namespace) -> dict:
    instance = harness.load_instance(args.instance_id)

    provider_env = _provider_environment(args)

    run_root = Path(args.run_root) if args.run_root else harness.DEFAULT_RUN_ROOT
    wheelhouse = (
        Path(args.wheelhouse).resolve()
        if args.wheelhouse
        else harness.DEFAULT_WHEELHOUSE
    )
    harness.prepare_wheelhouse_for_run(wheelhouse, prepare_all=args.prepare_wheelhouse)

    suite = _suite(args)
    backend = docker_cli.backend(args, wheelhouse)

    name = container_name(suite.name, args.instance_id, args.run_id)

    print(
        f"==> ProgramBench {args.instance_id} run={args.run_id} container={name} "
        f"isolation={'off' if args.no_network_isolation else 'on'}"
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
        max_turns=args.max_turns,
        wall_time_seconds=args.wall_time_seconds,
        provider_env=provider_env,
        wheelhouse_mount=harness.DEFAULT_WHEELHOUSE_MOUNT,
        name=name,
    )

    if result.logs:
        print(result.logs, end="" if result.logs.endswith("\n") else "\n")
    print(f"==> status={result.status_code} run_dir={result.run_dir}")
    print(
        "    score it: uv run python -m evals.programbench.evaluate_submissions "
        f"--run-root {run_root} --run-id {args.run_id}"
    )
    return docker_cli.result_record(NAME, result)


def _batch_instances(args: argparse.Namespace) -> list[dict[str, Any]]:
    instance_ids = list(args.instance_id)
    if (args.all or args.filter or args.slice) and instance_ids:
        raise SystemExit(
            "Pass positional instance ids or --all/--filter/--slice, not both."
        )
    if args.all or args.filter or args.slice:
        instances = harness.load_instances(
            filter_spec=args.filter,
            slice_spec=args.slice,
        )
    elif instance_ids:
        instances = [harness.load_instance(instance_id) for instance_id in instance_ids]
    else:
        instances = [harness.load_instance("abishekvashok__cmatrix.5c082c6")]
    if not instances:
        raise SystemExit("No instances selected.")
    return instances


def run_batch(args: argparse.Namespace) -> dict:
    if args.parallel < 1:
        raise SystemExit("--parallel must be a positive integer")
    args.run_id = canonical_run_id(
        args.run_id or f"programbench-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    run_root = Path(args.run_root) if args.run_root else harness.DEFAULT_RUN_ROOT
    wheelhouse = (
        Path(args.wheelhouse).resolve()
        if args.wheelhouse
        else harness.DEFAULT_WHEELHOUSE
    )
    instances = _batch_instances(args)
    provider_env = _provider_environment(args)
    args.uv_binary = args.uv_binary or str(ensure_linux_uv())
    print("Preparing wheelhouse and Linux uv once before launching batch...")
    harness.prepare_wheelhouse(wheelhouse)

    suite = _suite(args)
    backend = docker_cli.backend(args, wheelhouse)

    def per_instance(instance: Mapping[str, Any]) -> dict[str, Any]:
        instance_id = str(instance["instance_id"])
        return {"name": container_name(suite.name, instance_id, args.run_id)}

    print("=== ProgramBench batch ===")
    print(f"Run ID: {args.run_id}")
    print(f"Instances: {len(instances)}")
    print(f"Parallel: {args.parallel}")
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
        wall_time_seconds=args.wall_time_seconds,
        provider_env=provider_env,
        wheelhouse_mount=harness.DEFAULT_WHEELHOUSE_MOUNT,
    )
    print(f"Outputs: {run_root / args.run_id}/")
    print(
        "Score with: uv run --extra programbench python "
        "-m evals.programbench.evaluate_submissions "
        f"--run-id {args.run_id} --workers {args.parallel}"
    )
    if failed:
        print(f"Failed runs: {failed}", file=sys.stderr)
    return {
        "bench": NAME,
        "status_code": int(bool(failed)),
        "run_dir": str(run_root / args.run_id),
        "result_path": None,
        "summary": {**report.summary(), "nonzero": failed},
    }
