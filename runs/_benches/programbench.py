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

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evals.programbench import harness  # noqa: E402
from evals.programbench.suite import ProgrambenchSuite  # noqa: E402
from evals.swebench.harness import ensure_linux_uv  # noqa: E402
from runs.lib.container_batch import run_container_batch  # noqa: E402
from simple_agent_lab.evals import (  # noqa: E402
    LocalDirStore,
    LocalDockerBackend,
    run_suite_instance,
)
from simple_agent_lab.evals.suites.programbench import container  # noqa: E402
from simple_agent_lab.evals.backends.docker_local import (  # noqa: E402
    DEFAULT_DOCKER_TIMEOUT_S,
)
from simple_agent_lab.evals.runner import canonical_run_id, container_name  # noqa: E402

NAME = "programbench"
DESCRIPTION = (
    "ProgramBench reverse-engineering instance in a Docker container "
    "(single instance per run; per-command network isolation)."
)
SCORER = ("evals/programbench/evaluate_submissions.py",)


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
    parser.add_argument("--network-mode", default="host")
    parser.add_argument(
        "--security-opt",
        action="append",
        help="Repeatable Docker security option (default: seccomp=unconfined).",
    )
    parser.add_argument(
        "--platform", default="", help="Override docker --platform (e.g. linux/amd64)"
    )
    parser.add_argument(
        "--pull",
        choices=["missing", "always", "never"],
        default="never",
        help="Image pull policy (default: never).",
    )
    parser.add_argument("--dotenv", default=str(ROOT / ".env"))
    parser.add_argument("--run-root")
    parser.add_argument("--wheelhouse")
    parser.add_argument("--uv-binary", default=harness.DEFAULT_UV_BINARY)
    parser.add_argument(
        "--docker-timeout-seconds",
        type=float,
        default=DEFAULT_DOCKER_TIMEOUT_S,
        help="Docker SDK timeout.",
    )
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
    parser.add_argument("--prepare-wheelhouse", action="store_true")
    parser.add_argument("--keep-container", action="store_true")
    parser.add_argument(
        "--no-network-isolation",
        action="store_true",
        help="Allow agent bash commands network access.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def _build_batch_parser() -> argparse.ArgumentParser:
    parser = _build_parser()
    parser.description = (
        "Run one or many ProgramBench instances; prepare shared assets once."
    )
    for action in parser._actions:
        if action.dest == "instance_id":
            action.nargs = "*"
        elif action.dest == "pull":
            action.nargs = "?"
            action.const = "missing"
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
    security_opt = (
        tuple(args.security_opt)
        if args.security_opt is not None
        else ("seccomp=unconfined",)
    )
    return ProgrambenchSuite(
        image_tag=args.image_tag,
        platform=args.platform,
        network_mode=args.network_mode,
        cap_add=() if args.no_network_isolation else ("SYS_ADMIN",),
        security_opt=security_opt,
        cpus=args.cpus,
        mem_limit=args.mem_limit or None,
    )


def _backend(args: argparse.Namespace, wheelhouse: Path | None) -> LocalDockerBackend:
    return LocalDockerBackend(
        pull=args.pull,
        keep_container=args.keep_container,
        force_existing=args.force,
        wheelhouse=wheelhouse,
        uv_binary=args.uv_binary or None,
        docker_timeout_s=args.docker_timeout_seconds,
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
    backend = _backend(args, wheelhouse)

    name = container_name(suite.name, args.instance_id, args.run_id)

    print(
        f"==> ProgramBench {args.instance_id} run={args.run_id} container={name} "
        f"isolation={'off' if args.no_network_isolation else 'on'}"
    )
    if any("seccomp=unconfined" in opt for opt in suite.security_opt):
        print(
            "    WARNING: seccomp disabled (seccomp=unconfined) — reduced "
            "container isolation. Pass --security-opt seccomp=default to "
            "restore the daemon's profile."
        )
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
        "    score it: uv run python evals/programbench/evaluate_submissions.py "
        f"--run-root {run_root} --run-id {args.run_id}"
    )
    return {
        "bench": NAME,
        "status_code": result.status_code,
        "run_dir": str(result.run_dir),
        "result_path": str(result.run_dir / "out" / "result.json"),
        "summary": None,
    }


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
    backend = _backend(args, wheelhouse)

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
        "evals/programbench/evaluate_submissions.py "
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
