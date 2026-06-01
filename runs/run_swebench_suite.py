"""Run one SWE-bench instance through the generic `Suite` framework (ADR 0017).

This is the SWE-bench run entry point: it drives the SWE-bench container half
through `run_suite_instance(SwebenchSuite, LocalDockerBackend, LocalDirStore)`,
the same primitive every suite uses. The launch shape (image, workdir, shell,
cap_add) and the wheelhouse/uv mounts come from the shared `harness` helpers that
`SwebenchSuite` delegates to.

Usage (host with Docker + a built SWE-bench image):

    uv run python runs/run_swebench_suite.py <instance-id> \
        [--max-turns N] [--run-id ID] [--agent-flavor bash|bash_task] \
        [--in-env-scoring] [--force]

Reads OPENAI_MODEL / OPENAI_AUTH_TOKEN (and optional OPENAI_BASE_URL) from .env.
For batch / parallel runs over a whole split, see runs/run_swebench_verified.sh
and runs/run_swebench_pro.sh.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evals.swebench import harness  # noqa: E402
from evals.swebench.suite import SwebenchSuite  # noqa: E402
from simple_agent_lab.evals import (  # noqa: E402
    LocalDirStore,
    LocalDockerBackend,
    run_suite_instance,
)
from simple_agent_lab.evals.runner import container_name  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance_id")
    parser.add_argument(
        "--instance-json",
        default=None,
        help="Defaults to evals/out/swebench/instance_<id>.jsonl",
    )
    parser.add_argument("--dataset-name", default=harness.DEFAULT_DATASET)
    parser.add_argument("--max-turns", type=int, default=75)
    parser.add_argument(
        "--run-id", default=f"swebench-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    parser.add_argument(
        "--agent-flavor",
        choices=harness.AGENT_FLAVOR_CHOICES,
        default=harness.DEFAULT_AGENT_FLAVOR,
    )
    parser.add_argument(
        "--provider", choices=["fake", "openai", "oracle"], default="openai"
    )
    parser.add_argument("--api-kind", default=None)
    parser.add_argument("--namespace", default="swebench")
    parser.add_argument("--network-mode", default="host")
    parser.add_argument(
        "--platform", default="", help="Override docker --platform (e.g. linux/amd64)"
    )
    parser.add_argument(
        "--pull",
        choices=["missing", "always", "never"],
        default="missing",
        help="Image pull policy before create.",
    )
    parser.add_argument("--dotenv", default=str(ROOT / ".env"))
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--wheelhouse", default=None)
    parser.add_argument("--uv-binary", default=harness.DEFAULT_UV_BINARY)
    parser.add_argument("--prepare-wheelhouse", action="store_true")
    parser.add_argument("--in-env-scoring", action="store_true")
    parser.add_argument("--keep-container", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove an existing container with the same name before starting.",
    )
    return parser


def _resolve_paths(args: argparse.Namespace, instance: dict) -> tuple[Path, Path | None]:
    """Pick the run root + wheelhouse, swapping to Pro defaults for Pro instances."""

    pro = harness.is_swebench_pro_instance(instance, dataset_name=args.dataset_name)
    run_root = (
        Path(args.run_root)
        if args.run_root
        else (harness.DEFAULT_PRO_RUN_ROOT if pro else harness.DEFAULT_RUN_ROOT)
    )
    wheelhouse_arg = args.wheelhouse or str(
        harness.DEFAULT_PRO_WHEELHOUSE if pro else harness.DEFAULT_WHEELHOUSE
    )
    wheelhouse = Path(wheelhouse_arg).resolve() if wheelhouse_arg else None
    return run_root, wheelhouse


def main() -> None:
    args = _build_parser().parse_args()

    instance_json = args.instance_json or str(
        ROOT / f"evals/out/swebench/instance_{args.instance_id}.jsonl"
    )
    instance = harness.load_instance(instance_json, args.instance_id)

    # Provider credentials + flavor flow in as the container's environment
    # (_container_environment validates the required OPENAI_* vars and exits with
    # a clear message if absent).
    if args.provider == "openai":
        harness.load_dotenv(args.dotenv)
    provider_env = harness._container_environment(args.provider)
    provider_env[harness.API_KIND_ENV] = harness.resolve_api_kind(args.api_kind)
    provider_env["AGENT_FLAVOR"] = args.agent_flavor

    run_root, wheelhouse = _resolve_paths(args, instance)
    harness.prepare_wheelhouse_for_run(wheelhouse, prepare_all=args.prepare_wheelhouse)

    suite = SwebenchSuite(
        dataset_name=args.dataset_name,
        namespace=args.namespace,
        platform=args.platform,
        network_mode=args.network_mode,
        in_env_scoring=args.in_env_scoring,
    )
    backend = LocalDockerBackend(
        pull=args.pull,
        keep_container=args.keep_container,
        wheelhouse=wheelhouse,
        uv_binary=args.uv_binary or None,
    )

    name = container_name(suite.name, args.instance_id, args.run_id)
    if args.force:
        _force_remove(name)

    print("==> Running SWE-bench instance through SwebenchSuite")
    print(f"    instance:   {args.instance_id}")
    print(f"    max-turns:  {args.max_turns}")
    print(f"    run-id:     {args.run_id}")
    print(f"    agent:      {args.agent_flavor}")
    print(f"    container:  {name}")
    print("")

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
        provider_env=provider_env,
        wheelhouse_mount=harness.DEFAULT_WHEELHOUSE_MOUNT,
        name=name,
    )

    if result.logs:
        print(result.logs, end="" if result.logs.endswith("\n") else "\n")
    print("")
    print(f"==> run dir: {result.run_dir}")
    print(f"    result:  {result.run_dir / 'out' / 'result.json'}")
    print(f"    status:  {result.status_code}")
    if result.status_code != 0:
        raise SystemExit(result.status_code)


def _force_remove(name: str) -> None:
    """Drop a leftover container with the deterministic run name (legacy --force)."""

    import docker

    client = docker.from_env()
    existing = harness._get_container(client, name)
    if existing is not None:
        existing.remove(force=True)


if __name__ == "__main__":
    main()
