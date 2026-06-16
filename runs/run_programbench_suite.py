"""Run one ProgramBench instance through the generic `Suite` framework.

This is the ProgramBench run entry point: it drives the ProgramBench container
half through `run_suite_instance(ProgrambenchSuite, LocalDockerBackend,
LocalDirStore)`, the same primitive every suite uses. The launch shape (image,
workdir, ``cap_add=("SYS_ADMIN",)`` for per-command network isolation) and the
wheelhouse/uv mounts come from `ProgrambenchSuite` + the shared `harness`
helpers.

Usage (host with Docker + the ProgramBench image pulled):

    uv run python runs/run_programbench_suite.py <instance-id> \
        [--max-turns N] [--run-id ID] [--no-network-isolation] [--force]

Reads OPENAI_MODEL / OPENAI_AUTH_TOKEN (and optional OPENAI_BASE_URL) from .env.
The agent runs *inside* the container with the model API reachable, but each
agent bash command runs in a network-isolated namespace (see
`programbench-reverse-engineering-adapter`). Score the run afterwards with
evals/programbench/evaluate_submissions.py. For batch / parallel runs over the
whole task set, see runs/run_programbench.sh.
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

from evals.programbench import harness  # noqa: E402
from evals.programbench.suite import ProgrambenchSuite  # noqa: E402
from simple_agent_lab.evals import (  # noqa: E402
    LocalDirStore,
    LocalDockerBackend,
    run_suite_instance,
)
from simple_agent_lab.evals.suites.programbench import container  # noqa: E402
from simple_agent_lab.evals.backends.docker_local import (  # noqa: E402
    DEFAULT_DOCKER_TIMEOUT_S,
)
from simple_agent_lab.evals.runner import container_name  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance_id")
    parser.add_argument("--max-turns", type=int, default=1000)
    parser.add_argument(
        "--wall-time-seconds",
        type=float,
        default=21600,
        help="Wall-clock time limit for the agent run in seconds (default: 21600 = 6h).",
    )
    parser.add_argument(
        "--run-id", default=f"programbench-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    parser.add_argument("--provider", choices=["fake", "openai"], default="openai")
    parser.add_argument("--api-kind", default=None)
    parser.add_argument(
        "--image-tag",
        default=harness.DEFAULT_IMAGE_TAG,
        help="Inference image tag (default: task_cleanroom).",
    )
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
    parser.add_argument(
        "--docker-timeout-seconds",
        type=float,
        default=DEFAULT_DOCKER_TIMEOUT_S,
        help="Docker SDK HTTP timeout in seconds for daemon calls.",
    )
    parser.add_argument("--prepare-wheelhouse", action="store_true")
    parser.add_argument("--keep-container", action="store_true")
    parser.add_argument(
        "--no-network-isolation",
        action="store_true",
        help=(
            "Do not add CAP_SYS_ADMIN. Agent bash commands then run WITH network "
            "access (no `unshare --net`), weakening ProgramBench's anti-cheat."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove an existing container with the same name before starting.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    instance = harness.load_instance(args.instance_id)

    if args.provider == "openai":
        harness.load_dotenv(args.dotenv)
    provider_env = harness.container_environment(args.provider)
    provider_env[harness.API_KIND_ENV] = harness.resolve_api_kind(args.api_kind)
    # Fail closed in-container: a missing `unshare --net` aborts the run unless
    # the operator opted out here (so isolation is never lost silently).
    provider_env[container.REQUIRE_ISOLATION_ENV] = (
        "0" if args.no_network_isolation else "1"
    )

    run_root = Path(args.run_root) if args.run_root else harness.DEFAULT_RUN_ROOT
    wheelhouse = (
        Path(args.wheelhouse).resolve()
        if args.wheelhouse
        else harness.DEFAULT_WHEELHOUSE
    )
    harness.prepare_wheelhouse_for_run(wheelhouse, prepare_all=args.prepare_wheelhouse)

    suite = ProgrambenchSuite(
        image_tag=args.image_tag,
        platform=args.platform,
        network_mode=args.network_mode,
        cap_add=() if args.no_network_isolation else ("SYS_ADMIN",),
    )
    backend = LocalDockerBackend(
        pull=args.pull,
        keep_container=args.keep_container,
        wheelhouse=wheelhouse,
        uv_binary=args.uv_binary or None,
        docker_timeout_s=args.docker_timeout_seconds,
    )

    name = container_name(suite.name, args.instance_id, args.run_id)
    if args.force:
        _force_remove(name)

    isolation = "off (CAP_SYS_ADMIN withheld)" if args.no_network_isolation else "on"
    print("==> Running ProgramBench instance through ProgrambenchSuite")
    print(f"    instance:        {args.instance_id}")
    print(f"    max-turns:       {args.max_turns}")
    print(
        f"    wall-time:       {args.wall_time_seconds}s ({args.wall_time_seconds / 3600:.1f}h)"
    )
    print(f"    run-id:          {args.run_id}")
    print(f"    image-tag:       {args.image_tag}")
    print(f"    cmd net-isolate: {isolation}")
    print(f"    container:       {name}")
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
        wall_time_seconds=args.wall_time_seconds,
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
    print(
        "    score it: uv run python evals/programbench/evaluate_submissions.py "
        f"--run-root {run_root} --run-id {args.run_id}"
    )
    if result.status_code != 0:
        raise SystemExit(result.status_code)


def _force_remove(name: str) -> None:
    """Drop a leftover container with the deterministic run name."""

    import docker

    client = docker.from_env(timeout=DEFAULT_DOCKER_TIMEOUT_S)
    for existing in client.containers.list(all=True, filters={"name": name}):
        if existing.name == name:
            existing.remove(force=True)


if __name__ == "__main__":
    main()
