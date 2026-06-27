"""Run one SWE-bench instance through the generic `Suite` framework (ADR generic-containerized-eval-framework).

This is the SWE-bench run entry point: it drives the SWE-bench container half
through `run_suite_instance(SwebenchSuite, LocalDockerBackend, LocalDirStore)`,
the same primitive every suite uses. The launch shape (image, workdir, shell,
cap_add) and the wheelhouse/uv mounts come from the shared `harness` helpers that
`SwebenchSuite` delegates to.

Usage (host with Docker + a built SWE-bench image):

    uv run python runs/run_bench.py swebench <instance-id> \
        [--max-turns N] [--run-id ID] \
        [--agent-flavor bash|bash_task|bash_task_read|bash_skills|loop|pdr] \
        [--in-env-scoring] [--force]

Reads OPENAI_MODEL / OPENAI_AUTH_TOKEN (and optional OPENAI_BASE_URL) from .env.
For batch / parallel runs over a whole split, see runs/run_swebench_verified.sh,
runs/run_swebench_multilingual.sh, and runs/run_swebench_pro.sh.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evals.swebench import harness  # noqa: E402
from evals.swebench.suite import SwebenchSuite  # noqa: E402
from simple_agent_lab.agent_flavors import (  # noqa: E402
    AGENT_FLAVOR_ENV,
    WORKFLOW_AGENT_FLAVORS,
)
import simple_agent_lab.config as config  # noqa: E402
from simple_agent_lab.evals import (  # noqa: E402
    LocalDirStore,
    LocalDockerBackend,
    parse_with_profile,
    run_suite_instance,
)
from simple_agent_lab.evals.backends.docker_local import (  # noqa: E402
    DEFAULT_DOCKER_TIMEOUT_S,
)
from simple_agent_lab.evals.runner import container_name  # noqa: E402
from simple_agent_lab.evals.suites.swebench.patch import instance_language  # noqa: E402

# Identity for the unified entry (runs/run_bench.py). `run(args)` returns a
# result dict so the dispatcher / dashboard can read a machine-readable outcome.
NAME = "swebench"
DESCRIPTION = "SWE-bench instance in a Docker container (single instance per run)."


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance_id")
    parser.add_argument(
        "--profile",
        default=None,
        help=(
            "Path to a JSON run-profile (its `env` fills env gaps, its `run` "
            "flags are defaults overridable by explicit flags). See ADR "
            "run-profile-file."
        ),
    )
    parser.add_argument(
        "--instance-json",
        default=None,
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
        help=(
            "The single agent selector. Simple flavors (bash | bash_task | "
            "bash_task_read | bash_skills) run one multi-turn agent; workflow "
            "arms (loop | pdr) run a multi-agent choreography — when an arm is "
            "chosen, --max-turns becomes the per-agent budget and the outer "
            "facade loop runs once."
        ),
    )
    parser.add_argument("--pdr-rounds", type=int, default=None)
    parser.add_argument("--pdr-width", type=int, default=None)
    parser.add_argument("--loop-max-turns", type=int, default=None)
    parser.add_argument(
        "--provider", choices=["fake", "openai", "oracle"], default="openai"
    )
    parser.add_argument("--api-kind", default=None)
    parser.add_argument("--namespace", default="swebench")
    parser.add_argument("--network-mode", default="host")
    parser.add_argument(
        "--security-opt",
        action="append",
        default=None,
        help="Docker --security-opt (repeatable). Defaults to seccomp=unconfined; "
        "pass --security-opt seccomp=default to restore the daemon's profile.",
    )
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
        help=(
            "Docker SDK HTTP timeout in seconds for daemon calls such as "
            "pull/create/start/wait."
        ),
    )
    parser.add_argument("--prepare-wheelhouse", action="store_true")
    parser.add_argument("--in-env-scoring", action="store_true")
    parser.add_argument("--keep-container", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove an existing container with the same name before starting.",
    )
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
    wheelhouse_arg = args.wheelhouse or str(default_wheelhouse)
    wheelhouse = Path(wheelhouse_arg).resolve() if wheelhouse_arg else None
    return run_root, wheelhouse


def run(args: argparse.Namespace) -> dict:
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
    provider_env[AGENT_FLAVOR_ENV] = args.agent_flavor

    # The single AGENT_FLAVOR selector picks the agent. For a workflow arm
    # (loop | pdr) the facade `build_agent` runs the whole choreography in ONE
    # outer turn, so --max-turns becomes the per-agent budget (passed as
    # SAL_WORKFLOW_WORKER_MAX_TURNS) and the outer loop runs once. Simple flavors run the
    # normal multi-turn agent with --max-turns as their own budget.
    is_arm = args.agent_flavor in WORKFLOW_AGENT_FLAVORS
    outer_max_turns = args.max_turns
    if is_arm:
        provider_env[config.WORKER_MAX_TURNS.name] = str(args.max_turns)
        provider_env[config.REPO_LANGUAGE.name] = instance_language(dict(instance))
        for value, env_name in (
            (args.pdr_rounds, config.PDR_ROUNDS.name),
            (args.pdr_width, config.PDR_WIDTH.name),
            (args.loop_max_turns, config.LOOP_MAX_TURNS.name),
        ):
            if value is not None:
                provider_env[env_name] = str(value)
        outer_max_turns = 1

    run_root, wheelhouse = _resolve_paths(args, instance)
    package_extras: tuple[str, ...] = ()
    harness.prepare_wheelhouse_for_run(
        wheelhouse,
        prepare_all=args.prepare_wheelhouse,
        extras=package_extras,
    )

    security_opt = (
        tuple(args.security_opt)
        if args.security_opt is not None
        else ("seccomp=unconfined",)
    )
    suite = SwebenchSuite(
        dataset_name=args.dataset_name,
        namespace=args.namespace,
        platform=args.platform,
        network_mode=args.network_mode,
        security_opt=security_opt,
        in_env_scoring=args.in_env_scoring,
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

    print(f"==> Running SWE-bench instance through {SwebenchSuite.__name__}")
    print(f"    instance:   {args.instance_id}")
    print(f"    max-turns:  {args.max_turns}")
    print(f"    run-id:     {args.run_id}")
    print(f"    agent:      {args.agent_flavor}{' (arm)' if is_arm else ''}")
    print(f"    container:  {name}")
    print(f"    docker api timeout: {args.docker_timeout_seconds:g}s")
    if any("seccomp=unconfined" in opt for opt in security_opt):
        print(
            "    WARNING: seccomp disabled (seccomp=unconfined) — reduced "
            "container isolation. Pass --security-opt seccomp=default to "
            "restore the daemon's profile."
        )
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
        max_turns=outer_max_turns,
        provider_env=provider_env,
        package_extras=package_extras,
        wheelhouse_mount=harness.DEFAULT_WHEELHOUSE_MOUNT,
        name=name,
    )

    if result.logs:
        print(result.logs, end="" if result.logs.endswith("\n") else "\n")
    print("")
    print(f"==> run dir: {result.run_dir}")
    print(f"    result:  {result.run_dir / 'out' / 'result.json'}")
    print(f"    status:  {result.status_code}")
    return {
        "bench": NAME,
        "status_code": result.status_code,
        "run_dir": str(result.run_dir),
        "result_path": str(result.run_dir / "out" / "result.json"),
        "summary": None,
    }


def main() -> None:
    raise SystemExit(run(parse_with_profile(_build_parser()))["status_code"])


def _force_remove(name: str) -> None:
    """Drop a leftover container with the deterministic run name (legacy --force)."""

    import docker

    client = docker.from_env(timeout=DEFAULT_DOCKER_TIMEOUT_S)
    existing = harness._get_container(client, name)
    if existing is not None:
        existing.remove(force=True)


if __name__ == "__main__":
    main()
