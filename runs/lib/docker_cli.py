"""Shared Docker options for the benchmark run entries."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from simple_long_horizon_agent.evals import LocalDockerBackend, RunArtifacts

DEFAULT_SECURITY_OPTIONS = ("seccomp=unconfined",)
_UNCONFINED_WARNING = (
    "    WARNING: seccomp disabled (seccomp=unconfined) — reduced container "
    "isolation. Pass --security-opt seccomp=default to restore the daemon's profile."
)


def add_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_uv_binary: str,
    default_timeout_seconds: float,
) -> None:
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
    for option in ("run-root", "wheelhouse"):
        parser.add_argument(f"--{option}")
    parser.add_argument("--uv-binary", default=default_uv_binary)
    parser.add_argument(
        "--docker-timeout-seconds",
        type=float,
        default=default_timeout_seconds,
        help="Docker SDK timeout.",
    )
    for option in ("prepare-wheelhouse", "keep-container", "force"):
        parser.add_argument(f"--{option}", action="store_true")


def enable_batch(parser: argparse.ArgumentParser, *, instance_nargs: str) -> None:
    for action in parser._actions:
        if action.dest == "instance_id":
            action.nargs = instance_nargs
        elif action.dest == "pull":
            action.nargs, action.const = "?", "missing"


def backend(args: argparse.Namespace, wheelhouse: Path | None) -> LocalDockerBackend:
    return LocalDockerBackend(
        pull=args.pull,
        keep_container=args.keep_container,
        force_existing=args.force,
        wheelhouse=wheelhouse,
        uv_binary=args.uv_binary or None,
        docker_timeout_s=args.docker_timeout_seconds,
    )


def security_options(values: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(values) if values is not None else DEFAULT_SECURITY_OPTIONS


def warn_if_unconfined(options: Sequence[str]) -> None:
    if any("seccomp=unconfined" in option for option in options):
        print(_UNCONFINED_WARNING)


def result_record(bench: str, result: RunArtifacts) -> dict[str, object]:
    return {
        "bench": bench,
        "status_code": result.status_code,
        "run_dir": str(result.run_dir),
        "result_path": str(result.run_dir / "out" / "result.json"),
        "summary": None,
    }
