"""Recipe runtime helpers for env loading, Docker probing, and worker counts.

These live in the recipe layer, not the package: they import ``docker`` directly,
which arch_lint confines to ``evals.backends`` inside the package. Recipes are
scaffolding and are not arch-linted.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def load_dotenv(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def check_docker_available(*, client_factory=None) -> None:
    """Fail early when Docker is not reachable for SWE-bench execution."""

    try:
        if client_factory is None:
            import docker

            client_factory = docker.from_env
        client = client_factory()
        ping = getattr(client, "ping", None)
        if callable(ping):
            ping()
    except Exception as exc:
        raise SystemExit(
            "Docker is required for --execute SWE-bench self-evolution, but the "
            "Docker daemon is not reachable. Start Docker Desktop or Colima and "
            "set DOCKER_HOST if needed. On macOS/Colima, try: "
            "colima start --cpu 4 --memory 8 --arch aarch64 --vm-type vz --vz-rosetta "
            "&& export DOCKER_HOST=unix://$HOME/.colima/default/docker.sock. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


def preflight_suite_images(
    suite: Any,
    instances: Sequence[Mapping[str, Any]],
    *,
    client_factory=None,
    pull: str = "missing",
    label: str = "instances",
) -> int:
    """Check SWE-bench images before spending model tokens.

    Returns the number of unique images checked. ``pull`` mirrors the Docker
    backend policy: ``never`` only checks local cache, ``missing`` pulls absent
    images, and ``always`` pulls every image up front.
    """

    if pull not in {"missing", "always", "never"}:
        raise SystemExit(
            f"image preflight pull policy must be missing/always/never, got {pull!r}"
        )
    try:
        if client_factory is None:
            import docker

            client_factory = docker.from_env
        client = client_factory()
    except Exception as exc:
        raise SystemExit(
            "Docker is required for SWE-bench image preflight, but the daemon "
            f"is not reachable. Original error: {type(exc).__name__}: {exc}"
        ) from exc

    image_specs: dict[str, str | None] = {}
    for instance in instances:
        spec = suite.launch_spec(instance)
        image_specs.setdefault(spec.image, spec.platform)

    failures: list[str] = []
    for image, platform in image_specs.items():
        try:
            if pull == "always":
                client.images.pull(image, platform=platform)
            elif pull == "missing":
                try:
                    client.images.get(image)
                except Exception:
                    client.images.pull(image, platform=platform)
            else:
                client.images.get(image)
        except Exception as exc:  # noqa: BLE001 - summarize all image failures.
            failures.append(f"{image}: {type(exc).__name__}: {exc}")

    if failures:
        shown = "\n".join(f"  - {line}" for line in failures[:12])
        more = "" if len(failures) <= 12 else f"\n  ... {len(failures) - 12} more"
        raise SystemExit(
            f"SWE-bench image preflight failed for {len(failures)}/"
            f"{len(image_specs)} unique {label} images using pull={pull!r}.\n"
            f"{shown}{more}"
        )
    print(
        f"SWE-bench image preflight: {len(image_specs)} unique {label} images "
        f"reachable (pull={pull})"
    )
    return len(image_specs)


def cleanup_reset_containers(run_root: str | Path, *, client_factory=None) -> int:
    """Remove stale SWE-bench Docker containers owned by an existing run root."""

    swebench_runs = Path(run_root) / "swebench_runs"
    if not swebench_runs.is_dir():
        return 0
    rollout_ids = sorted(path.name for path in swebench_runs.iterdir() if path.is_dir())
    if not rollout_ids:
        return 0

    try:
        if client_factory is None:
            import docker

            client_factory = docker.from_env
        client = client_factory()
    except Exception:
        return 0

    removed = 0
    for rollout_id in rollout_ids:
        try:
            containers = client.containers.list(all=True, filters={"name": rollout_id})
        except Exception:
            continue
        for container in containers:
            name = str(getattr(container, "name", ""))
            if not (name.startswith("swebench.") and name.endswith(f".{rollout_id}")):
                continue
            try:
                container.remove(force=True)
            except Exception:
                continue
            removed += 1
    return removed


@dataclass(frozen=True)
class ParallelResolution:
    """A resolved worker count plus a human-readable reason for the plan output."""

    workers: int
    detail: str


def resolve_parallel_workers(
    requested: str | int | None,
    num_instances: int,
    *,
    client_factory: Any = None,
) -> ParallelResolution:
    """Resolve ``--parallel`` as an explicit positive integer.

    ``None`` means the caller omitted the setting, so the safe framework default
    is one worker. ``client_factory`` is accepted for older tests/callers but is
    intentionally unused: worker counts are no longer inferred from Docker.
    """

    del client_factory
    instances = max(1, int(num_instances))
    text = "1" if requested is None else str(requested).strip()
    try:
        explicit = int(text)
    except ValueError:
        raise SystemExit(
            f"--parallel must be a positive integer; got {requested!r}"
        ) from None
    if explicit < 1:
        raise SystemExit(f"--parallel must be >= 1; got {requested!r}")
    return ParallelResolution(explicit, f"explicit; {instances} instances")


def branch_concurrency(*, global_workers: int, branches: int) -> int:
    """Per-branch Docker concurrency so all branches stay within the global cap.

    Two parallelism levels (instances within a rollout x branches per round)
    share one Docker VM. ``global_workers`` is the memory-bounded hard cap from
    ``resolve_parallel_workers``; splitting it across branches keeps in-flight
    containers <= ``global_workers`` (the stability guarantee), with a floor of 1
    so every branch always makes progress.
    """

    return max(1, int(global_workers) // max(1, int(branches)))
