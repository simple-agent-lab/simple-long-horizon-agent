"""The default rollout: a version in, eval-suite runs out.

A thin wrapper over the existing concurrent eval driver (``run_dataset``). The
run_id is deterministic in ``(version, slice)`` so an unchanged side is measured
once and reused across steps (measurement reuse, for free). The factory binds
deployment concerns; the slice travels per call.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

from simple_agent_lab.evals.dataset import run_dataset
from simple_agent_lab.evals.in_container import (
    API_KIND_CHOICES,
    API_KIND_ENV,
    OPENAI_AUTH_ENV,
    OPENAI_BASE_URL_ENV,
    OPENAI_MODEL_ENV,
)
from simple_agent_lab.evals.instances import InstanceSet
from simple_agent_lab.evals.protocols import ArtifactStore, ContainerBackend, Suite
from simple_agent_lab.evolution.surface import AgentSurface
from simple_agent_lab.evolution.types import PROVIDER_NAME, Run, Version

Rollout = Callable[[Version, InstanceSet], Sequence[Run]]


def _provider_args(version: Version) -> tuple[str, dict[str, str]]:
    """Translate a version's provider.json into run_dataset provider kwargs."""

    path = version.dir / PROVIDER_NAME
    if not path.is_file():
        return "fake", {}
    data = json.loads(path.read_text(encoding="utf-8"))
    api = data.get("api", "fake")
    if api == "fake":
        return "fake", {}
    model = data.get("model")
    if not model:
        raise ValueError(
            f"provider.json for version {version.hash} sets api={api!r} but no 'model'"
        )
    env = {OPENAI_MODEL_ENV: model}
    if data.get("base_url"):
        env[OPENAI_BASE_URL_ENV] = data["base_url"]
    if api in API_KIND_CHOICES:
        env[API_KIND_ENV] = api
    key_env = data.get("api_key_env", "")
    if key_env and os.environ.get(key_env):
        env[OPENAI_AUTH_ENV] = os.environ[key_env]
    return "openai", env


def dataset_rollout(
    *,
    suite: Suite,
    backend: ContainerBackend,
    store: ArtifactStore,
    runs_root: Path,
    concurrency: int = 1,
    run_kwargs: Mapping[str, Any] | None = None,
    version_artifacts: Callable[[Version], Mapping[str, bytes]] | None = None,
) -> Rollout:
    """Bind deployment concerns (suite/backend/store/runs_root) into a ``Rollout``.

    Returns a closure that takes the per-call ``(version, slice_)`` and returns
    one ``Run`` per instance, rolling only when the ``(version, slice)`` has not
    already been measured (otherwise it reuses the existing result.json).

    ``version_artifacts`` is an optional, suite-agnostic hook: given the version
    being rolled, it returns ``{artifact_key: bytes}`` staged into every run's
    ``input/`` dir. The evolution recipe uses it to make a version's evolved
    files (e.g. an agent scaffold) available in the container; the generic runner
    stays free of any suite knowledge.
    """

    runs_root = Path(runs_root).resolve()

    def rollout(version: Version, slice_: InstanceSet) -> Sequence[Run]:
        run_id = f"{version.hash}-{slice_.sha}"
        run_dir = runs_root / run_id
        if not _already_measured(run_dir, slice_):
            provider, provider_env = _provider_args(version)
            extra = dict(run_kwargs or {})
            api_kind = extra.pop("api_kind", _api_kind(provider_env))
            if version_artifacts is not None:
                staged = version_artifacts(version)
                if staged:
                    extra["extra_artifacts"] = dict(staged)
            report = run_dataset(
                suite=suite,
                instances=slice_.instances,
                backend=backend,
                store=store,
                run_root=runs_root,
                run_id=run_id,
                concurrency=concurrency,
                provider=provider,
                provider_env=provider_env,
                api_kind=api_kind,
                **extra,
            )
            if report.failed:
                details = "; ".join(
                    f"{result.instance_id}: {result.error or 'unknown error'}"
                    for result in report.failed
                )
                raise RuntimeError(f"dataset rollout failed: {details}")
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "version.json").write_text(
                json.dumps(
                    {"version": version.hash, "slice": slice_.id, "sha": slice_.sha}
                ),
                encoding="utf-8",
            )
        return [Run(p) for p in sorted(run_dir.iterdir()) if p.is_dir()]

    return rollout


def rollout_from_suite(
    *,
    suite: Suite,
    surface: AgentSurface,
    backend: ContainerBackend,
    store: ArtifactStore,
    runs_root: Path,
    concurrency: int = 1,
    run_kwargs: Mapping[str, Any] | None = None,
) -> Rollout:
    return dataset_rollout(
        suite=suite,
        backend=backend,
        store=store,
        runs_root=runs_root,
        concurrency=concurrency,
        run_kwargs=run_kwargs,
        version_artifacts=surface.artifacts_from_version,
    )


def _api_kind(provider_env: Mapping[str, str]) -> str:
    """The in-container API kind, mirroring the env the provider was built with."""

    return provider_env.get(API_KIND_ENV, "openai-chat")


def _already_measured(run_dir: Path, slice_: InstanceSet) -> bool:
    """A ``(version, slice)`` is measured only when every wanted instance wrote
    its ``out/result.json``. The harness creates each instance dir at run START,
    so dir presence alone would treat a crashed/partial run as done and skip the
    re-roll; gating on the result file makes a partial run re-roll instead."""

    if not run_dir.is_dir():
        return False
    want = {str(inst.get("instance_id", n)) for n, inst in enumerate(slice_.instances)}
    return all((run_dir / iid / "out" / "result.json").is_file() for iid in want)
