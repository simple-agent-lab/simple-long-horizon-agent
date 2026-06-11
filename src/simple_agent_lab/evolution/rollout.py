"""Rollout adapter: a bundle in, eval-suite runs out.

``Rollout`` (defined in evolution/gate.py) is just a callable so the gate
can be tested with stubs; this module provides the real implementation over
the containerized eval framework. The slice arrives per call — the same
rollout serves the main gate, guard slices, held-out rotation, and shadow
evaluation. The signature deliberately carries everything a future HTTP
wrapper needs — serving it later changes the executor, not the contract.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from simple_agent_lab.evals.dataset import run_dataset
from simple_agent_lab.evals.in_container import (
    API_KIND_CHOICES,
    API_KIND_ENV,
    OPENAI_AUTH_ENV,
    OPENAI_BASE_URL_ENV,
    OPENAI_MODEL_ENV,
)
from simple_agent_lab.evals.protocols import ArtifactStore, ContainerBackend, Suite
from simple_agent_lab.evolution.bundle import Bundle, load_provider
from simple_agent_lab.evolution.catalog import Run
from simple_agent_lab.evolution.gate import EvalSlice, Rollout


def _provider_args(bundle_dir: Path) -> tuple[str, dict[str, str]]:
    """Map the bundle's provider.json onto the in-container env contract.

    The container builds its Provider from env (`provider_from_env`), so the
    bundle's model choice travels as environment variables. A bundle without
    provider.json runs the deterministic fake provider.
    """

    provider = load_provider(bundle_dir)
    if provider is None or provider.api == "fake":
        return "fake", {}
    env = {OPENAI_MODEL_ENV: provider.model}
    if provider.base_url:
        env[OPENAI_BASE_URL_ENV] = provider.base_url
    if provider.api in API_KIND_CHOICES:
        env[API_KIND_ENV] = provider.api
    if provider.api_key_env and os.environ.get(provider.api_key_env):
        env[OPENAI_AUTH_ENV] = os.environ[provider.api_key_env]
    return "openai", env


def dataset_rollout(
    *,
    suite: Suite,
    backend: ContainerBackend,
    store: ArtifactStore,
    runs_root: Path,
    concurrency: int = 1,
    run_kwargs: Mapping[str, Any] | None = None,
) -> Rollout:
    """Build a Rollout that fans the per-call slice out through `run_dataset`.

    NOTE (skeleton): only the provider travels into the container today.
    Injecting prompt.md / playbook / lessons / skills requires the suite
    `agent_spec` seam described in the spec (§2.4) — the next increment.
    """

    def rollout(bundle: Bundle, slice_: EvalSlice, run_id: str) -> Sequence[Run]:
        provider, provider_env = _provider_args(bundle.dir)
        run_dataset(
            suite=suite,
            instances=slice_.instances,
            backend=backend,
            store=store,
            run_root=runs_root,
            run_id=run_id,
            concurrency=concurrency,
            provider=provider,
            provider_env=provider_env,
            **dict(run_kwargs or {}),
        )
        run_dir = runs_root / run_id
        # Host-side provenance stamp: which bundle produced this run set.
        (run_dir / "bundle.json").write_text(
            json.dumps(
                {
                    "bundle": bundle.hash,
                    "level": bundle.manifest.level,
                    "slice": slice_.describe(),
                }
            )
        )
        return [Run(p) for p in sorted(run_dir.iterdir()) if p.is_dir()]

    return rollout
