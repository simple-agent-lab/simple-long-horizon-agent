"""In-process backend: run the agent loop here, no Docker (local development).

This is the seam that makes "same code, local dev → multi-machine deploy" real:
`run_suite_instance(..., backend=LocalProcessBackend())` runs the *exact* suite
container half that a container would run, but in the current Python process —
fast iteration, a debugger, no image build. Switch to `LocalDockerBackend()` (or
a remote backend) with no other code change to run it containerized.

The genuine difference is the execution environment, not the code: a container
gets its workspace + toolchain from the image (`plan.workdir`, e.g. `/testbed`);
in-process you supply a local `workspace` directory. Heavy suites (SWE-bench)
still need their image; light suites and judges run in-process directly.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

from ..protocols import ArtifactStore, ContainerBinding, RunOutcome, RunSpec


class LocalProcessBackend:
    """Execute a `RunSpec` in this process by calling the generic runner directly.

    Concurrency note: a single fixed `workspace` is shared by every run, so it is
    only safe with `concurrency=1`. To fan out in-process safely, pass either
    nothing (each run gets its own throwaway temp dir) or a *factory*
    ``workspace=lambda spec: base / spec.instance_id`` that hands each run its own
    directory. The factory form is also how you give each run pre-seeded content
    without runs colliding.
    """

    def __init__(
        self,
        *,
        workspace: str | Path | Callable[[RunSpec], Path] | None = None,
    ) -> None:
        # In a container the workspace comes from the image; in process the caller
        # provides it. Split into two attributes so the resolve path is a clean
        # callable-vs-fixed branch:
        #   factory  -> per-run dir derived from the spec (concurrency-safe)
        #   fixed    -> one shared dir (concurrency=1 only)
        #   neither  -> a per-run temp dir (concurrency-safe)
        self._workspace_fixed: Path | None = None
        self._workspace_factory: Callable[[RunSpec], Path] | None = None
        if isinstance(workspace, (str, Path)):
            self._workspace_fixed = Path(workspace)
        elif workspace is not None:
            self._workspace_factory = workspace

    def _resolve_workspace(self, spec: RunSpec) -> Path:
        if self._workspace_factory is not None:
            workdir = Path(self._workspace_factory(spec))
        elif self._workspace_fixed is not None:
            workdir = self._workspace_fixed
        else:
            workdir = Path(tempfile.mkdtemp(prefix="sal-workspace-"))
        workdir.mkdir(parents=True, exist_ok=True)
        return workdir

    def run(
        self,
        spec: RunSpec,
        *,
        store: ArtifactStore,
        binding: ContainerBinding,
    ) -> RunOutcome:
        # Imported lazily so host-only callers (and `import ...evals`) stay light:
        # this pulls in the agent runtime.
        import json

        from ..in_container import (
            provider_from_env,
            request_extra_from_env,
            run_in_container,
        )
        from ..protocols import INSTANCE_KEY

        del (
            binding
        )  # in-process uses the bound store object directly, no env round-trip

        instance = json.loads(store.get(INSTANCE_KEY).decode("utf-8"))
        provider = provider_from_env(
            kind=spec.provider, api_kind=spec.api_kind, env=spec.provider_env
        )
        workdir = self._resolve_workspace(spec)
        request_extra = (
            request_extra_from_env(env=spec.provider_env)
            if spec.provider == "openai"
            else {}
        )
        try:
            run_in_container(
                instance=instance,
                container_module=spec.container_module,
                provider=provider,
                workdir=workdir,
                max_turns=spec.max_turns,
                store=store,
                trace_id=f"{spec.suite_name}.{spec.instance_id}",
                producer=f"suite:{spec.suite_name}",
                suite_name=spec.suite_name,
                request_extra=request_extra,
            )
        except Exception as exc:  # surface as a nonzero status, like a container
            return RunOutcome(status_code=1, logs=f"{type(exc).__name__}: {exc}")
        return RunOutcome(status_code=0)
