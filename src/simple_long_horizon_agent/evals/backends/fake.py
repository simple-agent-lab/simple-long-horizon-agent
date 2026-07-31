"""In-memory backend for tests and teaching — runs no real container/agent.

Use it to test orchestration (`run_suite_instance` store wiring, prediction
shaping, nonzero status) without running an agent: the optional `on_run` callback
simulates the container half by writing outputs through the bound store. For a
*real* end-to-end run without Docker, use `LocalProcessBackend` (it runs the
actual agent loop with whatever provider you configure).

Also implements the optional `submit` / `poll` pair so host-reentrant batches
(`submit_dataset` / `reconcile_dataset`) are testable without Docker: `submit`
runs `on_run` (the "container" writes its result to the store immediately, the
way a detached container eventually would) and `poll` reports completion. Because
results are persisted to the store at submit, a *fresh* `FakeBackend` with no
in-memory state can poll a handle to done — mirroring a new host process
re-attaching to already-finished containers.
"""

from __future__ import annotations

from typing import Callable

from ..protocols import (
    ArtifactStore,
    ContainerBinding,
    RunHandle,
    RunOutcome,
    RunSpec,
)

OnRun = Callable[[RunSpec, ArtifactStore], None]
BACKEND_KIND = "fake"


class FakeBackend:
    """Record specs and (optionally) simulate the run by writing via the store."""

    def __init__(
        self,
        *,
        on_run: OnRun | None = None,
        status_code: int = 0,
        log_text: str = "",
        pending_polls: int = 0,
    ) -> None:
        self._on_run = on_run
        self._status_code = status_code
        self._log_text = log_text
        # >0: poll() returns None this many times per ref before reporting done,
        # so the reconcile loop's "still running" path is exercised.
        self._pending_polls = pending_polls
        self.runs: list[RunSpec] = []
        self._remaining: dict[str, int] = {}

    def run(
        self,
        spec: RunSpec,
        *,
        store: ArtifactStore,
        binding: ContainerBinding,
    ) -> RunOutcome:
        del binding
        self.runs.append(spec)
        if self._on_run is not None:
            self._on_run(spec, store)
        return RunOutcome(status_code=self._status_code, logs=self._log_text)

    def submit(
        self,
        spec: RunSpec,
        *,
        store: ArtifactStore,
        binding: ContainerBinding,
    ) -> RunHandle:
        del binding
        self.runs.append(spec)
        if self._on_run is not None:
            self._on_run(spec, store)  # the "container" writes its result now
        return RunHandle(backend_kind=BACKEND_KIND, ref=spec.run_name, run_dir="")

    def poll(self, handle: RunHandle) -> RunOutcome | None:
        remaining = self._remaining.get(handle.ref, self._pending_polls)
        if remaining > 0:
            self._remaining[handle.ref] = remaining - 1
            return None
        return RunOutcome(status_code=self._status_code, logs=self._log_text)
