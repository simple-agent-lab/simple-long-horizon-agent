"""In-memory backend for tests and teaching — runs no real container/agent.

Use it to test orchestration (`run_suite_instance` store wiring, prediction
shaping, nonzero status) without running an agent: the optional `on_run` callback
simulates the container half by writing outputs through the bound store. For a
*real* end-to-end run without Docker, use `LocalProcessBackend` (it runs the
actual agent loop with whatever provider you configure).
"""

from __future__ import annotations

from typing import Callable

from ..protocols import ArtifactStore, ContainerBinding, RunOutcome, RunSpec

OnRun = Callable[[RunSpec, ArtifactStore], None]


class FakeBackend:
    """Record specs and (optionally) simulate the run by writing via the store."""

    def __init__(
        self,
        *,
        on_run: OnRun | None = None,
        status_code: int = 0,
        log_text: str = "",
    ) -> None:
        self._on_run = on_run
        self._status_code = status_code
        self._log_text = log_text
        self.runs: list[RunSpec] = []

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
