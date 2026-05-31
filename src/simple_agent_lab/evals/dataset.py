"""Concurrent dataset driver — the minimal "controller" over `run_suite_instance`.

Distributed RL/eval frameworks (slime, veRL, ROLL) put a controller above a pool
of rollout/reward workers. Most of their controller weight — GPU placement,
weight sync, streaming queues — is RL-training-specific and irrelevant to eval.
What remains, for *eval*, is just: run many instances concurrently over a pool of
backends and aggregate the outcomes.

So the controller here is a single function, not a process or a class hierarchy:
`run_dataset(...)` calls the already-stateless `run_suite_instance` once per
instance on a bounded `ThreadPoolExecutor` (stdlib only — no Ray, no asyncio, no
queue). The suite / backend / store protocols are unchanged; concurrency is
purely "call the pure function N times in a pool," with the `ArtifactStore` as
the result bus. The host loop itself is the controller.

Threads (not async) because `run_suite_instance` is blocking — a Docker `wait`
or an in-process agent loop — and the whole project is synchronous.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .protocols import ArtifactStore, ContainerBackend, RunArtifacts, Suite
from .runner import run_suite_instance


@dataclass(frozen=True)
class InstanceResult:
    """Outcome of one instance: either artifacts (it ran) or an error string."""

    instance_id: str
    artifacts: RunArtifacts | None
    error: str | None
    attempts: int

    @property
    def ok(self) -> bool:
        """True when the run completed without raising (regardless of exit code).

        A nonzero ``artifacts.status_code`` is a completed-but-failed run, not an
        error — inspect `artifacts.status_code` for that. `error` is reserved for
        raised exceptions (infra/transient), which is what `run_dataset` retries.
        """
        return self.error is None


@dataclass(frozen=True)
class DatasetReport:
    """Aggregate of one `run_dataset` call. Results follow input order."""

    results: list[InstanceResult]

    @property
    def ok(self) -> list[InstanceResult]:
        return [r for r in self.results if r.ok]

    @property
    def failed(self) -> list[InstanceResult]:
        return [r for r in self.results if not r.ok]

    def summary(self) -> dict[str, int]:
        return {
            "total": len(self.results),
            "ok": len(self.ok),
            "failed": len(self.failed),
        }


def _run_one(
    suite: Suite,
    instance: Mapping[str, Any],
    *,
    backend: ContainerBackend,
    store: ArtifactStore,
    run_root: Path,
    run_id: str,
    max_attempts: int,
    run_kwargs: dict[str, Any],
) -> InstanceResult:
    instance_id = str(instance["instance_id"])
    last_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            artifacts = run_suite_instance(
                suite=suite,
                instance=instance,
                backend=backend,
                store=store,
                run_root=run_root,
                run_id=run_id,
                **run_kwargs,
            )
            return InstanceResult(instance_id, artifacts, None, attempt)
        except Exception as exc:  # transient/infra failure — retry up to max_attempts
            last_error = f"{type(exc).__name__}: {exc}"
    return InstanceResult(instance_id, None, last_error, max_attempts)


def run_dataset(
    *,
    suite: Suite,
    instances: Iterable[Mapping[str, Any]],
    backend: ContainerBackend,
    store: ArtifactStore,
    run_root: Path,
    run_id: str,
    concurrency: int = 1,
    max_attempts: int = 1,
    on_result: Callable[[InstanceResult], None] | None = None,
    **run_kwargs: Any,
) -> DatasetReport:
    """Run a whole dataset by calling `run_suite_instance` once per instance.

    `concurrency` is the thread-pool size (default 1 = sequential, no behavior
    change). `max_attempts` retries only on a raised exception (infra/transient),
    not on a nonzero exit code. `on_result` is called once per finished instance
    from the calling thread (handy for progress or live aggregation). Extra
    keyword args (`provider`, `provider_env`, `max_turns`, `model_name`, …) pass
    straight through to each `run_suite_instance`.

    Per-instance artifacts land under the usual
    ``<run_root>/<run_id>/<instance_id>/`` tree, so concurrent runs never collide
    and the `ArtifactStore` is the shared result bus. Caveat: a single
    `LocalProcessBackend(workspace=...)` shares one workspace, so do not run it
    with `concurrency > 1`; Docker backends each get their own container and are
    safe to fan out.
    """

    items = list(instances)
    results: list[InstanceResult] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {
            pool.submit(
                _run_one,
                suite,
                instance,
                backend=backend,
                store=store,
                run_root=run_root,
                run_id=run_id,
                max_attempts=max_attempts,
                run_kwargs=run_kwargs,
            ): instance
            for instance in items
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if on_result is not None:
                on_result(result)

    order = {str(inst["instance_id"]): i for i, inst in enumerate(items)}
    results.sort(key=lambda r: order.get(r.instance_id, len(order)))
    return DatasetReport(results=results)
