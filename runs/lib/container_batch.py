"""Small shared host loop for container benchmark batches."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from simple_long_horizon_agent.evals import (
    ArtifactStore,
    ContainerBackend,
    DatasetReport,
    InstanceResult,
    Suite,
    run_dataset,
)


def run_container_batch(
    *,
    suite: Suite,
    instances: Sequence[Mapping[str, Any]],
    backend: ContainerBackend,
    store: ArtifactStore,
    run_root: Path,
    run_id: str,
    parallel: int,
    per_instance_kwargs: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    **run_kwargs: Any,
) -> tuple[DatasetReport, int]:
    """Run, log, and count both raised and nonzero container outcomes."""

    def record(result: InstanceResult) -> None:
        if result.artifacts is None:
            text, status = result.error or "run failed", "error"
        else:
            text, status = result.artifacts.logs, result.artifacts.status_code
        log = run_root / run_id / f"{result.instance_id}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(text, encoding="utf-8")
        print(f"    {result.instance_id}: {status}")

    report = run_dataset(
        suite=suite,
        instances=instances,
        backend=backend,
        store=store,
        run_root=run_root,
        run_id=run_id,
        concurrency=parallel,
        on_result=record,
        per_instance_kwargs=per_instance_kwargs,
        **run_kwargs,
    )
    failed = sum(
        result.error is not None
        or result.artifacts is None
        or result.artifacts.status_code != 0
        for result in report.results
    )
    return report, failed
