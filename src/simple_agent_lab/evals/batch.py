"""Host-reentrant batches: submit now, reconcile later (even from a new process).

Long eval runs (a SWE-bench agent can take minutes per instance) shouldn't pin
the host. With a backend that detaches its work (`LocalDockerBackend`), you can:

    submit_dataset(...)        # start every instance's container, write a manifest, return
    # ... host may exit / disconnect; containers keep running ...
    reconcile_dataset(...)     # a fresh process reloads the manifest and polls to completion

The manifest is a plain-JSON list of `RunHandle`s persisted in the
`ArtifactStore` (key ``BATCH_KEY``), so reconciliation depends on the store and
the daemon, not on any in-memory state from the submitting process. Predictions
are shaped from each run's ``result.json`` exactly as `run_suite_instance` does,
so a reconciled batch produces the same artifacts as a blocking run.

This sits beside `run_dataset` (blocking, simplest) — it does not replace it. Use
`run_dataset` for short / local runs; use submit + reconcile when the host must
be able to leave.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from .dataset import DatasetReport, InstanceResult
from .protocols import (
    RESULT_KEY,
    TRACE_KEY,
    ArtifactStore,
    RunArtifacts,
    RunHandle,
    RunOutcome,
    Suite,
)
from .runner import (
    GENERIC_RUNNER_MODULE,
    _prepare_run,
)

# Batch manifest lives at the batch root (run_root/<run_id>), above per-instance
# dirs, so one reload finds every handle.
BATCH_KEY = "batch.json"

# Lower bound on the idle wait between reconcile poll sweeps, so a caller passing
# poll_interval_s=0 (or near-0) cannot busy-spin the daemon. A `sleep_fn` is still
# injectable for tests (which stub it to a no-op).
_MIN_POLL_INTERVAL_S = 0.5


def _write_manifest(batch_store: ArtifactStore, manifest: list[dict[str, Any]]) -> None:
    """Atomically (re)write the whole manifest. Small + atomic, so crash-safe."""

    batch_store.put(
        BATCH_KEY, (json.dumps(manifest, ensure_ascii=False) + "\n").encode("utf-8")
    )


def _has_result(store: ArtifactStore, run_dir: str) -> bool:
    # ValueError covers a store (e.g. HostHttpStore) whose bind rejects a run_dir
    # outside its base — treat "can't reach it" as "no result", never crash the
    # reconcile loop over one handle.
    try:
        store.bind(Path(run_dir)).get(RESULT_KEY)
        return True
    except (FileNotFoundError, OSError, ValueError):
        return False


def _batch_store(store: ArtifactStore, run_root: Path, run_id: str) -> ArtifactStore:
    from .runner import _safe_part

    return store.bind(Path(run_root).resolve() / _safe_part(run_id))


def submit_dataset(
    *,
    suite: Suite,
    instances: Iterable[Mapping[str, Any]],
    backend: Any,  # must provide submit(); checked below
    store: ArtifactStore,
    run_root: Path,
    run_id: str,
    provider: str = "openai",
    api_kind: str = "openai-chat",
    max_turns: int = 75,
    wall_time_seconds: float | None = None,
    provider_env: Mapping[str, str] | None = None,
    runner_module: str = GENERIC_RUNNER_MODULE,
    install: bool = True,
    package_extras: tuple[str, ...] = (),
    wheelhouse_mount: str | None = None,
) -> list[RunHandle]:
    """Start every instance's run (without waiting) and persist a batch manifest.

    Returns the handles; they are also written to ``<run_id>/batch.json`` so a
    later `reconcile_dataset` can recover them without this process's memory.
    The manifest is re-persisted **after each container starts**, so if the host
    dies mid-submit, every already-started container is recorded and recoverable
    (no orphans). Requires a backend that implements `submit` (detached runs) —
    e.g. `LocalDockerBackend`; `LocalProcessBackend` cannot outlive the host.
    """

    if not hasattr(backend, "submit"):
        raise TypeError(
            f"{type(backend).__name__} has no submit(); submit_dataset needs a "
            "detaching backend such as LocalDockerBackend."
        )

    batch_store = _batch_store(store, run_root, run_id)
    handles: list[RunHandle] = []
    manifest: list[dict[str, Any]] = []
    for instance in instances:
        prepared = _prepare_run(
            suite=suite,
            instance=instance,
            store=store,
            run_root=run_root,
            run_id=run_id,
            max_turns=max_turns,
            provider=provider,
            api_kind=api_kind,
            provider_env=provider_env,
            runner_module=runner_module,
            install=install,
            package_extras=package_extras,
            wheelhouse_mount=wheelhouse_mount,
            name=None,
            wall_time_seconds=wall_time_seconds,
        )
        handle = backend.submit(
            prepared.spec, store=prepared.store, binding=prepared.binding
        )
        # Pin the run_dir so reconcile can locate the result without re-deriving.
        handle = RunHandle(
            backend_kind=handle.backend_kind,
            ref=handle.ref,
            run_dir=str(prepared.paths.root),
            extra={
                **dict(handle.extra),
                "instance_id": prepared.spec.instance_id,
            },
        )
        handles.append(handle)
        manifest.append(_handle_to_dict(handle))
        # Persist after each start: a mid-submit crash still leaves every
        # already-started container in the manifest, so none becomes an orphan.
        _write_manifest(batch_store, manifest)

    return handles


def reconcile_dataset(
    *,
    suite: Suite,
    backend: Any,  # must provide poll()
    store: ArtifactStore,
    run_root: Path,
    run_id: str,
    poll_interval_s: float = 5.0,
    timeout_s: float | None = None,
    on_result: Callable[[InstanceResult], None] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> DatasetReport:
    """Reload a submitted batch's manifest and poll every run to completion.

    Safe to call from a fresh process: it reads the handles from the store, not
    from memory. A run is done when its ``out/result.json`` exists (the single
    decoupling artifact); reconcile only confirms the result landed and reports
    where artifacts are. Any follow-up scoring reads ``out/result.json`` back.
    """

    if not hasattr(backend, "poll"):
        raise TypeError(
            f"{type(backend).__name__} has no poll(); reconcile_dataset needs a "
            "detaching backend such as LocalDockerBackend."
        )

    raw = _batch_store(store, run_root, run_id).get(BATCH_KEY)
    handles = [_handle_from_dict(d) for d in json.loads(raw.decode("utf-8"))]

    pending = {h.ref: h for h in handles}
    done: dict[str, InstanceResult] = {}
    deadline = None if timeout_s is None else time.monotonic() + timeout_s

    while pending:
        for ref, handle in list(pending.items()):
            outcome = backend.poll(handle)
            if outcome is None:
                # The container may be running, already collected (a prior
                # reconcile / another process), or gone. A `result.json` on disk
                # is the terminal truth: take it as done so we never deadlock on
                # a container the daemon no longer reports.
                if _has_result(store, handle.run_dir):
                    outcome = RunOutcome(status_code=0)
                else:
                    continue
            instance_id = str(handle.extra.get("instance_id") or ref)
            result = _finish(
                store=store,
                handle=handle,
                instance_id=instance_id,
                status_code=outcome.status_code,
            )
            done[ref] = result
            del pending[ref]
            if on_result is not None:
                on_result(result)
        if not pending:
            break
        if deadline is not None and time.monotonic() >= deadline:
            for ref, handle in pending.items():
                done[ref] = InstanceResult(
                    instance_id=str(handle.extra.get("instance_id") or ref),
                    artifacts=None,
                    error="timeout waiting for run to finish",
                    attempts=1,
                )
            break
        # Floor the wait so a caller passing 0 (or a tiny value) doesn't turn the
        # poll loop into a 100%-CPU spin hammering the daemon. timeout_s is still
        # opt-in: a long batch legitimately waits indefinitely by default, but it
        # waits *idle*.
        sleep_fn(max(poll_interval_s, _MIN_POLL_INTERVAL_S))

    ordered = [done[h.ref] for h in handles if h.ref in done]
    return DatasetReport(results=ordered)


def _finish(
    *,
    store: ArtifactStore,
    handle: RunHandle,
    instance_id: str,
    status_code: int,
) -> InstanceResult:
    run_dir = Path(handle.run_dir)
    # Artifact paths follow the same keys as the blocking path, so they track
    # TRACE_KEY rather than re-hardcoding the layout. ``result.json`` is the
    # decoupling artifact; any follow-up scoring reads it back.
    trajectory_path = run_dir / TRACE_KEY
    has_result = _has_result(store, handle.run_dir)

    artifacts = RunArtifacts(
        instance_id=instance_id,
        run_dir=run_dir,
        trajectory_path=trajectory_path,
        status_code=status_code,
        logs="",
    )
    error = (
        None if has_result else f"run finished (exit={status_code}) without a result"
    )
    return InstanceResult(instance_id, artifacts, error, attempts=1)


def _handle_to_dict(h: RunHandle) -> dict[str, Any]:
    return {
        "backend_kind": h.backend_kind,
        "ref": h.ref,
        "run_dir": h.run_dir,
        "extra": dict(h.extra),
    }


def _handle_from_dict(d: Mapping[str, Any]) -> RunHandle:
    return RunHandle(
        backend_kind=str(d["backend_kind"]),
        ref=str(d["ref"]),
        run_dir=str(d.get("run_dir", "")),
        extra=dict(d.get("extra", {})),
    )
