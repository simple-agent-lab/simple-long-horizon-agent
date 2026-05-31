"""Bind-mount transport: the local default.

The host run directory is bind-mounted into the container, so the container
writes ``out/trajectory.jsonl`` and ``out/prediction.jsonl`` straight onto the
host filesystem. Inputs are already visible the same way. Outputs need no
copy step, and a live viewer can tail the file with zero latency.

Assumes the Docker daemon and this process share a filesystem. For a remote
or cloud daemon, use `CopyOutTransport` instead (see ADR 0017).
"""

from __future__ import annotations

from pathlib import Path

from ..protocols import ContainerHandle, StagedFile

# The container-side mount point for the per-instance run directory. Matches
# the existing SWE-bench launcher's ``--run-mount`` default.
RUN_MOUNT = "/agent/run"


class BindMountTransport:
    """Shared-filesystem transport. Inputs/outputs flow through a bind mount."""

    def __init__(self, run_mount: str = RUN_MOUNT) -> None:
        self.run_mount = run_mount

    def mounts(self, run_dir: Path) -> dict[str, dict[str, str]]:
        return {str(run_dir.resolve()): {"bind": self.run_mount, "mode": "rw"}}

    def stage_inputs(
        self,
        handle: ContainerHandle,
        *,
        run_dir: Path,
        files: tuple[StagedFile, ...],
    ) -> None:
        # The run dir (with input/instance.json) is already visible via the
        # bind mount; only out-of-tree files (runner code, uv binary) are
        # copied directly into the container.
        for staged in files:
            handle.put_file(staged)

    def collect_outputs(self, handle: ContainerHandle, *, run_dir: Path) -> None:
        # Nothing to do: the container wrote ``out/*`` onto the host through the
        # bind mount. The runner reads them straight from ``run_dir``.
        del handle, run_dir
