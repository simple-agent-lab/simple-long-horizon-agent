"""Copy-out transport: the cloud path. Documented stub (ADR 0017).

Unlike `BindMountTransport`, this requests **no** bind mounts. Instead it:

  - stages inputs by ``put_file`` / ``put_archive`` into the container before
    start (instance record, runner code, wheels), and
  - collects outputs after the run via ``handle.get_archive(out_dir)``,
    unpacking the tar stream into the host ``run_dir/out/``.

Because neither step assumes a shared filesystem, this is what lets the same
eval run against a remote daemon (``DOCKER_HOST=tcp://…``), Kubernetes, or a
managed container service. Live trace under this transport must use a non-file
`TraceSink` (e.g. `HttpTraceSink`) since there is no file to tail.

Implementing this is the cloud cutover tracked by ADR 0017; the seam is fixed
here so suites and the runner never change when it lands.
"""

from __future__ import annotations

from pathlib import Path

from ..protocols import ContainerHandle, StagedFile


class CopyOutTransport:
    """Archive-in / archive-out transport for remote daemons. Not implemented yet."""

    def mounts(self, run_dir: Path) -> dict[str, dict[str, str]]:
        # No bind mounts: the whole point is to avoid a shared filesystem.
        del run_dir
        return {}

    def stage_inputs(
        self,
        handle: ContainerHandle,
        *,
        run_dir: Path,
        files: tuple[StagedFile, ...],
    ) -> None:  # pragma: no cover - stub
        raise NotImplementedError(
            "CopyOutTransport is a documented stub (ADR 0017): it will copy "
            "input/instance.json + runner code in via put_file, then pull "
            "out/ back via get_archive. Use BindMountTransport locally."
        )

    def collect_outputs(
        self, handle: ContainerHandle, *, run_dir: Path
    ) -> None:  # pragma: no cover - stub
        raise NotImplementedError(
            "CopyOutTransport.collect_outputs will unpack handle.get_archive("
            "out_dir) into run_dir/out/ (ADR 0017)."
        )
