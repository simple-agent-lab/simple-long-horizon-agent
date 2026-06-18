"""Safe filesystem paths for self-evolving run roots."""

from __future__ import annotations

from pathlib import Path


def safe_run_root(output_root: str | Path, run_id: str) -> Path:
    """Return the resolved run root for one run id under an output root.

    A run id names exactly one immediate child directory. It is not a path.
    """

    if not run_id:
        raise ValueError("unsafe run id: must be non-empty")
    if run_id in {".", ".."}:
        raise ValueError(f"unsafe run id {run_id!r}: must not be '.' or '..'")
    if "/" in run_id or "\\" in run_id:
        raise ValueError(f"unsafe run id {run_id!r}: must not contain path separators")

    run_path = Path(run_id)
    if run_path.is_absolute():
        raise ValueError(f"unsafe run id {run_id!r}: must be relative")

    root = Path(output_root).resolve(strict=False)
    candidate = (root / run_id).resolve(strict=False)
    if candidate.parent != root:
        raise ValueError(
            f"unsafe run id {run_id!r}: run root must be an immediate child of {root}"
        )
    return candidate
