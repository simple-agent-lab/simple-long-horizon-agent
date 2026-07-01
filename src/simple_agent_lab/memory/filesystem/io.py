"""Low-level, domain-agnostic file and text primitives for filesystem memory."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _write_if_missing(path: Path, text: str) -> None:
    if not path.exists():
        _write_text_atomic(path, text)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        tmp = Path(handle.name)
        try:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    # NamedTemporaryFile creates the file 0600; memory is meant to persist and be
    # inspected across runs (and across a container/host bind mount where the
    # writer is root), so normalize to a normal readable mode honoring umask.
    _relax_file_permissions(path)


def _relax_file_permissions(path: Path) -> None:
    try:
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(path, 0o666 & ~umask)
    except OSError:
        # Best-effort: a filesystem that rejects chmod must not fail the write.
        return


def _escape_cell(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", " ").strip()


def _unescape_cell(value: str) -> str:
    return value.replace(r"\|", "|").strip()


def _read_limited(path: Path, *, limit: int = 8_000) -> str:
    text = path.read_text(encoding="utf-8")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n\n... truncated ...\n"


def _truncate_for_distiller(text: str, *, limit: int) -> str:
    """Bound transcript text sent to the distiller, keeping head and tail.

    The full transcript is still written to disk; this only limits the
    model-call input so long runs do not overflow the distiller context.
    """

    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return (
        text[:head].rstrip()
        + "\n\n... transcript truncated for distillation ...\n\n"
        + text[-tail:].lstrip()
    )
