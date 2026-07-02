"""On-disk layout for filesystem memory.

Everything that touches memory files lives here: atomic writes, the skeleton
files ``ensure_layout`` seeds, INDEX.md table edits, the navigation summary,
and the guarded MEMORY.md handbook rewrite.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from simple_agent_lab.memory.records import (
    DEFAULT_MAX_HANDBOOK_CHARS,
    MEMORY_HANDBOOK_FILENAME,
    MEMORY_SUMMARY_FILENAME,
    FilesystemIndexRow,
)


def safe_component(value: str) -> str:
    """Return a filesystem-safe path component."""

    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return safe or "default"


def unique_component(value: str, used: set[str]) -> str:
    """Return a safe path component that does not collide with prior components."""

    name = safe_component(value)
    if name not in used:
        used.add(name)
        return name
    stem, suffix = os.path.splitext(name)
    stem = stem or "artifact"
    index = 2
    while True:
        candidate = f"{stem}_{index}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        index += 1


def unique_run_id(memory_dir: Path, base_run_id: str) -> str:
    run_id = safe_component(base_run_id)
    run_dir = memory_dir / "runs" / run_id
    if not run_dir.exists():
        return run_id

    index = 2
    while True:
        candidate = f"{run_id}_{index}"
        if not (memory_dir / "runs" / candidate).exists():
            return candidate
        index += 1


def write_if_missing(path: Path, text: str) -> None:
    if not path.exists():
        write_text_atomic(path, text)


def write_text_atomic(path: Path, text: str) -> None:
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


def read_limited(path: Path, *, limit: int = 8_000) -> str:
    text = path.read_text(encoding="utf-8")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n\n... truncated ...\n"


def _escape_cell(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", " ").strip()


def _unescape_cell(value: str) -> str:
    return value.replace(r"\|", "|").strip()


def upsert_index_row(
    index_path: Path,
    *,
    run: str,
    row: FilesystemIndexRow,
    summary_path: str,
) -> None:
    """Insert or replace one INDEX.md row by its path cell."""

    text = (
        index_path.read_text(encoding="utf-8")
        if index_path.exists()
        else index_skeleton()
    )
    rendered = (
        f"| {_escape_cell(run)} | {_escape_cell(row.summary)} | "
        f"{_escape_cell(row.scope)} | {_escape_cell(row.signals)} | "
        f"{_escape_cell(row.keywords)} | {_escape_cell(row.artifacts)} | "
        f"{_escape_cell(summary_path)} |"
    )
    lines = text.splitlines()
    target_suffix = f"| {_escape_cell(summary_path)} |"
    replaced = False
    for index, line in enumerate(lines):
        if line.rstrip().endswith(target_suffix):
            lines[index] = rendered
            replaced = True
            break
    if not replaced:
        insert_at = next(
            (
                index
                for index, line in enumerate(lines)
                if line.strip() in {"## Memory Handbook", "## Memory Notes"}
            ),
            len(lines),
        )
        while insert_at > 0 and not lines[insert_at - 1].strip():
            insert_at -= 1
        lines.insert(insert_at, rendered)
    write_text_atomic(index_path, "\n".join(lines).rstrip() + "\n")


def parse_index_rows(index_text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    headers: list[str] = []
    for line in index_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = [_unescape_cell(cell) for cell in stripped.strip("|").split("|")]
        normalized = [cell.lower().replace(" ", "_") for cell in cells]
        if "summary" in normalized and "path" in normalized:
            headers = normalized
            continue
        if not headers or all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells, strict=True)))
    return rows


def first_summary_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "v1" or stripped.startswith("#"):
            continue
        if stripped.startswith("- `"):
            continue
        if stripped.startswith("Memory namespace:"):
            continue
        if stripped.startswith("Use this memory"):
            continue
        if stripped.startswith("No durable memory"):
            continue
        return stripped
    return ""


def update_memory_summary(memory_dir: Path, distilled_summary: str) -> None:
    """Update the top-level navigation summary for one memory namespace."""

    summary = sanitize_memory_summary(distilled_summary)
    if not summary:
        summary = render_memory_summary_from_index(memory_dir)
    write_text_atomic(memory_dir / MEMORY_SUMMARY_FILENAME, summary.rstrip() + "\n")


def sanitize_memory_summary(summary: str) -> str:
    text = summary.strip()
    if not text:
        return ""
    if not text.startswith("v1"):
        text = "v1\n\n" + text
    return text


def render_memory_summary_from_index(memory_dir: Path, *, limit: int = 5) -> str:
    rows = parse_index_rows((memory_dir / "INDEX.md").read_text(encoding="utf-8"))
    recent = rows[-limit:]
    lines = [
        "v1",
        "",
        "# Memory Summary",
        "",
        f"Memory namespace: `{memory_dir.name}`.",
        "",
        "Use this memory when the task overlaps the scopes, keywords, or failure signals below.",
        "",
        "## Recent Runs",
        "",
    ]
    if not recent:
        lines.append("- No durable run summaries yet.")
    else:
        for row in recent:
            bits = [row.get("summary", "")]
            if row.get("scope"):
                bits.append(f"scope: {row['scope']}")
            if row.get("keywords"):
                bits.append(f"keywords: {row['keywords']}")
            if row.get("path"):
                bits.append(f"evidence: {row['path']}")
            lines.append("- " + "; ".join(bit for bit in bits if bit))
    lines.extend(
        [
            "",
            "## Primary Files",
            "",
            f"- `{MEMORY_HANDBOOK_FILENAME}`: durable preferences, patterns, references, and failure shields.",
            "- `INDEX.md`: run-level routing by summary, scope, signals, and keywords.",
            "- `runs/*/summary.md`: compact evidence summaries; open transcripts only for exact evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def apply_handbook_rewrite(path: Path, proposed: str, *, run_dir: Path) -> None:
    """Persist a model-owned full rewrite of MEMORY.md, guarded against loss.

    The distiller returns the entire updated handbook (it owns merging, rewriting,
    and deleting). An empty rewrite means "no durable change" and keeps the current
    handbook untouched. A non-empty rewrite is written verbatim only when it passes
    :func:`_handbook_rewrite_rejection`; a rejected rewrite keeps the previous
    handbook and drops a compact marker so the skip stays visible instead of
    silently corrupting or erasing memory.
    """

    proposed = (proposed or "").strip()
    if not proposed:
        return
    existing = (
        path.read_text(encoding="utf-8") if path.exists() else handbook_skeleton()
    )
    reason = _handbook_rewrite_rejection(proposed, existing)
    if reason:
        write_text_atomic(
            run_dir / "memory_error.md",
            handbook_rejection_text(reason),
        )
        return
    write_text_atomic(path, proposed.rstrip() + "\n")


def _handbook_rewrite_rejection(proposed: str, existing: str) -> str:
    """Return a reason to reject a proposed handbook rewrite, or "" to accept.

    The model owns handbook content; these guards only catch catastrophic outputs:
    - oversize: a rewrite past ``DEFAULT_MAX_HANDBOOK_CHARS`` is a runaway response
      or a transcript dumped back into memory, not a small curated handbook.
    - malformed: no Markdown heading and no bullet looks truncated, not a handbook.
    - erasure: dropping every bullet when the current handbook had several is almost
      always accidental loss (e.g. truncation), not intentional pruning.
    """

    if len(proposed) > DEFAULT_MAX_HANDBOOK_CHARS:
        return (
            f"rewrite is {len(proposed)} chars, over the "
            f"{DEFAULT_MAX_HANDBOOK_CHARS}-char cap"
        )
    has_heading = any(line.lstrip().startswith("#") for line in proposed.splitlines())
    proposed_bullets = _handbook_bullet_count(proposed)
    if not has_heading and proposed_bullets == 0:
        return "rewrite has no heading and no bullet (looks truncated)"
    if _handbook_bullet_count(existing) >= 3 and proposed_bullets == 0:
        return "rewrite would drop every durable lesson from a non-empty handbook"
    return ""


def _handbook_bullet_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if re.match(r"^\s*[-*]\s+", line))


def handbook_rejection_text(reason: str) -> str:
    """Compact, durable marker for a rejected handbook rewrite."""

    return "\n".join(
        [
            "# Handbook rewrite rejected",
            "",
            f"- Reason: {reason}.",
            "- The previous MEMORY.md was kept unchanged.",
            "",
        ]
    )


def index_skeleton() -> str:
    return "\n".join(
        [
            "# Memory Index",
            "",
            "## Runs",
            "",
            "| Run | Summary | Scope | Signals | Keywords | Artifacts | Path |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            "",
            "## Memory Handbook",
            "",
            f"See `{MEMORY_HANDBOOK_FILENAME}`.",
            "",
        ]
    )


def handbook_skeleton() -> str:
    return "\n".join(
        [
            "# Memory Handbook",
            "",
            "Durable, high-signal lessons that should change future agent behavior.",
            "Keep this file small: merge, rewrite, or drop entries instead of appending forever.",
            "",
            "## Lessons",
            "",
        ]
    )


def memory_summary_skeleton(memory_name: str) -> str:
    return "\n".join(
        [
            "v1",
            "",
            "# Memory Summary",
            "",
            f"Memory namespace: `{memory_name}`.",
            "",
            "No durable memory has been recorded yet.",
            "",
            "Primary files:",
            f"- `{MEMORY_HANDBOOK_FILENAME}` for durable lessons.",
            "- `INDEX.md` for run-level routing.",
            "- `runs/*/summary.md` for compact evidence summaries.",
            "",
        ]
    )
