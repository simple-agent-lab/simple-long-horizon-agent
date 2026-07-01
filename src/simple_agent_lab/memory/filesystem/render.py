"""Markdown rendering and parsing for INDEX.md, MEMORY.md, and the summary file.

Everything here is pure text in, text out: it knows the on-disk Markdown
formats but nothing about run orchestration or the LLM distillation call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from simple_agent_lab.memory.filesystem.artifacts import FilesystemArtifact
from simple_agent_lab.memory.filesystem.io import (
    _escape_cell,
    _unescape_cell,
    _write_text_atomic,
)

MEMORY_SUMMARY_FILENAME = "memory_summary.md"
MEMORY_HANDBOOK_FILENAME = "MEMORY.md"

# Hard upper bound on the rewritten MEMORY.md handbook. The distiller returns the
# full updated handbook (the model owns merge/delete/rewrite), so a runaway or
# truncated response must not be persisted verbatim. A proposed rewrite larger than
# this is rejected and the previous handbook is kept untouched. ~20k characters is a
# generous ceiling for a small, curated lessons file while still catching unbounded
# growth or a model that dumped the whole transcript back into memory.
DEFAULT_MAX_HANDBOOK_CHARS = 20_000


@dataclass(frozen=True)
class FilesystemIndexRow:
    summary: str = ""
    scope: str = ""
    signals: str = ""
    keywords: str = ""
    artifacts: str = ""


def sanitize_summary(summary: str) -> str:
    """Remove evaluation/outcome sections from distilled memory."""

    return re.sub(
        r"(?ims)^#+\s*(Outcome|Evaluation|Score)\b.*?(?=^#+\s|\Z)",
        "",
        summary,
    ).strip()


def complete_index_row(
    row: FilesystemIndexRow,
    task: str,
    artifacts: tuple[FilesystemArtifact, ...],
) -> FilesystemIndexRow:
    """Fill minimal recall cells so INDEX.md stays useful without distillation."""

    return FilesystemIndexRow(
        summary=row.summary.strip() or _short_line(task),
        scope=row.scope.strip() or "general",
        signals=row.signals.strip(),
        keywords=row.keywords.strip() or keywords_from_text(task),
        artifacts=row.artifacts.strip()
        or ", ".join(artifact.name for artifact in artifacts),
    )


def fallback_summary(
    task: str,
    artifacts: tuple[FilesystemArtifact, ...],
    error: Exception | None = None,
) -> str:
    """Return a small summary that keeps INDEX.md links valid."""

    artifact_lines = (
        "\n".join(f"- `{artifact.name}`" for artifact in artifacts)
        if artifacts
        else "- None"
    )
    status = (
        f"Distillation unavailable: {type(error).__name__}."
        if error is not None
        else "No distilled reusable lesson was produced."
    )
    return "\n".join(
        [
            "## Task",
            "",
            task.strip() or "(unknown)",
            "",
            "## Key Signals",
            "",
            status,
            "",
            "## Useful Context",
            "",
            "Raw evidence is available in `task.md` and `transcript.md`.",
            "",
            "## Actions And Artifacts",
            "",
            artifact_lines,
            "",
            "## Failed Or Risky Attempts",
            "",
            "- None recorded.",
            "",
            "## Reusable Lessons",
            "",
            "- None recorded.",
            "",
        ]
    )


def artifact_manifest(artifacts: tuple[FilesystemArtifact, ...]) -> str:
    """Describe stored artifacts without changing their raw content."""

    lines = ["# Artifacts", ""]
    if not artifacts:
        lines.append("No artifacts were recorded.")
        lines.append("")
        return "\n".join(lines)
    for artifact in artifacts:
        lines.extend(
            [
                f"## {artifact.name}",
                "",
                f"- Path: `artifacts/{artifact.name}`",
                f"- Description: {artifact.description.strip() or 'No description provided.'}",
                "",
            ]
        )
    return "\n".join(lines)


def memory_error_text(title: str, exc: Exception) -> str:
    """Render a durable but compact marker for best-effort memory failures."""

    return "\n".join(
        [
            f"# {title}",
            "",
            f"- Error type: `{type(exc).__name__}`",
            f"- Message: {str(exc).strip() or '(empty)'}",
            "",
        ]
    )


def keywords_from_text(text: str) -> str:
    """Build a tiny fallback keyword list from the task text."""

    words = re.findall(r"[A-Za-z0-9_][A-Za-z0-9_.-]{2,}", text.lower())
    seen: list[str] = []
    for word in words:
        if word not in seen:
            seen.append(word)
        if len(seen) >= 6:
            break
    return ", ".join(seen)


def _short_line(text: str, *, limit: int = 80) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "..."


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
        else _index_skeleton()
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
    _write_text_atomic(index_path, "\n".join(lines).rstrip() + "\n")


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
    _write_text_atomic(memory_dir / MEMORY_SUMMARY_FILENAME, summary.rstrip() + "\n")


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


def _apply_handbook_rewrite(path: Path, proposed: str, *, run_dir: Path) -> None:
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
        path.read_text(encoding="utf-8") if path.exists() else _handbook_skeleton()
    )
    reason = _handbook_rewrite_rejection(proposed, existing)
    if reason:
        _write_text_atomic(
            run_dir / "memory_error.md",
            handbook_rejection_text(reason),
        )
        return
    _write_text_atomic(path, proposed.rstrip() + "\n")


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


def _index_skeleton() -> str:
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


def _handbook_skeleton() -> str:
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


def _memory_summary_skeleton(memory_name: str) -> str:
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
