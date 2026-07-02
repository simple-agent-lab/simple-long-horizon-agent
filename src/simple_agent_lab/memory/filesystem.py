"""Filesystem memory.

This implementation keeps raw evidence and distilled notes in Markdown files
under a memory-specific directory. The model reads memory through ordinary file
tools such as bash or an MCP filesystem server; this memory module only injects
the policy/path and writes evidence after the run.

The package splits along the run lifecycle: :mod:`.records` holds the shared
types and limits, :mod:`.store` owns the on-disk layout, :mod:`.distill` turns
run evidence into distilled Markdown, and this module orchestrates them behind
the :class:`FilesystemMemory` interface.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from simple_agent_lab.memory.base import Memory, MemoryContext, memory_context_message
from simple_agent_lab.memory.distill import (
    artifact_manifest,
    coerce_distillation,
    complete_index_row,
    default_artifacts,
    fallback_summary,
    memory_error_text,
    normalize_artifacts,
    retarget_distillation,
    sanitize_summary,
    truncate_for_distiller,
)
from simple_agent_lab.memory.records import (
    DEFAULT_DISTILLER_TRANSCRIPT_LIMIT,
    DEFAULT_FILESYSTEM_MEMORY_ROOT,
    MEMORY_HANDBOOK_FILENAME,
    MEMORY_SUMMARY_FILENAME,
    ArtifactBuilder,
    Distiller,
    FilesystemDistillation,
    FilesystemMemoryPayload,
)
from simple_agent_lab.memory.store import (
    apply_handbook_rewrite,
    first_summary_line,
    handbook_skeleton,
    index_skeleton,
    memory_summary_skeleton,
    read_limited,
    safe_component,
    unique_run_id,
    update_memory_summary,
    upsert_index_row,
    write_if_missing,
    write_text_atomic,
)
from simple_agent_lab.memory.transcript import (
    first_user_text,
    render_transcript_markdown,
)
from simple_agent_lab.messages import Message


class FilesystemMemory(Memory):
    """Per-memory Markdown directory plus run-end evidence writes."""

    def __init__(
        self,
        *,
        root: str | Path = DEFAULT_FILESYSTEM_MEMORY_ROOT,
        distiller: Distiller | None = None,
        artifact_builder: ArtifactBuilder | None = None,
        enabled: bool = True,
        distiller_transcript_limit: int = DEFAULT_DISTILLER_TRANSCRIPT_LIMIT,
    ) -> None:
        self.root = Path(root).expanduser()
        self.distiller = distiller
        self.artifact_builder = artifact_builder or default_artifacts
        self.enabled = enabled
        self.distiller_transcript_limit = distiller_transcript_limit

    def initial(self, ctx: MemoryContext) -> tuple[Message, ...]:
        if not self.enabled:
            return ()
        if ctx.memory_name:
            memory_dir = self.memory_dir(ctx)
            self.ensure_layout(memory_dir)
            return (
                memory_context_message(
                    _policy_block(
                        memory_dir,
                        read_limited(memory_dir / MEMORY_SUMMARY_FILENAME, limit=2_000),
                    ),
                    target=ctx.agent,
                ),
            )
        available = self.available_memories()
        if not available:
            return ()
        return (
            memory_context_message(
                _root_policy_block(self.root, self.memory_overview(available)),
                target=ctx.agent,
            ),
        )

    def finish(self, ctx: MemoryContext) -> None:
        if not self.enabled or ctx.state is None:
            return
        try:
            self._finish(ctx)
        except Exception as exc:
            self._record_finish_error(ctx, exc)

    def memory_dir(self, ctx: MemoryContext) -> Path:
        name = ctx.memory_name or ctx.session_id or "default"
        return self.root / safe_component(name)

    def available_memories(self) -> tuple[str, ...]:
        root = self.root
        if not root.exists():
            return ()
        return tuple(sorted(path.name for path in root.iterdir() if path.is_dir()))

    def ensure_layout(self, memory_dir: Path) -> None:
        (memory_dir / "runs").mkdir(parents=True, exist_ok=True)
        write_if_missing(memory_dir / "INDEX.md", index_skeleton())
        write_if_missing(
            memory_dir / MEMORY_SUMMARY_FILENAME,
            memory_summary_skeleton(memory_dir.name),
        )
        write_if_missing(
            memory_dir / MEMORY_HANDBOOK_FILENAME,
            handbook_skeleton(),
        )

    def memory_overview(self, names: tuple[str, ...] | None = None) -> tuple[str, ...]:
        selected = self.available_memories() if names is None else names
        overview = []
        for name in selected:
            memory_dir = self.root / name
            summary_path = memory_dir / MEMORY_SUMMARY_FILENAME
            if summary_path.exists():
                summary = first_summary_line(summary_path.read_text(encoding="utf-8"))
            else:
                summary = ""
            overview.append(f"{name}: {summary or 'no summary yet'}")
        return tuple(overview)

    def _finish(self, ctx: MemoryContext) -> None:
        assert ctx.state is not None
        base_run_id = _run_id(ctx)
        payload_run_path = f"runs/{base_run_id}"
        messages = ctx.state.messages
        task = first_user_text(messages, fallback=ctx.task)
        transcript = render_transcript_markdown(messages)
        artifacts = normalize_artifacts(tuple(self.artifact_builder(ctx)))

        distillation = FilesystemDistillation()
        distillation_error: Exception | None = None
        if self.distiller is not None:
            try:
                memory_summary, index, handbook = self._distillation_context_files(ctx)
                payload = FilesystemMemoryPayload(
                    task=task,
                    transcript=truncate_for_distiller(
                        transcript, limit=self.distiller_transcript_limit
                    ),
                    artifacts=artifacts,
                    memory_summary=memory_summary,
                    index=index,
                    notes=handbook,
                    run_path=payload_run_path,
                    available_memories=self.available_memories(),
                    context=ctx,
                )
                distillation = coerce_distillation(self.distiller(payload))
            except Exception as exc:
                distillation_error = exc

        target_ctx = ctx
        memory_name = distillation.memory_name.strip()
        if not target_ctx.memory_name and memory_name:
            target_ctx = replace(target_ctx, memory_name=safe_component(memory_name))

        memory_dir = self.memory_dir(target_ctx)
        self.ensure_layout(memory_dir)
        run_id = unique_run_id(memory_dir, base_run_id)
        run_path = f"runs/{run_id}"
        if run_path != payload_run_path:
            distillation = retarget_distillation(
                distillation,
                old_path=payload_run_path,
                new_path=run_path,
            )
        run_dir = memory_dir / "runs" / run_id
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        write_text_atomic(run_dir / "task.md", "# Task\n\n" + task.strip() + "\n")
        write_text_atomic(run_dir / "transcript.md", transcript)
        for artifact in artifacts:
            write_text_atomic(
                artifacts_dir / safe_component(artifact.name),
                artifact.content,
            )
        write_text_atomic(run_dir / "artifacts.md", artifact_manifest(artifacts))
        if distillation_error is not None:
            write_text_atomic(
                run_dir / "memory_error.md",
                memory_error_text("Distillation failed", distillation_error),
            )

        summary_path = f"{run_path}/summary.md"
        summary = sanitize_summary(distillation.summary_md) or fallback_summary(
            task, artifacts, distillation_error
        )
        write_text_atomic(run_dir / "summary.md", summary.rstrip() + "\n")

        upsert_index_row(
            memory_dir / "INDEX.md",
            run=run_id,
            row=complete_index_row(distillation.index_row, task, artifacts),
            summary_path=summary_path,
        )

        apply_handbook_rewrite(
            memory_dir / MEMORY_HANDBOOK_FILENAME,
            distillation.memory_md,
            run_dir=run_dir,
        )
        update_memory_summary(memory_dir, distillation.memory_summary_md)

    def _distillation_context_files(
        self,
        ctx: MemoryContext,
    ) -> tuple[str, str, str]:
        if not ctx.memory_name:
            return self._available_memory_context_files()
        memory_dir = self.memory_dir(ctx)
        self.ensure_layout(memory_dir)
        return (
            (memory_dir / MEMORY_SUMMARY_FILENAME).read_text(encoding="utf-8"),
            (memory_dir / "INDEX.md").read_text(encoding="utf-8"),
            (memory_dir / MEMORY_HANDBOOK_FILENAME).read_text(encoding="utf-8"),
        )

    def _available_memory_context_files(self) -> tuple[str, str, str]:
        names = self.available_memories()
        if not names:
            return (
                memory_summary_skeleton("default"),
                index_skeleton(),
                handbook_skeleton(),
            )

        sections = (
            ["# Available Memory Summaries", ""],
            ["# Available Memory Indexes", ""],
            ["# Available Memory Handbooks", ""],
        )
        filenames = (MEMORY_SUMMARY_FILENAME, "INDEX.md", MEMORY_HANDBOOK_FILENAME)
        for name in names:
            memory_dir = self.root / name
            self.ensure_layout(memory_dir)
            for lines, filename in zip(sections, filenames, strict=True):
                lines.extend(
                    [
                        f"## {name}/{filename}",
                        "",
                        read_limited(memory_dir / filename),
                        "",
                    ]
                )
        summaries, indexes, handbooks = (
            "\n".join(lines).rstrip() + "\n" for lines in sections
        )
        return (summaries, indexes, handbooks)

    def _record_finish_error(self, ctx: MemoryContext, exc: Exception) -> None:
        try:
            memory_dir = self.memory_dir(ctx)
            self.ensure_layout(memory_dir)
            run_id = unique_run_id(memory_dir, _run_id(ctx))
            run_dir = memory_dir / "runs" / run_id
            write_text_atomic(
                run_dir / "memory_error.md",
                memory_error_text("Filesystem memory finish failed", exc),
            )
        except Exception:
            return


def _run_id(ctx: MemoryContext) -> str:
    fallback = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    return safe_component(ctx.run_id or ctx.session_id or fallback)


def _policy_block(memory_dir: Path, memory_summary: str) -> str:
    return "\n".join(
        [
            "<filesystem_memory>",
            "You have filesystem memory for this task family at this absolute path:",
            f"  {memory_dir}",
            "",
            "Read these files directly by their full absolute paths, using whatever"
            " file-reading tools you have. Do not assume a working directory or any"
            " state (variables, environment, cwd) persists between separate actions;"
            " refer to each file by its absolute path, for example:",
            f"  {memory_dir}/{MEMORY_HANDBOOK_FILENAME}",
            f"  {memory_dir}/INDEX.md",
            "",
            "Files in this memory (all under the path above):",
            f"- `{MEMORY_SUMMARY_FILENAME}` — cold-start navigation summary.",
            f"- `{MEMORY_HANDBOOK_FILENAME}` — small, curated handbook of durable, high-signal lessons (the distiller rewrites it whole after each run).",
            "- `INDEX.md` — one row per run; columns: summary, scope, signals, keywords, artifacts, path.",
            "- `runs/<run_id>/` — per-run evidence directory containing:",
            "  - `task.md` — the task that run was given.",
            "  - `transcript.md` — full message log; section headings are `## <n>. <role> (<kind>, <sender> -> <target>)`.",
            "  - `summary.md` — distilled, compact per-run summary.",
            "  - `artifacts.md` — manifest describing each saved raw artifact and why.",
            "  - `artifacts/` — raw products kept verbatim (e.g. `model_patch.diff`).",
            "",
            "Locate transcript.md evidence by searching for the cited anchor (a `## <n>.` section heading, file path, symbol, command, or error string), not raw line numbers — citation numbers are message section ids, not file lines.",
            "Keep recall lightweight: avoid broad scans unless the summary and index are insufficient.",
            "Treat memory as dated context, not current truth.",
            "If a memory fact names a file, command, test, or current repo state and verification is cheap, verify it against the current workspace before relying on it.",
            "Current user instructions, code, tests, and tool observations outrank memory.",
            "Do not modify memory files during the task unless explicitly instructed.",
            "",
            f"<{MEMORY_SUMMARY_FILENAME}_excerpt>",
            memory_summary.strip() or "(empty)",
            f"</{MEMORY_SUMMARY_FILENAME}_excerpt>",
            "</filesystem_memory>",
        ]
    )


def _root_policy_block(root: Path, overview: tuple[str, ...]) -> str:
    names = "\n".join(f"- {item}" for item in overview)
    return "\n".join(
        [
            "<filesystem_memory>",
            "Filesystem memory has prior task-family directories at this absolute path:",
            f"  {root}",
            "",
            "Read memory files directly by their full absolute paths, using whatever"
            " file-reading tools you have; do not assume a working directory or any state"
            " (variables, environment, cwd) persists between separate actions.",
            "",
            "Available memory names:",
            names,
            "",
            "If prior runs may help, inspect the most relevant directory lightly.",
            f"Each namespace directory holds `{MEMORY_SUMMARY_FILENAME}`, `{MEMORY_HANDBOOK_FILENAME}`, `INDEX.md`, and `runs/<run_id>/` (with `task.md`, `transcript.md`, `summary.md`, `artifacts.md`, and an `artifacts/` dir of raw products).",
            f"Start with {MEMORY_SUMMARY_FILENAME}; then use {MEMORY_HANDBOOK_FILENAME}, INDEX.md, and targeted run summaries.",
            "Locate transcript.md evidence by searching for a cited anchor (a `## <n>.` section heading, file path, symbol, or command), not raw line numbers; avoid broad scans unless needed.",
            "Treat memory as dated context, not current truth, and verify cheap drift-prone facts.",
            "Current user instructions, code, tests, and tool observations outrank memory.",
            "Do not modify memory files during the task unless explicitly instructed.",
            "</filesystem_memory>",
        ]
    )
