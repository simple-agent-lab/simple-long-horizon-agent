"""Filesystem memory: Markdown-file evidence and distillation for one memory namespace.

Split by concern:
  - ``core``: the ``FilesystemMemory`` class and run orchestration.
  - ``distill``: LLM distillation payload/result types, prompt, and the call.
  - ``render``: INDEX.md / MEMORY.md / summary Markdown rendering and parsing.
  - ``artifacts``: the artifact dataclass and path-safety helpers.

Everything below was previously a single ``filesystem.py`` module; this
``__init__`` re-exports the same public names so existing imports
(``from simple_agent_lab.memory.filesystem import ...``) keep working.
"""

from __future__ import annotations

from simple_agent_lab.memory.filesystem.artifacts import (
    ArtifactBuilder,
    FilesystemArtifact,
    default_artifacts,
    normalize_artifacts,
    safe_component,
    unique_component,
)
from simple_agent_lab.memory.filesystem.core import (
    DEFAULT_DISTILLER_TRANSCRIPT_LIMIT,
    DEFAULT_FILESYSTEM_MEMORY_ROOT,
    FilesystemMemory,
)
from simple_agent_lab.memory.filesystem.distill import (
    Distiller,
    FilesystemDistillation,
    FilesystemMemoryPayload,
    filesystem_distillation_prompt,
    make_filesystem_distiller,
    retarget_distillation,
)
from simple_agent_lab.memory.filesystem.render import (
    DEFAULT_MAX_HANDBOOK_CHARS,
    MEMORY_HANDBOOK_FILENAME,
    MEMORY_SUMMARY_FILENAME,
    FilesystemIndexRow,
    artifact_manifest,
    complete_index_row,
    fallback_summary,
    first_summary_line,
    keywords_from_text,
    memory_error_text,
    parse_index_rows,
    render_memory_summary_from_index,
    sanitize_memory_summary,
    sanitize_summary,
    update_memory_summary,
    upsert_index_row,
)

__all__ = [
    "ArtifactBuilder",
    "DEFAULT_DISTILLER_TRANSCRIPT_LIMIT",
    "DEFAULT_FILESYSTEM_MEMORY_ROOT",
    "DEFAULT_MAX_HANDBOOK_CHARS",
    "Distiller",
    "FilesystemArtifact",
    "FilesystemDistillation",
    "FilesystemIndexRow",
    "FilesystemMemory",
    "FilesystemMemoryPayload",
    "MEMORY_HANDBOOK_FILENAME",
    "MEMORY_SUMMARY_FILENAME",
    "artifact_manifest",
    "complete_index_row",
    "default_artifacts",
    "fallback_summary",
    "filesystem_distillation_prompt",
    "first_summary_line",
    "keywords_from_text",
    "make_filesystem_distiller",
    "memory_error_text",
    "normalize_artifacts",
    "parse_index_rows",
    "render_memory_summary_from_index",
    "retarget_distillation",
    "safe_component",
    "sanitize_memory_summary",
    "sanitize_summary",
    "unique_component",
    "update_memory_summary",
    "upsert_index_row",
]
