"""Shared records and limits for filesystem memory.

Leaf module: the store, distillation, and memory layers all build on these
types, so this module imports nothing from its siblings.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Callable

from simple_agent_lab.memory.base import MemoryContext

DEFAULT_FILESYSTEM_MEMORY_ROOT = "~/.simple/memory"
MEMORY_SUMMARY_FILENAME = "memory_summary.md"
MEMORY_HANDBOOK_FILENAME = "MEMORY.md"

# Default character cap on transcript text passed to the distiller (the full
# transcript still lands in ``transcript.md``). Truncated transcripts distill
# poorly, so the default is large and acts as an extreme-run guard, not a routine
# trim: ~500k characters is roughly 125k English tokens (more for CJK), leaving
# room for the rest of the prompt and the output inside a modern 200k+ token
# window. Override per instance via ``FilesystemMemory(distiller_transcript_limit=...)``.
DEFAULT_DISTILLER_TRANSCRIPT_LIMIT = 500_000

# Hard upper bound on the rewritten MEMORY.md handbook. The distiller returns the
# full updated handbook (the model owns merge/delete/rewrite), so a runaway or
# truncated response must not be persisted verbatim. A proposed rewrite larger than
# this is rejected and the previous handbook is kept untouched. ~20k characters is a
# generous ceiling for a small, curated lessons file while still catching unbounded
# growth or a model that dumped the whole transcript back into memory.
DEFAULT_MAX_HANDBOOK_CHARS = 20_000


@dataclass(frozen=True)
class FilesystemArtifact:
    """One durable run artifact to store under ``artifacts/``."""

    name: str
    content: str
    description: str = ""


@dataclass(frozen=True)
class FilesystemMemoryPayload:
    """Inputs passed to a filesystem-memory distiller."""

    task: str
    transcript: str
    artifacts: tuple[FilesystemArtifact, ...]
    index: str
    notes: str
    run_path: str
    available_memories: tuple[str, ...]
    context: MemoryContext
    memory_summary: str = ""


@dataclass(frozen=True)
class FilesystemIndexRow:
    summary: str = ""
    scope: str = ""
    signals: str = ""
    keywords: str = ""
    artifacts: str = ""


@dataclass(frozen=True)
class FilesystemDistillation:
    memory_name: str = ""
    memory_summary_md: str = ""
    summary_md: str = ""
    index_row: FilesystemIndexRow = FilesystemIndexRow()
    memory_md: str = ""


Distiller = Callable[
    [FilesystemMemoryPayload],
    FilesystemDistillation | Mapping[str, Any],
]
ArtifactBuilder = Callable[[MemoryContext], Iterable[FilesystemArtifact]]
