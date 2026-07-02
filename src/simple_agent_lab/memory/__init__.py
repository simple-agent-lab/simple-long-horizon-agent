"""Memory for Simple Agent Lab.

The memory package stays outside the core runtime. Memory implementations can
inject ordinary messages, recall context, provide ordinary tools, and observe
completed turns or final State.
"""

from .base import (
    Memory,
    MemoryBinding,
    MemoryContext,
    NoMemory,
    memory_context_message,
)
from .distill import make_filesystem_distiller
from .filesystem import FilesystemMemory
from .records import (
    DEFAULT_FILESYSTEM_MEMORY_ROOT,
    FilesystemArtifact,
    FilesystemDistillation,
    FilesystemIndexRow,
    FilesystemMemoryPayload,
)

__all__ = [
    "FilesystemArtifact",
    "DEFAULT_FILESYSTEM_MEMORY_ROOT",
    "FilesystemDistillation",
    "FilesystemIndexRow",
    "FilesystemMemory",
    "FilesystemMemoryPayload",
    "Memory",
    "MemoryBinding",
    "MemoryContext",
    "NoMemory",
    "make_filesystem_distiller",
    "memory_context_message",
]
