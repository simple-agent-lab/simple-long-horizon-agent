"""Memory extensions for Simple Agent Lab.

The memory package stays outside the core runtime. Memory implementations can
inject ordinary messages, recall context, provide ordinary tools, and observe
completed turns or final State.
"""

from .base import (
    AfterRunHook,
    AfterTurnHook,
    BeforeModelRequestHook,
    BeforeRunHook,
    Memory,
    MemoryBinding,
    MemoryContext,
    MemoryHooks,
    NoMemory,
    memory_context_message,
)
from .filesystem import (
    DEFAULT_FILESYSTEM_MEMORY_ROOT,
    FilesystemArtifact,
    FilesystemDistillation,
    FilesystemIndexRow,
    FilesystemMemory,
    FilesystemMemoryPayload,
    make_filesystem_distiller,
)
from .notes import NotesMemory

__all__ = [
    "FilesystemArtifact",
    "DEFAULT_FILESYSTEM_MEMORY_ROOT",
    "FilesystemDistillation",
    "FilesystemIndexRow",
    "FilesystemMemory",
    "FilesystemMemoryPayload",
    "AfterRunHook",
    "AfterTurnHook",
    "BeforeModelRequestHook",
    "BeforeRunHook",
    "Memory",
    "MemoryBinding",
    "MemoryContext",
    "MemoryHooks",
    "NoMemory",
    "NotesMemory",
    "make_filesystem_distiller",
    "memory_context_message",
]
