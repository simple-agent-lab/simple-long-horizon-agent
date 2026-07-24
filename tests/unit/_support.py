from __future__ import annotations

import importlib.util
from pathlib import Path
import struct
import sys
from types import ModuleType
from typing import Any
import zlib

from simple_agent_lab import AgentTool, ToolResult


def load_module(path: Path, name: str) -> ModuleType:
    """Load a Python file under a stable module name."""

    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def execute_tool(tool: AgentTool, args: dict[str, Any]) -> ToolResult:
    return tool.execute("call_1", args, lambda: False, None)


def make_red_png(side: int = 32) -> bytes:
    """Return a small dependency-free RGB PNG fixture."""

    raw = b"".join(b"\x00" + b"\xff\x00\x00" * side for _ in range(side))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data))
        )

    ihdr = struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
