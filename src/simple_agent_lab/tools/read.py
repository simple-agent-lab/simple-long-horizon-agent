"""Small file-read tool.

Skills load by *reading*: the model opens a ``SKILL.md`` (and its
``references/*`` files) to follow a workflow, then runs ``scripts/*`` via
bash. The bash tool truncates output at 4000 chars, which would mangle a
real ``SKILL.md`` mid-instruction, so reading skill content goes through
this dedicated tool with a more generous budget instead.

This is intentionally a plain reader: one UTF-8 text file, an optional
1-based line range, no globbing, no writes, no shell. It mirrors the shape
of ``tools/bash.py`` (a ``make_*_tool`` factory returning an ``AgentTool``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from simple_agent_lab.messages import TextBlock

from . import AbortFlag, AgentTool, ToolResult, ToolUpdateFn, text_result


READ_TOOL_NAME = "read"
DEFAULT_READ_MAX_CHARS = 20000
DEFAULT_READ_MAX_LINES = 2000
DEFAULT_READ_DIR_MAX_ENTRIES = 200
DEFAULT_READ_DIR_MAX_DEPTH = 2


def make_read_tool(
    *,
    cwd: str | Path | None = None,
    max_output_chars: int = DEFAULT_READ_MAX_CHARS,
    max_lines: int = DEFAULT_READ_MAX_LINES,
) -> AgentTool:
    """Return an ``AgentTool`` that reads one UTF-8 text file."""

    if max_output_chars <= 0:
        raise ValueError("max_output_chars must be > 0")
    if max_lines <= 0:
        raise ValueError("max_lines must be > 0")

    root = Path(cwd or ".").resolve()

    def execute(
        call_id: str,
        args: dict[str, Any],
        abort: AbortFlag,
        on_update: ToolUpdateFn | None,
    ) -> ToolResult:
        del call_id, on_update
        if abort():
            return text_result("Read aborted before start.", is_error=True)

        raw_path = str(args.get("path", "")).strip()
        if not raw_path:
            return text_result("Missing required read argument: path.", is_error=True)

        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        try:
            path = path.resolve()
        except OSError as exc:
            return text_result(
                f"Could not resolve path {raw_path!r}: {exc}", is_error=True
            )
        if path.is_dir():
            listing = _render_directory(
                path,
                max_entries=DEFAULT_READ_DIR_MAX_ENTRIES,
                max_depth=DEFAULT_READ_DIR_MAX_DEPTH,
            )
            return ToolResult(
                content=(TextBlock(listing),),
                details={"path": str(path), "kind": "directory"},
            )
        if not path.is_file():
            return text_result(f"No such file or directory: {raw_path}", is_error=True)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return text_result(f"Could not read {raw_path!r}: {exc}", is_error=True)

        offset = _coerce_positive_int(args.get("offset"), default=1)
        limit = min(
            _coerce_positive_int(args.get("limit"), default=max_lines), max_lines
        )
        rendered, truncated = _render(
            text, offset=offset, limit=limit, max_chars=max_output_chars
        )
        details = {
            "path": str(path),
            "offset": offset,
            "limit": limit,
            "truncated": truncated,
        }
        return ToolResult(content=(TextBlock(rendered),), details=details)

    return AgentTool(
        name=READ_TOOL_NAME,
        description=(
            "Read a UTF-8 text file from the workspace and return its contents. "
            "Use this to open a skill's SKILL.md, references/* docs, or schema "
            "files; it does not truncate as aggressively as bash. If `path` is a "
            "directory, returns a listing of the files inside it (use this to "
            "see a skill's scripts/, references/, and schemas before loading "
            "them). Optional `offset` (1-based first line) and `limit` (max "
            "lines) read a slice of a large file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "File path. Relative paths resolve against the workspace "
                        "cwd; skill files resolve under the skill's directory."
                    ),
                },
                "offset": {
                    "type": "integer",
                    "description": "1-based first line to read (default 1).",
                },
                "limit": {
                    "type": "integer",
                    "description": f"Max lines to read (default {max_lines}).",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        execute=execute,
        execution_mode="parallel",
    )


def _coerce_positive_int(value: Any, *, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 1 else default


def _render(text: str, *, offset: int, limit: int, max_chars: int) -> tuple[str, bool]:
    lines = text.splitlines()
    start = offset - 1
    if start >= len(lines):
        return (
            f"(file has {len(lines)} lines; offset {offset} is past the end)",
            False,
        )
    selected = lines[start : start + limit]
    body = "\n".join(selected)
    range_truncated = start > 0 or (start + limit) < len(lines)
    if len(body) > max_chars:
        clipped = body[:max_chars]
        return f"{clipped}\n... [truncated to {max_chars} chars] ...", True
    return body, range_truncated


def _render_directory(path: Path, *, max_entries: int, max_depth: int) -> str:
    """Shallow, sorted listing of a directory's files. Skips dotfiles and
    caps depth/entries so the model can see a skill's layout cheaply."""

    entries: list[str] = []
    for child in sorted(path.rglob("*")):
        rel = child.relative_to(path)
        if len(rel.parts) > max_depth:
            continue
        if any(part.startswith(".") for part in rel.parts):
            continue
        suffix = "/" if child.is_dir() else ""
        entries.append(f"{rel.as_posix()}{suffix}")
        if len(entries) >= max_entries:
            entries.append(f"... [truncated to {max_entries} entries] ...")
            break
    if not entries:
        return f"(empty directory: {path})"
    return f"Files under {path}:\n" + "\n".join(entries)
