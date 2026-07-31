"""Small file-read tool for local demos.

A compact port of the `read` tool from earendil-works/pi. It reads a single
file from the local workspace and returns model-visible content:

* Text files are returned verbatim, with two independent truncation limits —
  a line cap and a byte cap, whichever is hit first. When the view is cut
  short the result carries an actionable continuation notice (`offset=N`) so
  the model can keep reading instead of guessing.
* Images (PNG, JPEG, GIF, WebP) are inlined as `ImageBlock`s so a
  vision-capable model can see them, paired with a short text note.
* Directories return a shallow listing instead of an error. Skills load by
  *reading*: the model opens a directory to see which `scripts/`,
  `references/`, and schema files a skill bundles, then reads the specific
  files it needs. The generous text budget keeps a real `SKILL.md` readable
  without the aggressive truncation the bash tool applies.

This mirrors `bash.py`: a `make_read_tool(...)` factory returns an `AgentTool`,
the structured truncation accounting lives in a frozen dataclass, and the
helpers stay pure so they are easy to unit test.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from simple_long_horizon_agent.messages import ImageBlock, TextBlock

from . import (
    AbortFlag,
    AgentTool,
    ToolExecutionMode,
    ToolResult,
    ToolResultContent,
    ToolUpdateFn,
    coerce_int,
    text_result,
)


READ_TOOL_NAME = "read"

# Two independent truncation limits — whichever is hit first wins. Mirrors the
# upstream defaults: a generous line budget plus a byte cap so a few very long
# lines cannot blow the model-visible context.
DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024  # 50 KiB
DEFAULT_MAX_ATTACH_BYTES = 5 * 1024 * 1024  # 5 MiB per inlined image

# Directory listings stay shallow so a skill's layout is cheap to scan without
# dumping a deep tree into context.
DEFAULT_READ_DIR_MAX_ENTRIES = 200
DEFAULT_READ_DIR_MAX_DEPTH = 2

_IMAGE_MIME_BY_SUFFIX: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


@dataclass(frozen=True)
class TruncationResult:
    """Structured accounting for one head-truncation pass.

    `truncated_by` records which limit fired (`"lines"`, `"bytes"`, or `None`
    when the whole selection fit). `first_line_exceeds_limit` flags the case
    where line one alone is over the byte cap, so no complete line fits and the
    caller should fall back to a byte-bounded shell read.
    """

    content: str
    truncated: bool
    truncated_by: Literal["lines", "bytes"] | None
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    first_line_exceeds_limit: bool
    max_lines: int
    max_bytes: int


def make_read_tool(
    *,
    cwd: str | Path | None = None,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_attach_bytes: int = DEFAULT_MAX_ATTACH_BYTES,
    execution_mode: ToolExecutionMode = "parallel",
) -> AgentTool:
    """Return an `AgentTool` that reads one file from the local workspace."""

    if max_lines <= 0:
        raise ValueError("max_lines must be > 0")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be > 0")
    if max_attach_bytes <= 0:
        raise ValueError("max_attach_bytes must be > 0")

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

        try:
            offset = _coerce_line_arg("offset", args.get("offset"))
            limit = _coerce_line_arg("limit", args.get("limit"))
        except ValueError as exc:
            return text_result(f"Invalid read argument: {exc}", is_error=True)

        return read_file(
            raw_path,
            root=root,
            offset=offset,
            limit=limit,
            max_lines=max_lines,
            max_bytes=max_bytes,
            max_attach_bytes=max_attach_bytes,
        )

    return AgentTool(
        name=READ_TOOL_NAME,
        description=(
            "Read the contents of a file. Supports text files and images "
            "(jpg, png, gif, webp); images are returned as inline attachments. "
            f"For text files, output is truncated to {max_lines} lines or "
            f"{max_bytes // 1024}KB (whichever is hit first). Use offset/limit "
            "for large files; when you need the full file, continue with "
            "offset until complete. If `path` is a directory, returns a listing "
            "of the files inside it (use this to see a skill's scripts/ and "
            "references/ before loading them). Prefer this over `cat`/`sed`."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read (relative or absolute).",
                },
                "offset": {
                    "type": "number",
                    "description": "Line number to start reading from (1-indexed).",
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of lines to read.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        execute=execute,
        execution_mode=execution_mode,
    )


def read_file(
    raw_path: str,
    *,
    root: str | Path | None = None,
    offset: int | None = None,
    limit: int | None = None,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_attach_bytes: int = DEFAULT_MAX_ATTACH_BYTES,
) -> ToolResult:
    """Read `raw_path` and return a model-visible `ToolResult`.

    Resolves relative paths against `root`, lists directories so a skill's
    layout is visible, splits text on the image/text boundary by file
    extension, and applies head truncation to text reads.
    """

    base = Path(root or ".").resolve()
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = base / candidate
    try:
        path = candidate.resolve()
    except OSError as exc:
        return text_result(f"Could not resolve path {raw_path!r}: {exc}", is_error=True)

    if not path.exists():
        return text_result(f"No such file or directory: {raw_path}", is_error=True)
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

    mime = _IMAGE_MIME_BY_SUFFIX.get(path.suffix.lower())
    if mime is not None:
        return _read_image(path, raw_path, mime, max_attach_bytes=max_attach_bytes)

    try:
        data = path.read_bytes()
    except OSError as exc:
        return text_result(f"Failed to read {raw_path}: {exc}", is_error=True)

    return _read_text(
        data,
        raw_path=raw_path,
        offset=offset,
        limit=limit,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def truncate_head(
    content: str,
    *,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TruncationResult:
    """Keep the first whole lines that fit under both the line and byte caps.

    Never returns a partial line. If line one alone exceeds the byte cap the
    result is empty with `first_line_exceeds_limit=True`.
    """

    total_bytes = _byte_len(content)
    lines = _split_lines_for_counting(content)
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            first_line_exceeds_limit=False,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    if lines and _byte_len(lines[0]) > max_bytes:
        return TruncationResult(
            content="",
            truncated=True,
            truncated_by="bytes",
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=0,
            output_bytes=0,
            first_line_exceeds_limit=True,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    kept: list[str] = []
    used_bytes = 0
    truncated_by: Literal["lines", "bytes"] = "lines"
    for index, line in enumerate(lines[:max_lines]):
        line_bytes = _byte_len(line) + (1 if index > 0 else 0)  # +1 for the newline
        if used_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        kept.append(line)
        used_bytes += line_bytes

    output = "\n".join(kept)
    return TruncationResult(
        content=output,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(kept),
        output_bytes=_byte_len(output),
        first_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def format_size(num_bytes: int) -> str:
    """Render a byte count as a compact human-readable size."""

    if num_bytes < 1024:
        return f"{num_bytes}B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f}KB"
    return f"{num_bytes / (1024 * 1024):.1f}MB"


def _read_text(
    data: bytes,
    *,
    raw_path: str,
    offset: int | None,
    limit: int | None,
    max_lines: int,
    max_bytes: int,
) -> ToolResult:
    text = data.decode("utf-8", errors="replace")
    # Count and slice on the same line model as truncate_head: a trailing
    # newline terminates the last line, it is not an extra empty one. `or [""]`
    # keeps an empty file readable (one empty line) instead of erroring.
    all_lines = _split_lines_for_counting(text) or [""]
    total_file_lines = len(all_lines)

    # Offset is 1-indexed on the wire; convert to a 0-indexed slice start.
    start = max(0, offset - 1) if offset else 0
    if start >= len(all_lines):
        return text_result(
            f"Offset {offset} is beyond end of file ({total_file_lines} lines total).",
            is_error=True,
        )
    start_display = start + 1

    user_limited_lines: int | None = None
    if limit is not None:
        end = min(start + limit, len(all_lines))
        selected = "\n".join(all_lines[start:end])
        user_limited_lines = end - start
    else:
        selected = "\n".join(all_lines[start:])

    truncation = truncate_head(selected, max_lines=max_lines, max_bytes=max_bytes)
    details: dict[str, Any] = {
        "path": raw_path,
        "total_lines": total_file_lines,
        "start_line": start_display,
    }

    if truncation.first_line_exceeds_limit:
        first_line_size = format_size(_byte_len(all_lines[start]))
        output = (
            f"[Line {start_display} is {first_line_size}, exceeds "
            f"{format_size(max_bytes)} limit. Use bash: "
            f"sed -n '{start_display}p' {raw_path} | head -c {max_bytes}]"
        )
        details["truncation"] = asdict(truncation)
        return text_result(output, details=details)

    if truncation.truncated:
        end_display = start_display + truncation.output_lines - 1
        next_offset = end_display + 1
        if truncation.truncated_by == "lines":
            tail = (
                f"\n\n[Showing lines {start_display}-{end_display} of "
                f"{total_file_lines}. Use offset={next_offset} to continue.]"
            )
        else:
            tail = (
                f"\n\n[Showing lines {start_display}-{end_display} of "
                f"{total_file_lines} ({format_size(max_bytes)} limit). "
                f"Use offset={next_offset} to continue.]"
            )
        details["truncation"] = asdict(truncation)
        return text_result(truncation.content + tail, details=details)

    if user_limited_lines is not None and start + user_limited_lines < len(all_lines):
        remaining = len(all_lines) - (start + user_limited_lines)
        next_offset = start + user_limited_lines + 1
        output = (
            f"{truncation.content}\n\n[{remaining} more lines in file. "
            f"Use offset={next_offset} to continue.]"
        )
        return text_result(output, details=details)

    return text_result(truncation.content, details=details)


def _read_image(
    path: Path,
    raw_path: str,
    mime: str,
    *,
    max_attach_bytes: int,
) -> ToolResult:
    size = path.stat().st_size
    if size > max_attach_bytes:
        return text_result(
            f"Read image file [{mime}]\n"
            f"[Image omitted: {format_size(size)} exceeds the "
            f"{format_size(max_attach_bytes)} inline limit.]",
            details={"path": raw_path, "mime_type": mime, "size_bytes": size},
        )
    try:
        data = path.read_bytes()
    except OSError as exc:
        return text_result(f"Failed to read image {raw_path}: {exc}", is_error=True)

    content: ToolResultContent = (
        TextBlock(text=f"Read image file [{mime}]"),
        ImageBlock(data=base64.b64encode(data).decode("ascii"), mime_type=mime),
    )
    return ToolResult(
        content=content,
        details={"path": raw_path, "mime_type": mime, "size_bytes": size},
    )


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


def _split_lines_for_counting(content: str) -> list[str]:
    """Split into lines for counting, dropping a single trailing newline.

    A trailing newline marks the end of the last line, not an extra empty one,
    so `"a\\nb\\n"` counts as two lines — matching the upstream accounting.
    """

    if not content:
        return []
    lines = content.split("\n")
    if content.endswith("\n"):
        lines.pop()
    return lines


def _byte_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _coerce_line_arg(name: str, value: Any) -> int | None:
    """Coerce an optional 1-indexed line argument to a positive int or None."""

    if value is None or value == "":
        return None
    return coerce_int(name, value, minimum=1)
