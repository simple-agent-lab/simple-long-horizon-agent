"""GDPVal solver tools implemented on Simple Agent Lab's tool boundary."""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import subprocess
from pathlib import Path
from typing import Any

from simple_agent_lab.messages import ImageBlock, TextBlock
from simple_agent_lab.tools import AgentTool, ToolResult, text_result
from simple_agent_lab.tools.bash import (
    DEFAULT_BASH_MAX_OUTPUT_CHARS,
    bash_execution_to_tool_result,
    run_bash,
)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
_MAX_IMAGE_BYTES = 5 * 1024 * 1024
_MAX_READ_CHARS = 80_000


def make_gdpval_tools(
    *,
    workdir: str | Path,
    reference_dir: str | Path,
    output_head_chars: int = 6000,
    output_tail_chars: int = 6000,
) -> tuple[AgentTool, ...]:
    """Return the no-web GDPVal tool surface."""

    workspace = Path(workdir).resolve()
    references = Path(reference_dir).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    references.mkdir(parents=True, exist_ok=True)
    todos: list[dict[str, Any]] = []

    def list_dir(call_id: str, args: dict[str, Any], abort, on_update) -> ToolResult:
        del call_id, abort, on_update
        try:
            root = _resolve_read_path(
                args.get("path") or str(workspace), workspace, references
            )
            recursive = bool(args.get("recursive", False))
            limit = max(1, min(int(args.get("limit", 200) or 200), 1000))
            lines: list[str] = []
            iterator = root.rglob("*") if recursive else root.iterdir()
            for index, item in enumerate(sorted(iterator, key=lambda p: str(p))):
                if index >= limit:
                    lines.append(f"... truncated after {limit} entries")
                    break
                rel = item.relative_to(root)
                suffix = "/" if item.is_dir() else ""
                size = "" if item.is_dir() else f" {item.stat().st_size} bytes"
                lines.append(f"{rel}{suffix}{size}")
            return text_result("\n".join(lines) or "(empty directory)")
        except Exception as exc:
            return text_result(f"{type(exc).__name__}: {exc}", is_error=True)

    def read_file(call_id: str, args: dict[str, Any], abort, on_update) -> ToolResult:
        del call_id, abort, on_update
        try:
            path = _resolve_read_path(args.get("path"), workspace, references)
            if path.suffix.lower() in _IMAGE_EXTS:
                return _image_result(path)
            offset = max(1, int(args.get("offset", 1) or 1))
            limit = args.get("limit")
            line_limit = max(1, int(limit)) if limit is not None else None
            text = path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            selected = lines[
                offset - 1 : offset - 1 + line_limit if line_limit else None
            ]
            rendered = "\n".join(
                f"{line_no:>6}: {line}"
                for line_no, line in enumerate(selected, start=offset)
            )
            if len(rendered) > _MAX_READ_CHARS:
                rendered = _middle_truncate(
                    rendered,
                    head_chars=_MAX_READ_CHARS // 2,
                    tail_chars=_MAX_READ_CHARS // 2,
                )
            meta = (
                f"\n\n[file: {path} | lines={len(lines)} | "
                f"shown_from={offset} | shown_count={len(selected)}]"
            )
            return text_result(rendered + meta)
        except Exception as exc:
            return text_result(f"{type(exc).__name__}: {exc}", is_error=True)

    def grep_files(call_id: str, args: dict[str, Any], abort, on_update) -> ToolResult:
        del call_id, abort, on_update
        try:
            pattern = str(args.get("pattern") or "")
            if not pattern:
                return text_result("pattern is required", is_error=True)
            root = _resolve_read_path(
                args.get("path") or str(workspace), workspace, references
            )
            max_matches = max(1, min(int(args.get("max_matches", 100) or 100), 1000))
            command = [
                "rg",
                "--line-number",
                "--no-heading",
                "--color",
                "never",
                "-m",
                str(max_matches),
                pattern,
                str(root),
            ]
            if not bool(args.get("case_sensitive", True)):
                command.insert(1, "-i")
            proc = subprocess.run(
                command,
                cwd=str(workspace),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
            if proc.returncode == 1:
                return text_result("No matches found")
            if proc.returncode != 0:
                return text_result(proc.stderr.strip() or "rg failed", is_error=True)
            return text_result(
                _middle_truncate(
                    proc.stdout.strip(),
                    head_chars=output_head_chars,
                    tail_chars=output_tail_chars,
                )
            )
        except Exception as exc:
            return text_result(f"{type(exc).__name__}: {exc}", is_error=True)

    def write_file(call_id: str, args: dict[str, Any], abort, on_update) -> ToolResult:
        del call_id, abort, on_update
        try:
            path = _resolve_write_path(args.get("path"), workspace)
            content = str(args.get("content") or "")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return text_result(f"Wrote {path} ({len(content)} chars)")
        except Exception as exc:
            return text_result(f"{type(exc).__name__}: {exc}", is_error=True)

    def edit_file(call_id: str, args: dict[str, Any], abort, on_update) -> ToolResult:
        del call_id, abort, on_update
        try:
            path = _resolve_write_path(args.get("path"), workspace)
            old = str(args.get("old") or "")
            new = str(args.get("new") or "")
            if not old:
                return text_result("old is required", is_error=True)
            count = int(args.get("count", 1) or 1)
            text = path.read_text(encoding="utf-8", errors="replace")
            occurrences = text.count(old)
            if occurrences == 0:
                return text_result(f"old text not found in {path}", is_error=True)
            updated = text.replace(old, new, count)
            path.write_text(updated, encoding="utf-8")
            return text_result(
                f"Edited {path}; replaced {min(count, occurrences)} occurrence(s)"
            )
        except Exception as exc:
            return text_result(f"{type(exc).__name__}: {exc}", is_error=True)

    def execute_bash(
        call_id: str, args: dict[str, Any], abort, on_update
    ) -> ToolResult:
        del call_id, on_update
        if abort():
            return text_result("Command aborted before start", is_error=True)
        command = str(args.get("command") or "").strip()
        if not command:
            return text_result("command is required", is_error=True)
        try:
            timeout = float(args.get("timeout_seconds", 30) or 30)
            execution = run_bash(
                command,
                cwd=workspace,
                timeout_seconds=max(1.0, min(timeout, 120.0)),
                max_output_chars=max(
                    DEFAULT_BASH_MAX_OUTPUT_CHARS,
                    output_head_chars + output_tail_chars,
                ),
            )
            result = bash_execution_to_tool_result(execution)
            text = "\n".join(
                block.text for block in result.content if isinstance(block, TextBlock)
            )
            return ToolResult(
                content=(
                    TextBlock(
                        _middle_truncate(
                            text,
                            head_chars=output_head_chars,
                            tail_chars=output_tail_chars,
                        )
                    ),
                ),
                details=result.details,
                is_error=result.is_error,
            )
        except Exception as exc:
            return text_result(f"{type(exc).__name__}: {exc}", is_error=True)

    def todo_write(call_id: str, args: dict[str, Any], abort, on_update) -> ToolResult:
        del call_id, abort, on_update
        nonlocal todos
        raw = args.get("todos")
        if not isinstance(raw, list):
            return text_result("todos must be a list", is_error=True)
        todos = [dict(item) for item in raw if isinstance(item, dict)]
        payload = json.dumps({"todos": todos}, ensure_ascii=False, indent=2)
        (workspace / "_sal_todos.json").write_text(payload + "\n", encoding="utf-8")
        return text_result(payload)

    return (
        AgentTool(
            name="list_dir",
            description="List files under WORKDIR or REFERENCE_DIR.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "recursive": {"type": "boolean"},
                    "limit": {"type": "integer"},
                },
                "additionalProperties": False,
            },
            execute=list_dir,
        ),
        AgentTool(
            name="read_file",
            description="Read a text file with line numbers, or return an image block for image files.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            execute=read_file,
        ),
        AgentTool(
            name="grep_files",
            description="Search text files under WORKDIR or REFERENCE_DIR with ripgrep.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "case_sensitive": {"type": "boolean"},
                    "max_matches": {"type": "integer"},
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
            execute=grep_files,
        ),
        AgentTool(
            name="write_file",
            description="Create or rewrite a UTF-8 text file under WORKDIR.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            execute=write_file,
            execution_mode="sequential",
        ),
        AgentTool(
            name="edit_file",
            description="Replace exact text in a UTF-8 file under WORKDIR.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["path", "old", "new"],
                "additionalProperties": False,
            },
            execute=edit_file,
            execution_mode="sequential",
        ),
        AgentTool(
            name="execute_bash",
            description="Run a bash command in WORKDIR and return stdout/stderr.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_seconds": {"type": "number"},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            execute=execute_bash,
            execution_mode="sequential",
            timeout_seconds=125.0,
        ),
        AgentTool(
            name="TodoWrite",
            description="Replace the current task todo list.",
            parameters={
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {"type": "object"},
                    },
                },
                "required": ["todos"],
                "additionalProperties": False,
            },
            execute=todo_write,
            execution_mode="sequential",
        ),
    )


def _resolve_read_path(value: Any, workspace: Path, references: Path) -> Path:
    if value is None or str(value).strip() == "":
        return workspace
    path = Path(str(value))
    candidate = path.resolve() if path.is_absolute() else (workspace / path).resolve()
    if _inside(candidate, workspace) or _inside(candidate, references):
        return candidate
    alt = (references / path).resolve()
    if _inside(alt, references):
        return alt
    raise ValueError(f"path must be inside WORKDIR or REFERENCE_DIR: {value!r}")


def _resolve_write_path(value: Any, workspace: Path) -> Path:
    if value is None or str(value).strip() == "":
        raise ValueError("path is required")
    path = Path(str(value))
    candidate = path.resolve() if path.is_absolute() else (workspace / path).resolve()
    if not _inside(candidate, workspace):
        raise ValueError(f"write path must be inside WORKDIR: {value!r}")
    return candidate


def _inside(candidate: Path, root: Path) -> bool:
    return candidate == root or root in candidate.parents


def _middle_truncate(text: str, *, head_chars: int, tail_chars: int) -> str:
    text = text or ""
    budget = max(0, head_chars) + max(0, tail_chars)
    if budget <= 0 or len(text) <= budget:
        return text
    omitted = len(text) - budget
    tail = text[-tail_chars:] if tail_chars > 0 else ""
    return f"{text[:head_chars]}\n...[truncated {omitted} chars]...\n{tail}"


def _image_result(path: Path) -> ToolResult:
    data = path.read_bytes()
    if len(data) > _MAX_IMAGE_BYTES:
        return text_result(
            f"Image {path} is too large to attach ({len(data)} bytes)",
            is_error=True,
        )
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    digest = hashlib.sha256(data).hexdigest()
    note = f"Image file: {path}\nsize_bytes: {len(data)}\nsha256: {digest}"
    return ToolResult(
        content=(
            TextBlock(note),
            ImageBlock(data=base64.b64encode(data).decode("ascii"), mime_type=mime),
        )
    )
