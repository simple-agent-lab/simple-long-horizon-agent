"""GDPVal solver tools implemented on Simple Agent Lab's tool boundary."""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
from difflib import unified_diff
from pathlib import Path
from typing import Any, cast

from simple_agent_lab.messages import ImageBlock, TextBlock
from simple_agent_lab.tools import AgentTool, ToolResult, text_result
from simple_agent_lab.tools.bash import (
    DEFAULT_BASH_MAX_OUTPUT_CHARS,
    bash_execution_to_tool_result,
    run_bash,
)

from .web_tools import make_gdpval_web_tools

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
_MAX_IMAGE_BYTES = 5 * 1024 * 1024
_MULTI_EDIT_DIFF_MAX_CHARS = 12_000
_MULTI_EDIT_DIFF_MAX_LINES = 120


def make_gdpval_tools(
    *,
    workdir: str | Path,
    reference_dir: str | Path,
    enable_web_tools: bool = True,
    output_head_chars: int = 6000,
    output_tail_chars: int = 6000,
) -> tuple[AgentTool, ...]:
    """Return the GDPVal solver tool surface.

    This matches the swalm ``tool-call-optimized-bash-fileops`` solver shape:
    bash is the primary file-operation tool, with exact multi-edit, image
    inspection, and todos. Judge tools are intentionally built separately.
    """

    workspace = Path(workdir).resolve()
    references = Path(reference_dir).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    references.mkdir(parents=True, exist_ok=True)
    todos: list[dict[str, Any]] = []

    def multi_edit_file(
        call_id: str, args: dict[str, Any], abort, on_update
    ) -> ToolResult:
        del call_id, abort, on_update
        try:
            path = _resolve_write_path(args.get("file_path"), workspace)
            raw_edits = args.get("edits")
            if not isinstance(raw_edits, list) or not raw_edits:
                return text_result("edits must be a non-empty list", is_error=True)

            before = path.read_text(encoding="utf-8", errors="replace")
            after = before
            summaries: list[str] = []
            for index, raw_edit in enumerate(raw_edits, start=1):
                if not isinstance(raw_edit, dict):
                    return text_result(f"edit {index} must be an object", is_error=True)
                # ty narrows runtime dict checks to dict[Never, Never].
                edit = cast("dict[str, Any]", raw_edit)
                old = str(edit.get("old_string") or "")
                new = str(edit.get("new_string") or "")
                replace_all = bool(edit.get("replace_all", False))
                if old == "":
                    return text_result(
                        f"edit {index} old_string is required", is_error=True
                    )
                occurrences = after.count(old)
                if occurrences == 0:
                    return text_result(
                        f"edit {index} old_string not found in {path}",
                        is_error=True,
                    )
                if not replace_all and occurrences != 1:
                    return text_result(
                        f"edit {index} old_string matched {occurrences} times; "
                        "set replace_all=true or provide a unique old_string",
                        is_error=True,
                    )
                replacements = occurrences if replace_all else 1
                after = after.replace(old, new, replacements)
                summaries.append(
                    f"- edit {index}: replaced {replacements} occurrence(s)"
                )

            if after == before:
                return text_result("No changes made")

            path.write_text(after, encoding="utf-8")
            diff = _format_unified_diff(before, after, path)
            body = "\n".join([f"Edited {path}", *summaries, "", diff]).strip()
            return text_result(body)
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

    def view_image(call_id: str, args: dict[str, Any], abort, on_update) -> ToolResult:
        del call_id, abort, on_update
        try:
            path = _resolve_read_path(args.get("path"), workspace, references)
            if path.suffix.lower() not in _IMAGE_EXTS:
                return text_result(
                    f"view_image only supports: {', '.join(sorted(_IMAGE_EXTS))}",
                    is_error=True,
                )
            return _image_result(path)
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

    execute_bash_tool = AgentTool(
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
    )
    todo_write_tool = AgentTool(
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
    )
    multi_edit_file_tool = AgentTool(
        name="multi_edit_file",
        description=(
            "Perform multiple exact string replacements in one text file. "
            "Read the file first so every old_string matches exact content. "
            "All edits are validated before writing; if any edit fails, the "
            "file is not changed. On success, returns a concise change summary "
            "and unified diff preview."
        ),
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the text file to edit.",
                },
                "edits": {
                    "type": "array",
                    "description": "Ordered exact string replacements to apply.",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_string": {
                                "type": "string",
                                "description": "Exact text to replace.",
                            },
                            "new_string": {
                                "type": "string",
                                "description": "Replacement text.",
                            },
                            "replace_all": {
                                "type": "boolean",
                                "description": "Replace every match instead of requiring one unique match.",
                                "default": False,
                            },
                        },
                        "required": ["old_string", "new_string"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["file_path", "edits"],
            "additionalProperties": False,
        },
        execute=multi_edit_file,
        execution_mode="sequential",
    )
    view_image_tool = AgentTool(
        name="view_image",
        description=(
            "Attach a local image file for direct visual inspection in the "
            "next model turn. Use this for PNG, JPG/JPEG, WEBP, GIF, or BMP "
            "files in the sandbox."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path or WORKDIR-relative path to a local image file.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        execute=view_image,
    )

    tools = (
        execute_bash_tool,
        todo_write_tool,
        multi_edit_file_tool,
        view_image_tool,
    )
    if enable_web_tools:
        tools = (*tools, *make_gdpval_web_tools(workdir=workspace))
    return tools


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


def _format_unified_diff(before: str, after: str, path: Path) -> str:
    diff_lines = list(
        unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"{path} before",
            tofile=f"{path} after",
        )
    )
    if len(diff_lines) > _MULTI_EDIT_DIFF_MAX_LINES:
        omitted = len(diff_lines) - _MULTI_EDIT_DIFF_MAX_LINES
        diff_lines = diff_lines[:_MULTI_EDIT_DIFF_MAX_LINES]
        diff_lines.append(f"... diff truncated after {omitted} more line(s)\n")
    diff = "".join(diff_lines)
    if len(diff) > _MULTI_EDIT_DIFF_MAX_CHARS:
        diff = _middle_truncate(
            diff,
            head_chars=_MULTI_EDIT_DIFF_MAX_CHARS // 2,
            tail_chars=_MULTI_EDIT_DIFF_MAX_CHARS // 2,
        )
    return diff or "(no diff)"


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
