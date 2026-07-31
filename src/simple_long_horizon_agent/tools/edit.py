"""Small file-edit tool for local demos.

A compact port of the `Edit` tool from Anthropic's reference agent. It performs
an exact string replacement inside a single workspace file:

* ``old_string`` is matched verbatim against the file's text. By default the
  match must be unique — if it appears more than once the edit is refused so the
  model adds context instead of silently changing the wrong line. Set
  ``replace_all`` to rewrite every occurrence (handy for renames).
* An empty ``old_string`` means "create this file" — it writes ``new_string``
  to a new (or empty) file, creating parent directories as needed.
* Guard rails mirror the reference tool: identical old/new strings, missing
  files, and writing over an existing file are all returned as model-visible
  errors so the model can self-correct.

This mirrors `bash.py` and `read.py`: a `make_edit_tool(...)` factory returns an
`AgentTool`, the structured result lives in a frozen dataclass, and the actual
edit lives in a pure `edit_file(...)` helper that is easy to unit test. Unlike
the read tool, the default execution mode is ``sequential`` because an edit
mutates the workspace and concurrent writes to the same file would race.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import (
    AbortFlag,
    AgentTool,
    ToolExecutionMode,
    ToolResult,
    ToolUpdateFn,
    text_result,
)


EDIT_TOOL_NAME = "edit"

# Guard against loading a multi-gigabyte file fully into memory just to patch a
# few characters. Mirrors the spirit of the reference tool's size cap, scaled to
# this teaching codebase.
DEFAULT_MAX_EDIT_BYTES = 10 * 1024 * 1024  # 10 MiB


@dataclass(frozen=True)
class EditResult:
    """Structured accounting for one applied edit."""

    path: str
    created: bool
    replacements: int
    replace_all: bool


def make_edit_tool(
    *,
    cwd: str | Path | None = None,
    max_bytes: int = DEFAULT_MAX_EDIT_BYTES,
    execution_mode: ToolExecutionMode = "sequential",
) -> AgentTool:
    """Return an `AgentTool` that edits one file in the local workspace."""

    root = Path(cwd or ".").resolve()

    def execute(
        call_id: str,
        args: dict[str, Any],
        abort: AbortFlag,
        on_update: ToolUpdateFn | None,
    ) -> ToolResult:
        del call_id, on_update
        if abort():
            return text_result("Edit aborted before start.", is_error=True)

        raw_path = str(args.get("path", "")).strip()
        if not raw_path:
            return text_result("Missing required edit argument: path.", is_error=True)

        old_string = args.get("old_string")
        new_string = args.get("new_string")
        if not isinstance(old_string, str):
            return text_result(
                "Missing required edit argument: old_string.", is_error=True
            )
        if not isinstance(new_string, str):
            return text_result(
                "Missing required edit argument: new_string.", is_error=True
            )

        replace_all = bool(args.get("replace_all", False))

        return edit_file(
            raw_path,
            old_string,
            new_string,
            replace_all=replace_all,
            root=root,
            max_bytes=max_bytes,
        )

    return AgentTool(
        name=EDIT_TOOL_NAME,
        description=(
            "Performs exact string replacements in a single file. Match "
            "`old_string` verbatim (including indentation); the edit fails if it "
            "is not unique unless you set `replace_all` to true. Use an empty "
            "`old_string` to create a new file from `new_string`. Prefer this "
            "over editing files with `sed`/`awk` in bash."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit (relative or absolute).",
                },
                "old_string": {
                    "type": "string",
                    "description": (
                        "The exact text to replace. Use an empty string to "
                        "create a new file."
                    ),
                },
                "new_string": {
                    "type": "string",
                    "description": (
                        "The replacement text (must differ from old_string)."
                    ),
                },
                "replace_all": {
                    "type": "boolean",
                    "description": (
                        "Replace every occurrence of old_string (default false)."
                    ),
                },
            },
            "required": ["path", "old_string", "new_string"],
            "additionalProperties": False,
        },
        execute=execute,
        execution_mode=execution_mode,
    )


def edit_file(
    raw_path: str,
    old_string: str,
    new_string: str,
    *,
    replace_all: bool = False,
    root: str | Path | None = None,
    max_bytes: int = DEFAULT_MAX_EDIT_BYTES,
) -> ToolResult:
    """Apply one string replacement to `raw_path` and return a `ToolResult`.

    Resolves relative paths against `root`, validates the edit against the same
    guard rails as the reference tool, writes the result to disk, and reports a
    model-visible success or error message.
    """

    if old_string == new_string:
        return text_result(
            "No changes to make: old_string and new_string are identical.",
            is_error=True,
        )

    base = Path(root or ".").resolve()
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = base / candidate
    try:
        path = candidate.resolve()
    except OSError as exc:
        return text_result(f"Could not resolve path {raw_path!r}: {exc}", is_error=True)

    if path.is_dir():
        return text_result(
            f"Path is a directory, not a file: {raw_path}", is_error=True
        )

    # Creating a new file: empty old_string is the "create" signal.
    if not path.exists():
        if old_string != "":
            return text_result(
                f"File does not exist: {raw_path}. Use an empty old_string to "
                "create a new file.",
                is_error=True,
            )
        return _write_new_file(path, raw_path, new_string)

    try:
        size = path.stat().st_size
    except OSError as exc:
        return text_result(f"Failed to stat {raw_path}: {exc}", is_error=True)
    if size > max_bytes:
        return text_result(
            f"File is too large to edit ({size} bytes, limit {max_bytes}).",
            is_error=True,
        )

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return text_result(f"Failed to read {raw_path}: {exc}", is_error=True)

    if old_string == "":
        if content != "":
            return text_result(
                f"Cannot create new file - file already exists: {raw_path}.",
                is_error=True,
            )
        return _write_new_file(path, raw_path, new_string)

    matches = content.count(old_string)
    if matches == 0:
        return text_result(
            f"String to replace not found in file.\nString: {old_string}",
            is_error=True,
        )
    if matches > 1 and not replace_all:
        return text_result(
            f"Found {matches} matches of the string to replace, but replace_all "
            "is false. Add surrounding context to uniquely identify the instance, "
            "or set replace_all to true to change every occurrence.\n"
            f"String: {old_string}",
            is_error=True,
        )

    if replace_all:
        updated = content.replace(old_string, new_string)
        replacements = matches
    else:
        updated = content.replace(old_string, new_string, 1)
        replacements = 1

    try:
        path.write_text(updated, encoding="utf-8")
    except OSError as exc:
        return text_result(f"Failed to write {raw_path}: {exc}", is_error=True)

    result = EditResult(
        path=str(path),
        created=False,
        replacements=replacements,
        replace_all=replace_all,
    )
    if replace_all:
        message = (
            f"The file {raw_path} has been updated. "
            f"Replaced {replacements} occurrence(s)."
        )
    else:
        message = f"The file {raw_path} has been updated successfully."
    return text_result(message, details=asdict(result))


def _write_new_file(path: Path, raw_path: str, new_string: str) -> ToolResult:
    """Create (or fill) a file with `new_string`, making parent dirs as needed."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_string, encoding="utf-8")
    except OSError as exc:
        return text_result(f"Failed to create {raw_path}: {exc}", is_error=True)

    result = EditResult(
        path=str(path),
        created=True,
        replacements=0,
        replace_all=False,
    )
    return text_result(
        f"The file {raw_path} has been created successfully.",
        details=asdict(result),
    )
