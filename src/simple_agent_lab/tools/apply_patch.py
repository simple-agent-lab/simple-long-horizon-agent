"""Codex-style ``apply_patch`` edit tool.

This is a faithful port of OpenAI Codex's signature file-editing tool. Where the
Anthropic-flavoured ``edit`` tool (see ``edit.py``) does a single verbatim
string replacement, Codex edits files by sending a *patch envelope* in the V4A
diff format and lets the model touch several files in one call:

    *** Begin Patch
    *** Update File: path/to/file.py
    @@ def greet():
    -    print("hi")
    +    print("hello")
    *** Add File: path/to/new.py
    +print("brand new")
    *** Delete File: path/to/old.py
    *** End Patch

The grammar, in brief:

* The whole payload is wrapped in ``*** Begin Patch`` / ``*** End Patch``.
* Each file section starts with ``*** Add File:``, ``*** Update File:`` or
  ``*** Delete File:``. An ``Update File`` may be followed by ``*** Move to:``
  to rename as it patches.
* ``Add File`` lines are all ``+``-prefixed content.
* ``Update File`` hunks use ``@@`` headings to locate a region, then ` ``
  (context), ``-`` (remove) and ``+`` (add) lines — like a unified diff without
  line numbers. Context is matched against the file with a three-step fuzzy
  fallback (exact → ignore trailing whitespace → ignore all surrounding
  whitespace) so the model does not have to reproduce indentation byte-for-byte.

Why port it here: this lab compares the *Codex* harness against the *Claude
Code* harness, so we want both editing interfaces side by side. This tool
coexists with ``edit.py`` rather than replacing it.

As with the other tools, ``make_apply_patch_tool(...)`` is a factory returning
an ``AgentTool``, the parsing/applying logic lives in pure helpers that are easy
to unit test, and the structured result is a frozen dataclass. The default
execution mode is ``sequential`` because a patch mutates the workspace and
concurrent writes to the same file would race.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
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


APPLY_PATCH_TOOL_NAME = "apply_patch"

# Guard against pulling a giant file fully into memory just to patch a few
# lines. Mirrors ``edit.py``'s cap, scaled to this teaching codebase.
DEFAULT_MAX_PATCH_BYTES = 10 * 1024 * 1024  # 10 MiB

_BEGIN_PATCH = "*** Begin Patch"
_END_PATCH = "*** End Patch"
_UPDATE_FILE = "*** Update File: "
_DELETE_FILE = "*** Delete File: "
_ADD_FILE = "*** Add File: "
_MOVE_TO = "*** Move to: "
_END_OF_FILE = "*** End of File"
_SECTION_PREFIXES = (
    _BEGIN_PATCH,
    _END_PATCH,
    _UPDATE_FILE.rstrip(),
    _DELETE_FILE.rstrip(),
    _ADD_FILE.rstrip(),
    _END_OF_FILE,
)


class DiffError(ValueError):
    """A malformed patch or a context that does not match the target file."""


class ActionType(str, Enum):
    ADD = "add"
    DELETE = "delete"
    UPDATE = "update"


@dataclass
class Chunk:
    """One contiguous edit inside an ``Update File`` section.

    ``orig_index`` is the 0-based offset into the *original* file lines where
    the removed/added block begins; ``del_lines`` are removed, ``ins_lines`` are
    inserted in their place.
    """

    orig_index: int = -1
    del_lines: list[str] = field(default_factory=list)
    ins_lines: list[str] = field(default_factory=list)


@dataclass
class PatchAction:
    type: ActionType
    new_file: str | None = None
    chunks: list[Chunk] = field(default_factory=list)
    move_path: str | None = None


@dataclass
class Patch:
    actions: dict[str, PatchAction] = field(default_factory=dict)


@dataclass(frozen=True)
class ApplyPatchResult:
    """Structured accounting for one applied patch envelope.

    ``added``/``modified``/``deleted`` list the affected paths (modified
    includes renames). ``fuzz`` is the accumulated fuzzy-match cost across all
    hunks — ``0`` means every context matched exactly; a higher number means the
    patch only located its targets after relaxing whitespace, which is a useful
    signal that the model's context was slightly stale.
    """

    added: list[str]
    modified: list[str]
    deleted: list[str]
    fuzz: int


# --------------------------------------------------------------------------- #
#  Tool factory
# --------------------------------------------------------------------------- #
def make_apply_patch_tool(
    *,
    cwd: str | Path | None = None,
    max_bytes: int = DEFAULT_MAX_PATCH_BYTES,
    execution_mode: ToolExecutionMode = "sequential",
) -> AgentTool:
    """Return an ``AgentTool`` that applies a Codex V4A patch to the workspace."""

    root = Path(cwd or ".").resolve()

    def execute(
        call_id: str,
        args: dict[str, Any],
        abort: AbortFlag,
        on_update: ToolUpdateFn | None,
    ) -> ToolResult:
        del call_id, on_update
        if abort():
            return text_result("apply_patch aborted before start.", is_error=True)

        patch_text = args.get("input")
        if not isinstance(patch_text, str) or not patch_text.strip():
            return text_result(
                "Missing required apply_patch argument: input (the patch text).",
                is_error=True,
            )

        return apply_patch(patch_text, root=root, max_bytes=max_bytes)

    return AgentTool(
        name=APPLY_PATCH_TOOL_NAME,
        description=(
            "Apply a patch to one or more files in the V4A diff format. The "
            "patch is wrapped in '*** Begin Patch' / '*** End Patch'. Each file "
            "section starts with '*** Add File: <path>', '*** Update File: "
            "<path>' (optionally followed by '*** Move to: <path>' to rename), "
            "or '*** Delete File: <path>'. Add lines are '+'-prefixed. Update "
            "hunks use '@@' headings to locate a region, then ' ' context, '-' "
            "removed and '+' added lines. Prefer this over editing files with "
            "sed/awk in the shell."
        ),
        parameters={
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": (
                        "The full patch text, beginning with '*** Begin Patch' "
                        "and ending with '*** End Patch'."
                    ),
                },
            },
            "required": ["input"],
            "additionalProperties": False,
        },
        execute=execute,
        execution_mode=execution_mode,
    )


# --------------------------------------------------------------------------- #
#  Pure entry point
# --------------------------------------------------------------------------- #
def apply_patch(
    patch_text: str,
    *,
    root: str | Path | None = None,
    max_bytes: int = DEFAULT_MAX_PATCH_BYTES,
) -> ToolResult:
    """Parse ``patch_text`` and apply it to the workspace under ``root``.

    All disk reads/writes go through ``root``; the patch itself is validated
    fully before anything is written, so a malformed envelope leaves the
    workspace untouched. Returns a model-visible ``ToolResult`` — a per-file
    ``A``/``M``/``D`` summary on success, or a ``DiffError`` message on failure.
    """

    base = Path(root or ".").resolve()

    # Validate the envelope before touching the filesystem so a malformed patch
    # fails with a clear message rather than a confusing "missing file" from the
    # pre-load pass below.
    lines = patch_text.splitlines()
    if (
        len(lines) < 2
        or not lines[0].startswith(_BEGIN_PATCH)
        or lines[-1].strip() != _END_PATCH
    ):
        return text_result(
            f"apply_patch failed: patch must start with '{_BEGIN_PATCH}' and end "
            f"with '{_END_PATCH}'.",
            is_error=True,
        )

    needed = _identify_files_needed(patch_text)
    added = _identify_files_added(patch_text)

    # Load every file an Update/Delete section refers to, validating existence
    # and the size cap up front.
    current: dict[str, str] = {}
    for rel in needed:
        path = _resolve(base, rel)
        if not path.exists():
            return text_result(
                f"apply_patch failed: cannot update/delete missing file: {rel}",
                is_error=True,
            )
        if path.is_dir():
            return text_result(
                f"apply_patch failed: path is a directory, not a file: {rel}",
                is_error=True,
            )
        try:
            size = path.stat().st_size
        except OSError as exc:
            return text_result(
                f"apply_patch failed: cannot stat {rel}: {exc}", is_error=True
            )
        if size > max_bytes:
            return text_result(
                f"apply_patch failed: {rel} is too large to patch "
                f"({size} bytes, limit {max_bytes}).",
                is_error=True,
            )
        try:
            current[rel] = path.read_text(encoding="utf-8")
        except OSError as exc:
            return text_result(
                f"apply_patch failed: cannot read {rel}: {exc}", is_error=True
            )

    # An Add section must not clobber an existing file.
    for rel in added:
        if _resolve(base, rel).exists():
            return text_result(
                f"apply_patch failed: cannot add file that already exists: {rel}",
                is_error=True,
            )

    try:
        patch, fuzz = _text_to_patch(patch_text, current)
        commit = _patch_to_commit(patch, current)
    except DiffError as exc:
        return text_result(f"apply_patch failed: {exc}", is_error=True)

    try:
        summary = _apply_commit(commit, base)
    except OSError as exc:
        return text_result(f"apply_patch failed while writing: {exc}", is_error=True)

    result = ApplyPatchResult(
        added=summary["added"],
        modified=summary["modified"],
        deleted=summary["deleted"],
        fuzz=fuzz,
    )
    return text_result(_render_summary(result), details=asdict(result))


# --------------------------------------------------------------------------- #
#  Parser
# --------------------------------------------------------------------------- #
@dataclass
class _Parser:
    """Recursive-descent parser over the patch's lines.

    Mirrors the structure of Codex's reference implementation: ``parse()`` walks
    file sections; ``_parse_update_file`` resolves each hunk's context against
    the live file contents and records ``Chunk``s with absolute ``orig_index``.
    """

    current_files: dict[str, str]
    lines: list[str]
    index: int = 0
    patch: Patch = field(default_factory=Patch)
    fuzz: int = 0

    def _cur(self) -> str:
        if self.index >= len(self.lines):
            raise DiffError("unexpected end of input while parsing patch")
        return self.lines[self.index]

    def _startswith(self, prefix: str | tuple[str, ...]) -> bool:
        return self._cur().startswith(prefix)

    def _read_str(self, prefix: str) -> str:
        """Consume the current line if it starts with ``prefix``; return the rest."""
        if self._cur().startswith(prefix):
            text = self._cur()[len(prefix) :]
            self.index += 1
            return text
        return ""

    def _is_done(self, prefixes: tuple[str, ...]) -> bool:
        return self.index >= len(self.lines) or self._cur().startswith(prefixes)

    def parse(self) -> None:
        while not self._is_done((_END_PATCH,)):
            path = self._read_str(_UPDATE_FILE)
            if path:
                self._guard_unique(path)
                move_to = self._read_str(_MOVE_TO) or None
                if path not in self.current_files:
                    raise DiffError(f"update for missing file: {path}")
                action = self._parse_update_file(self.current_files[path])
                action.move_path = move_to
                self.patch.actions[path] = action
                continue
            path = self._read_str(_DELETE_FILE)
            if path:
                self._guard_unique(path)
                if path not in self.current_files:
                    raise DiffError(f"delete for missing file: {path}")
                self.patch.actions[path] = PatchAction(type=ActionType.DELETE)
                continue
            path = self._read_str(_ADD_FILE)
            if path:
                self._guard_unique(path)
                self.patch.actions[path] = self._parse_add_file()
                continue
            raise DiffError(f"unknown line while parsing: {self._cur()!r}")

        if not self._startswith(_END_PATCH):
            raise DiffError(f"missing '{_END_PATCH}' sentinel")
        self.index += 1

    def _guard_unique(self, path: str) -> None:
        if path in self.patch.actions:
            raise DiffError(f"duplicate section for file: {path}")

    def _parse_add_file(self) -> PatchAction:
        added: list[str] = []
        while not self._is_done(_SECTION_PREFIXES):
            line = self._cur()
            self.index += 1
            if not line.startswith("+"):
                raise DiffError(f"invalid Add File line (missing '+'): {line!r}")
            added.append(line[1:])
        return PatchAction(type=ActionType.ADD, new_file="\n".join(added))

    def _parse_update_file(self, text: str) -> PatchAction:
        action = PatchAction(type=ActionType.UPDATE)
        file_lines = text.split("\n")
        index = 0
        while not self._is_done(
            (
                _END_PATCH,
                _UPDATE_FILE.rstrip(),
                _DELETE_FILE.rstrip(),
                _ADD_FILE.rstrip(),
            )
        ):
            heading = self._read_str("@@ ")
            if not heading and self._cur() == "@@":
                self.index += 1  # a bare '@@' separator carries no heading
            elif not heading and index != 0:
                # After the first hunk, a new hunk must announce itself with a
                # context line; a stray line here is malformed.
                if not self._startswith((" ", "-", "+")):
                    raise DiffError(f"invalid line in update section: {self._cur()!r}")

            if heading.strip():
                index = _advance_to_heading(file_lines, heading, index, self)

            ctx, chunks, end_index, eof = _peek_next_section(self.lines, self.index)
            found, local_fuzz = _find_context(file_lines, ctx, index, eof)
            if found == -1:
                joined = "\n".join(ctx)
                raise DiffError(
                    f"could not find patch context near line {index}:\n{joined}"
                )
            self.fuzz += local_fuzz
            for chunk in chunks:
                chunk.orig_index += found
                action.chunks.append(chunk)
            index = found + len(ctx)
            self.index = end_index
        return action


def _advance_to_heading(
    file_lines: list[str], heading: str, index: int, parser: _Parser
) -> int:
    """Move ``index`` just past the file line matching an ``@@`` heading.

    Tries an exact match first, then a whitespace-insensitive one (counting a
    fuzz point). A heading that cannot be found is non-fatal — it is only a hint
    for locating the following context — so ``index`` is returned unchanged.
    """

    for offset, line in enumerate(file_lines[index:], index):
        if line == heading:
            return offset + 1
    for offset, line in enumerate(file_lines[index:], index):
        if line.strip() == heading.strip():
            parser.fuzz += 1
            return offset + 1
    return index


# --------------------------------------------------------------------------- #
#  Context matching
# --------------------------------------------------------------------------- #
def _find_context_core(
    lines: list[str], context: list[str], start: int
) -> tuple[int, int]:
    """Locate ``context`` in ``lines`` at or after ``start``.

    Returns ``(index, fuzz)`` with three escalating tolerances: exact (fuzz 0),
    ignore trailing whitespace (fuzz 1), ignore all surrounding whitespace
    (fuzz 100). ``(-1, 0)`` when nothing matches.
    """

    if not context:
        return start, 0
    end = len(lines) - len(context)
    for i in range(start, end + 1):
        if lines[i : i + len(context)] == context:
            return i, 0
    rstripped = [c.rstrip() for c in context]
    for i in range(start, end + 1):
        if [s.rstrip() for s in lines[i : i + len(context)]] == rstripped:
            return i, 1
    stripped = [c.strip() for c in context]
    for i in range(start, end + 1):
        if [s.strip() for s in lines[i : i + len(context)]] == stripped:
            return i, 100
    return -1, 0


def _find_context(
    lines: list[str], context: list[str], start: int, eof: bool
) -> tuple[int, int]:
    """``_find_context_core`` with an end-of-file bias.

    When a hunk is anchored to EOF, the context is preferentially matched at the
    very end of the file; falling back to a normal search costs a large fuzz
    penalty so an exact non-EOF match elsewhere still wins on a tie.
    """

    if eof:
        tail = max(start, len(lines) - len(context))
        found, fuzz = _find_context_core(lines, context, tail)
        if found != -1:
            return found, fuzz
        found, fuzz = _find_context_core(lines, context, start)
        return found, fuzz + 10_000
    return _find_context_core(lines, context, start)


def _peek_next_section(
    lines: list[str], index: int
) -> tuple[list[str], list[Chunk], int, bool]:
    """Read one hunk body, returning ``(context, chunks, next_index, eof)``.

    ``context`` is the full original block (context + removed lines) used to
    locate the hunk; ``chunks`` carry the actual insert/delete edits with
    file-relative ``orig_index`` offsets within that block.
    """

    original: list[str] = []
    del_lines: list[str] = []
    ins_lines: list[str] = []
    chunks: list[Chunk] = []
    mode = "keep"

    while index < len(lines):
        line = lines[index]
        if line.startswith(
            (
                "@@",
                _BEGIN_PATCH,
                _END_PATCH,
                _UPDATE_FILE.rstrip(),
                _DELETE_FILE.rstrip(),
                _ADD_FILE.rstrip(),
                _END_OF_FILE,
            )
        ):
            break
        if line == "***":
            break
        if line.startswith("***"):
            raise DiffError(f"invalid line in hunk: {line!r}")
        index += 1
        last_mode = mode
        if line.startswith("+"):
            mode = "add"
        elif line.startswith("-"):
            mode = "delete"
        elif line.startswith(" "):
            mode = "keep"
        elif line == "":
            # A model that drops the leading space on a blank context line is
            # tolerated: treat it as a blank keep line.
            mode = "keep"
            line = " "
        else:
            raise DiffError(f"invalid line in hunk: {line!r}")
        content = line[1:]

        # A transition back into context closes the current add/delete run.
        if mode == "keep" and last_mode != mode:
            if ins_lines or del_lines:
                chunks.append(
                    Chunk(
                        orig_index=len(original) - len(del_lines),
                        del_lines=del_lines,
                        ins_lines=ins_lines,
                    )
                )
                del_lines, ins_lines = [], []

        if mode == "delete":
            del_lines.append(content)
            original.append(content)
        elif mode == "add":
            ins_lines.append(content)
        else:  # keep
            original.append(content)

    if ins_lines or del_lines:
        chunks.append(
            Chunk(
                orig_index=len(original) - len(del_lines),
                del_lines=del_lines,
                ins_lines=ins_lines,
            )
        )

    eof = index < len(lines) and lines[index] == _END_OF_FILE
    if eof:
        index += 1
    return original, chunks, index, eof


# --------------------------------------------------------------------------- #
#  Patch -> in-memory commit -> disk
# --------------------------------------------------------------------------- #
def _text_to_patch(text: str, orig: dict[str, str]) -> tuple[Patch, int]:
    lines = text.splitlines()  # drop line terminators; we re-join with '\n'
    if (
        len(lines) < 2
        or not lines[0].startswith(_BEGIN_PATCH)
        or lines[-1].strip() != _END_PATCH
    ):
        raise DiffError(
            f"patch must start with '{_BEGIN_PATCH}' and end with '{_END_PATCH}'"
        )
    parser = _Parser(current_files=orig, lines=lines, index=1)
    parser.parse()
    return parser.patch, parser.fuzz


def _get_updated_file(text: str, action: PatchAction, path: str) -> str:
    orig_lines = text.split("\n")
    dest: list[str] = []
    cursor = 0
    for chunk in action.chunks:
        if chunk.orig_index > len(orig_lines):
            raise DiffError(
                f"{path}: chunk index {chunk.orig_index} exceeds file length "
                f"{len(orig_lines)}"
            )
        if cursor > chunk.orig_index:
            raise DiffError(f"{path}: overlapping chunks at {chunk.orig_index}")
        dest.extend(orig_lines[cursor : chunk.orig_index])
        dest.extend(chunk.ins_lines)
        cursor = chunk.orig_index + len(chunk.del_lines)
    dest.extend(orig_lines[cursor:])
    return "\n".join(dest)


def _patch_to_commit(patch: Patch, orig: dict[str, str]) -> dict[str, dict[str, Any]]:
    commit: dict[str, dict[str, Any]] = {}
    for path, action in patch.actions.items():
        if action.type is ActionType.DELETE:
            commit[path] = {"type": "delete"}
        elif action.type is ActionType.ADD:
            commit[path] = {"type": "add", "content": action.new_file or ""}
        else:  # update
            commit[path] = {
                "type": "update",
                "content": _get_updated_file(orig[path], action, path),
                "move_path": action.move_path,
            }
    return commit


def _apply_commit(
    commit: dict[str, dict[str, Any]], base: Path
) -> dict[str, list[str]]:
    """Write a parsed commit to disk; return per-action path lists."""

    summary: dict[str, list[str]] = {"added": [], "modified": [], "deleted": []}
    for path, change in commit.items():
        target = _resolve(base, path)
        if change["type"] == "delete":
            target.unlink()
            summary["deleted"].append(path)
        elif change["type"] == "add":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(change["content"], encoding="utf-8")
            summary["added"].append(path)
        else:  # update
            move_to = change.get("move_path")
            if move_to:
                dest = _resolve(base, move_to)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(change["content"], encoding="utf-8")
                if dest != target:
                    target.unlink()
                summary["modified"].append(f"{path} -> {move_to}")
            else:
                target.write_text(change["content"], encoding="utf-8")
                summary["modified"].append(path)
    return summary


def _render_summary(result: ApplyPatchResult) -> str:
    lines = ["Success. Updated the following files:"]
    for path in result.added:
        lines.append(f"A {path}")
    for path in result.modified:
        lines.append(f"M {path}")
    for path in result.deleted:
        lines.append(f"D {path}")
    if result.fuzz:
        lines.append(
            f"(applied with fuzz={result.fuzz}; context matched after relaxing "
            "whitespace — double-check the result)"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _identify_files_needed(text: str) -> list[str]:
    """Paths referenced by Update/Delete sections (must already exist)."""

    paths: list[str] = []
    for line in text.splitlines():
        if line.startswith(_UPDATE_FILE):
            paths.append(line[len(_UPDATE_FILE) :])
        elif line.startswith(_DELETE_FILE):
            paths.append(line[len(_DELETE_FILE) :])
    return paths


def _identify_files_added(text: str) -> list[str]:
    """Paths referenced by Add sections (must not yet exist)."""

    return [
        line[len(_ADD_FILE) :]
        for line in text.splitlines()
        if line.startswith(_ADD_FILE)
    ]


def _resolve(base: Path, rel: str) -> Path:
    candidate = Path(rel)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate
