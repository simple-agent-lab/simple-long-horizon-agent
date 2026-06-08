"""Notes-memory implementation: durable notes plus transcript search."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import time
from contextlib import closing
from pathlib import Path
from typing import Any

from simple_agent_lab.memory.base import Memory, MemoryContext, memory_context_message
from simple_agent_lab.memory.transcript import (
    extract_memory_text,
    simple_session_summary,
)
from simple_agent_lab.messages import Message
from simple_agent_lab.tools import AbortFlag, AgentTool, ToolResult, ToolUpdateFn
from simple_agent_lab.tools import text_result


DEFAULT_MEMORY_HOME = "~/.simple"
ENTRY_SEPARATOR = "\n\n---\n\n"
DEFAULT_CHAR_LIMIT = 2_200
INVISIBLE_RE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060-\u2064\uFEFF]")
SEPARATOR_RE = re.compile(r"(?m)^---\s*$")


class NotesMemory(Memory):
    """Method A: bounded notes plus searchable past-session transcripts."""

    def __init__(
        self,
        *,
        home: str | Path = DEFAULT_MEMORY_HOME,
        memory_path: str | Path | None = None,
        sessions_path: str | Path | None = None,
        char_limit: int = DEFAULT_CHAR_LIMIT,
        builtin_enabled: bool = True,
    ) -> None:
        root = Path(home).expanduser()
        self.builtin_enabled = builtin_enabled
        self.sessions_path = (
            Path(sessions_path).expanduser()
            if sessions_path is not None
            else root / "sessions.db"
        )
        self.memory_file = MemoryFile(
            memory_path or root / "MEMORY.md",
            char_limit=char_limit,
        )
        self.sessions = SessionSearchStore(self.sessions_path)

    def initial(self, ctx: MemoryContext) -> tuple[Message, ...]:
        if not self.builtin_enabled:
            return ()
        snapshot = self.memory_file.load_snapshot()
        if not snapshot:
            return ()
        return (memory_context_message(snapshot, target=ctx.agent),)

    def tools(self, ctx: MemoryContext) -> tuple[AgentTool, ...]:
        del ctx
        tools: list[AgentTool] = []
        if self.builtin_enabled:
            tools.append(self._memory_tool())
        tools.append(self._session_search_tool())
        return tuple(tools)

    def finish(self, ctx: MemoryContext) -> None:
        if ctx.state is None or not ctx.session_id:
            return
        try:
            messages = ctx.state.messages
            self.sessions.record_session(
                ctx.session_id,
                messages,
                summary=simple_session_summary(messages),
            )
        except Exception:
            return

    def _memory_tool(self) -> AgentTool:
        def execute(
            call_id: str,
            args: dict[str, Any],
            abort: AbortFlag,
            on_update: ToolUpdateFn | None,
        ) -> ToolResult:
            del call_id, abort, on_update
            action = str(args.get("action", "")).strip()
            if action == "add":
                result = self.memory_file.add(str(args.get("content", "")))
            elif action == "replace":
                result = self.memory_file.replace(
                    str(args.get("old_text", "")),
                    str(args.get("content", "")),
                )
            elif action == "remove":
                result = self.memory_file.remove(str(args.get("old_text", "")))
            else:
                result = {
                    "success": False,
                    "error": "action must be one of: add, replace, remove",
                }
            return _json_tool_result(result)

        return AgentTool(
            name="memory",
            description=(
                "Save compact durable notes for future runs. Use add, replace, "
                "or remove. Keep entries evidence-backed and avoid raw logs or "
                "task-local details."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "replace", "remove"],
                    },
                    "content": {
                        "type": "string",
                        "description": "New entry text. Required for add/replace.",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Unique substring. Required for replace/remove.",
                    },
                },
                "required": ["action"],
                "additionalProperties": False,
            },
            execute=execute,
            execution_mode="sequential",
        )

    def _session_search_tool(self) -> AgentTool:
        store = self.sessions

        def execute(
            call_id: str,
            args: dict[str, Any],
            abort: AbortFlag,
            on_update: ToolUpdateFn | None,
        ) -> ToolResult:
            del call_id, abort, on_update
            limit = _bounded_int(args.get("limit", 5), default=5, minimum=1, maximum=20)
            session_id = str(args.get("session_id", "")).strip()
            around_index = args.get("around_message_index")
            if session_id and around_index is not None:
                try:
                    index = int(around_index)
                except (TypeError, ValueError):
                    return _json_tool_result(
                        {
                            "success": False,
                            "error": "around_message_index must be an integer.",
                        }
                    )
                window = _bounded_int(
                    args.get("window", 5),
                    default=5,
                    minimum=1,
                    maximum=20,
                )
                return _json_tool_result(
                    {
                        "success": True,
                        "mode": "scroll",
                        **store.scroll(session_id, index, window=window),
                    }
                )

            query = str(args.get("query", "")).strip()
            if not query:
                sessions = store.browse(limit=limit)
                return _json_tool_result(
                    {
                        "success": True,
                        "mode": "browse",
                        "session_count": len(sessions),
                        "sessions": sessions,
                    }
                )
            sessions = store.search(query, limit=limit)
            return _json_tool_result(
                {
                    "success": True,
                    "mode": "discovery",
                    "query": query,
                    "session_count": len(sessions),
                    "sessions": sessions,
                }
            )

        return AgentTool(
            name="session_search",
            description=(
                "Full-text search over transcripts of past sessions stored "
                "locally. Pass query for discovery, pass session_id plus "
                "around_message_index to scroll a found session, or omit query "
                "to browse recent sessions."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "limit": {
                        "type": "integer",
                        "description": "Max past sessions to return.",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Past session id to scroll.",
                    },
                    "around_message_index": {
                        "type": "integer",
                        "description": "Message index to center the scroll window on.",
                    },
                    "window": {
                        "type": "integer",
                        "description": "Messages before and after the anchor to return.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
            execute=execute,
            execution_mode="sequential",
        )


def _json_tool_result(payload: dict[str, Any]) -> ToolResult:
    return text_result(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        details=payload,
        is_error=False,
    )


class MemoryFile:
    """Bounded Markdown notes file with frozen per-run snapshots."""

    def __init__(
        self,
        path: str | Path,
        *,
        char_limit: int = DEFAULT_CHAR_LIMIT,
    ) -> None:
        if char_limit <= 0:
            raise ValueError("char_limit must be > 0")
        self.path = Path(path).expanduser()
        self.char_limit = char_limit
        self._snapshot: str | None = None

    def load(self) -> list[str]:
        if not self.path.exists():
            return []
        text = self.path.read_text(encoding="utf-8")
        entries = [entry.strip() for entry in text.split(ENTRY_SEPARATOR)]
        return list(dict.fromkeys(entry for entry in entries if entry))

    def save(self, entries: list[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        content = ENTRY_SEPARATOR.join(entries)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self.path.parent,
            delete=False,
        ) as handle:
            tmp = Path(handle.name)
            try:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
        try:
            os.replace(tmp, self.path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def load_snapshot(self) -> str:
        self._snapshot = self._render(self.load())
        return self._snapshot

    def render_snapshot(self) -> str:
        if self._snapshot is None:
            return self.load_snapshot()
        return self._snapshot

    def add(self, content: str) -> dict[str, Any]:
        content = content.strip()
        if not content:
            return self._error("Content cannot be empty.")
        if message := _invisible_error(content):
            return self._error(message)
        if message := _separator_error(content):
            return self._error(message)
        entries = self.load()
        if content in entries:
            return self._ok(entries, "Entry already exists (no duplicate added).")
        new_entries = [*entries, content]
        if (
            error := self._capacity_error(new_entries, "adding", len(content))
        ) is not None:
            return error
        self.save(new_entries)
        return self._ok(new_entries, "Entry added.")

    def replace(self, old_text: str, content: str) -> dict[str, Any]:
        old_text = old_text.strip()
        content = content.strip()
        if not old_text:
            return self._error("old_text cannot be empty.")
        if not content:
            return self._error(
                "content cannot be empty. Use 'remove' to delete entries."
            )
        if message := _invisible_error(content):
            return self._error(message)
        if message := _separator_error(content):
            return self._error(message)
        entries = self.load()
        index = _unique_match(entries, old_text)
        if index is None:
            return self._match_error(entries, old_text)
        if content in entries and entries.index(content) != index:
            return self._error(
                "content already exists in another entry. Use remove instead."
            )
        new_entries = list(entries)
        new_entries[index] = content
        if (
            error := self._capacity_error(new_entries, "replacing", len(content))
        ) is not None:
            return error
        self.save(new_entries)
        return self._ok(new_entries, "Entry replaced.")

    def remove(self, old_text: str) -> dict[str, Any]:
        old_text = old_text.strip()
        if not old_text:
            return self._error("old_text cannot be empty.")
        entries = self.load()
        index = _unique_match(entries, old_text)
        if index is None:
            return self._match_error(entries, old_text)
        new_entries = [entry for i, entry in enumerate(entries) if i != index]
        self.save(new_entries)
        return self._ok(new_entries, "Entry removed.")

    def _render(self, entries: list[str]) -> str:
        if not entries:
            return ""
        return "\n".join(
            [
                "MEMORY (persistent notes) [" + _usage(entries, self.char_limit) + "]",
                ENTRY_SEPARATOR.join(entries),
            ]
        )

    def _ok(self, entries: list[str], message: str) -> dict[str, Any]:
        return {
            "success": True,
            "message": message,
            "entries": entries,
            "entry_count": len(entries),
            "usage": _usage(entries, self.char_limit),
        }

    def _error(self, message: str) -> dict[str, Any]:
        entries = self.load()
        return {
            "success": False,
            "error": message,
            "entries": entries,
            "entry_count": len(entries),
            "usage": _usage(entries, self.char_limit),
        }

    def _capacity_error(
        self,
        entries: list[str],
        action: str,
        content_len: int,
    ) -> dict[str, Any] | None:
        used = _chars(entries)
        if used <= self.char_limit:
            return None
        current_entries = self.load()
        current = _chars(current_entries)
        return {
            "success": False,
            "error": (
                f"Memory at {current}/{self.char_limit} chars. "
                f"{action.title()} this entry ({content_len} chars) would push to "
                f"{used}, exceeding the limit. Replace or remove existing entries first."
            ),
            "entries": current_entries,
            "entry_count": len(current_entries),
            "usage": _usage(current_entries, self.char_limit),
        }

    def _match_error(self, entries: list[str], old_text: str) -> dict[str, Any]:
        matches = [entry for entry in entries if old_text in entry]
        if not matches:
            return self._error(f"No entry contains {old_text!r}.")
        return {
            "success": False,
            "error": f"{len(matches)} entries match {old_text!r}. Be more specific.",
            "matches": [_truncate(match, 80) for match in matches],
            "entries": entries,
            "entry_count": len(entries),
            "usage": _usage(entries, self.char_limit),
        }


class SessionSearchStore:
    """SQLite FTS archive over past runtime messages."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self._ensure_schema()

    def record_session(
        self,
        session_id: str,
        messages: list[Message],
        *,
        summary: str = "",
    ) -> int:
        if not session_id:
            return 0
        ended_at = time.time()
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?)",
                (session_id, ended_at, ended_at, summary, len(messages)),
            )
            conn.execute("DELETE FROM messages_fts WHERE session_id = ?", (session_id,))
            count = 0
            for index, message in enumerate(messages):
                content = extract_memory_text(message)
                if not content:
                    continue
                conn.execute(
                    "INSERT INTO messages_fts(session_id, role, idx, content) "
                    "VALUES (?, ?, ?, ?)",
                    (session_id, message.role, index, content),
                )
                count += 1
        return count

    def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        limit = _bounded_int(limit, default=5, minimum=1, maximum=20)
        sanitized = sanitize_fts5_query(query)
        if not sanitized:
            return []
        try:
            with closing(self._connect()) as conn:
                rows = conn.execute(
                    """
                    SELECT messages_fts.session_id, sessions.summary,
                           sessions.started_at, sessions.n_messages,
                           messages_fts.role, messages_fts.idx,
                           snippet(messages_fts, 3, '<<', '>>', '...', 32) AS snippet
                    FROM messages_fts
                    LEFT JOIN sessions ON sessions.session_id = messages_fts.session_id
                    WHERE messages_fts MATCH ?
                    ORDER BY bm25(messages_fts)
                    LIMIT ?
                    """,
                    (sanitized, limit * 8),
                ).fetchall()
                results: list[dict[str, Any]] = []
                by_session: dict[str, dict[str, Any]] = {}
                for row in rows:
                    session_id = str(row["session_id"])
                    if session_id not in by_session:
                        if len(results) >= limit:
                            continue
                        item = {
                            "session_id": session_id,
                            "summary": _truncate(str(row["summary"] or ""), 2200),
                            "started_at": row["started_at"],
                            "n_messages": row["n_messages"],
                            "matches": [],
                        }
                        by_session[session_id] = item
                        results.append(item)
                    matches = by_session[session_id]["matches"]
                    if len(matches) >= 3:
                        continue
                    matches.append(
                        {
                            "role": row["role"],
                            "idx": row["idx"],
                            "message_index": row["idx"],
                            "snippet": _truncate(str(row["snippet"] or ""), 350),
                            "context": self._context(conn, session_id, int(row["idx"])),
                        }
                    )
                for result in results:
                    result["match_count"] = len(result["matches"])
                return results
        except sqlite3.OperationalError:
            return []

    def browse(self, *, limit: int = 5) -> list[dict[str, Any]]:
        limit = _bounded_int(limit, default=5, minimum=1, maximum=20)
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT session_id, summary, started_at, ended_at, n_messages
                FROM sessions
                ORDER BY ended_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [self._session_row(row) for row in rows]

    def scroll(
        self,
        session_id: str,
        around_message_index: int,
        *,
        window: int = 5,
    ) -> dict[str, Any]:
        window = _bounded_int(window, default=5, minimum=1, maximum=20)
        start = max(0, around_message_index - window)
        end = around_message_index + window
        with closing(self._connect()) as conn:
            info_row = conn.execute(
                """
                SELECT session_id, summary, started_at, ended_at, n_messages
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            rows = conn.execute(
                """
                SELECT role, idx, content
                FROM messages_fts
                WHERE session_id = ? AND CAST(idx AS INTEGER) BETWEEN ? AND ?
                ORDER BY CAST(idx AS INTEGER)
                """,
                (session_id, start, end),
            ).fetchall()
            messages = [
                {
                    "role": row["role"],
                    "idx": row["idx"],
                    "message_index": row["idx"],
                    "content": _truncate(str(row["content"] or ""), 1200),
                }
                for row in rows
            ]
            first_index = messages[0]["idx"] if messages else start
            last_index = messages[-1]["idx"] if messages else end
            before = conn.execute(
                """
                SELECT COUNT(*)
                FROM messages_fts
                WHERE session_id = ? AND CAST(idx AS INTEGER) < ?
                """,
                (session_id, first_index),
            ).fetchone()[0]
            after = conn.execute(
                """
                SELECT COUNT(*)
                FROM messages_fts
                WHERE session_id = ? AND CAST(idx AS INTEGER) > ?
                """,
                (session_id, last_index),
            ).fetchone()[0]

        return {
            "session": self._session_row(info_row) if info_row is not None else None,
            "session_id": session_id,
            "around_message_index": around_message_index,
            "window": window,
            "message_count": len(messages),
            "messages_before": before,
            "messages_after": after,
            "messages": messages,
        }

    def _context(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        index: int,
        *,
        radius: int = 1,
    ) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT role, idx, content
            FROM messages_fts
            WHERE session_id = ? AND CAST(idx AS INTEGER) BETWEEN ? AND ?
            ORDER BY CAST(idx AS INTEGER)
            """,
            (session_id, index - radius, index + radius),
        ).fetchall()
        return [
            {
                "role": row["role"],
                "idx": row["idx"],
                "message_index": row["idx"],
                "content": _truncate(str(row["content"] or ""), 350),
            }
            for row in rows
        ]

    def _session_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "session_id": str(row["session_id"]),
            "summary": _truncate(str(row["summary"] or ""), 2200),
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "n_messages": row["n_messages"],
        }

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    started_at REAL,
                    ended_at REAL,
                    summary TEXT,
                    n_messages INTEGER
                )
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                    session_id UNINDEXED,
                    role UNINDEXED,
                    idx UNINDEXED,
                    content,
                    tokenize='unicode61'
                )
                """
            )


def sanitize_fts5_query(query: str) -> str:
    """Make a model/user query safe enough for SQLite FTS5."""

    query = query.strip()
    if not query:
        return ""
    phrases: list[str] = []

    def save_phrase(match: re.Match[str]) -> str:
        phrases.append(match.group(0))
        return f" __PHRASE_{len(phrases) - 1}__ "

    working = re.sub(r'"[^"]+"', save_phrase, query)
    working = re.sub(r"[+{}()\"^]", " ", working)
    working = re.sub(r"\*{2,}", "*", working)
    tokens = working.split()
    while tokens and tokens[0].upper() in {"AND", "OR", "NOT"}:
        tokens.pop(0)
    while tokens and tokens[-1].upper() in {"AND", "OR", "NOT"}:
        tokens.pop()

    rendered: list[str] = []
    for token in tokens:
        if token.startswith("__PHRASE_") and token.endswith("__"):
            rendered.append(token)
            continue
        upper = token.upper()
        if upper in {"AND", "OR", "NOT"}:
            rendered.append(upper)
            continue
        token = token.lstrip("*")
        if not token:
            continue
        if token.endswith("*") and token != "*":
            rendered.append(token)
            continue
        if any(ch in token for ch in "._-:/"):
            rendered.append('"' + token.replace('"', '""') + '"')
        else:
            rendered.append(token)
    output = " ".join(rendered)
    for index, phrase in enumerate(phrases):
        output = output.replace(f"__PHRASE_{index}__", phrase)
    return output.strip()


def _bounded_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        number = int(value or default)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def _invisible_error(content: str) -> str | None:
    match = INVISIBLE_RE.search(content)
    if match is None:
        return None
    return (
        "Blocked: content contains invisible unicode "
        f"U+{ord(match.group(0)):04X} (possible injection)."
    )


def _separator_error(content: str) -> str | None:
    if SEPARATOR_RE.search(content) is None:
        return None
    return "Content cannot contain a standalone '---' memory entry separator."


def _unique_match(entries: list[str], needle: str) -> int | None:
    matches = [index for index, entry in enumerate(entries) if needle in entry]
    if len(matches) == 1:
        return matches[0]
    if matches and len({entries[index] for index in matches}) == 1:
        return matches[0]
    return None


def _chars(entries: list[str]) -> int:
    return len(ENTRY_SEPARATOR.join(entries))


def _usage(entries: list[str], limit: int) -> str:
    used = _chars(entries)
    pct = min(100, round(100 * used / limit)) if limit else 100
    return f"{pct}% - {used}/{limit} chars"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 14)] + "...[truncated]"
