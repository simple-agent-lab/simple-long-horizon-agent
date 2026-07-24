#!/usr/bin/env python3
"""Serve the Observatory trace viewer with a polling API over eval outputs.

Run from the repo root (or anywhere; paths resolve to the trace-viewer
folder by default):

    python3 studio/trace-viewer/serve.py
    python3 studio/trace-viewer/serve.py --dir evals/out --port 8765

The HTTP server is stdlib only (no dependencies). Endpoints:

    GET  /                      static index.html
    GET  /<static-asset>        any file under studio/trace-viewer/
    GET  /api/info              { project_root, scan_dir, static_dir }
    GET  /api/traces            list of files under --dir with shape detection
    GET  /api/trace?path=&line= return one parsed record (line index for JSONL)
    GET  /api/records?path=     return ALL parsed records of a file (small JSONL)

The viewer polls /api/traces every few seconds so new evals show up live.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


# Only bother compressing payloads above this size; below it the wire savings
# are smaller than the CPU + latency cost of zipping.
GZIP_MIN_BYTES = 4 * 1024
# Level 6 (gzip's default) sits at the knee of the level/ratio curve for the
# JSON traces this viewer serves. On a 15MB trajectory it cuts the body from
# 15.2MB → 2.5MB (16%) in ~150ms, vs level 1's 3.4MB in ~50ms. The smaller
# wire payload more than pays back the extra ~100ms of server CPU once the
# browser becomes the bottleneck draining the response body.
GZIP_LEVEL = 6
GZIPPABLE_PREFIXES = (
    "application/json",
    "application/x-ndjson",
    "text/",
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_SCAN_DIR = PROJECT_ROOT / "evals" / "out"

MAX_PEEK_BYTES = 96 * 1024
TRAJECTORY_SCHEMA_PREFIX = "simple-agent-lab.trajectory"
SCANNED_EXTENSIONS = {".jsonl", ".json"}
# Don't waste a stat() walking into these
IGNORED_DIRS = {"wheelhouse", "docker-config", "__pycache__", ".git"}

# When the first record is too large to parse from the peek window we fall
# back to text-level fingerprinting on the head bytes.
_SCHEMA_RE = re.compile(r'"schema"\s*:\s*"([^"]+)"')
_TRACE_ID_RE = re.compile(r'"trace_id"\s*:\s*"([^"]+)"')
_TASK_RE = re.compile(r'"task"\s*:\s*"((?:[^"\\]|\\.){0,200})"')
_INSTANCE_RE = re.compile(r'"instance_id"\s*:\s*"([^"]+)"')


def detect_record_kind(record: object) -> str:
    """Classify one parsed record by its visible shape."""
    if not isinstance(record, dict):
        return "other"
    schema = record.get("schema") or ""
    if isinstance(schema, str) and schema.startswith(TRAJECTORY_SCHEMA_PREFIX):
        return "trajectory"
    if isinstance(record.get("events"), list):
        return "trajectory"
    if record.get("type") == "eval_result" or "scorer" in record:
        return "eval_result"
    if "model_patch" in record and "instance_id" in record:
        return "prediction"
    if "problem_statement" in record and "instance_id" in record:
        return "instance_metadata"
    return "other"


def _detect_kind_by_text(head_text: str) -> str:
    """Best-effort classifier for records too big to fully parse."""
    schema_match = _SCHEMA_RE.search(head_text)
    if schema_match and schema_match.group(1).startswith(TRAJECTORY_SCHEMA_PREFIX):
        return "trajectory"
    if '"events"' in head_text and '"messages"' in head_text:
        return "trajectory"
    if schema_match and schema_match.group(1).endswith("evaluation.v1"):
        return "eval_result"
    if '"model_patch"' in head_text and '"instance_id"' in head_text:
        return "prediction"
    if '"problem_statement"' in head_text and '"instance_id"' in head_text:
        return "instance_metadata"
    return "other"


def peek_first_record(path: Path) -> tuple[dict | None, dict, str | None]:
    """Parse the first record if it fits in the peek window.

    Returns ``(parsed_or_none, text_fingerprint, error_or_none)``. The
    fingerprint is always populated even when JSON parsing fails, so the
    caller can still surface schema / trace_id / task for very large records.
    """
    try:
        with path.open("rb") as f:
            chunk = f.read(MAX_PEEK_BYTES)
    except OSError as e:
        return None, {}, f"read error: {e}"
    if not chunk:
        return None, {}, "empty file"
    text = chunk.decode("utf-8", errors="replace")
    fingerprint: dict = {}
    for name, regex in (
        ("schema", _SCHEMA_RE),
        ("trace_id", _TRACE_ID_RE),
        ("instance_id", _INSTANCE_RE),
    ):
        m = regex.search(text)
        if m:
            fingerprint[name] = m.group(1)
    task_m = _TASK_RE.search(text)
    if task_m:
        try:
            fingerprint["task"] = json.loads(f'"{task_m.group(1)}"')
        except json.JSONDecodeError:
            fingerprint["task"] = task_m.group(1)
    try:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(text.lstrip())
        return obj, fingerprint, None
    except json.JSONDecodeError as e:
        return None, fingerprint, f"unparseable first record: {e.msg}"


def count_records(path: Path) -> int:
    """Count top-level JSON records in a JSONL file.

    Works for both single-line JSONL and pretty-printed records by counting
    lines whose first character is ``{`` (only top-level objects start at
    column 0 in an ``indent=2`` layout).

    Fast path: a single-line record (one trajectory is one line that can be many
    MB) has no newline in its head — return 1 WITHOUT reading the multi-MB body.
    Trajectory/sub-trace files are GBs in aggregate; reading them all in full on
    every 2.5s poll was the scan's dominant cost. Multi-line files (pretty
    JSONL, predictions) carry a newline in the head and are small, so the full
    line-count below is cheap for them.
    """
    try:
        with path.open("rb") as f:
            head = f.read(MAX_PEEK_BYTES)
            if not head:
                return 0
            if b"\n" not in head:
                # One record on one (possibly huge) line — don't read the rest.
                return 1
            n = 0
            f.seek(0)
            for line in f:
                if line[:1] == b"{":
                    n += 1
        return n
    except OSError:
        return 0


def _nth_json_object(text: str, n: int) -> dict | None:
    """Parse the *n*-th JSON object from *text* (0-based)."""
    decoder = json.JSONDecoder()
    idx = 0
    length = len(text)
    record_num = 0
    while idx < length:
        while idx < length and text[idx] in " \t\n\r":
            idx += 1
        if idx >= length:
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            break
        if record_num == n:
            return obj
        idx = end
        record_num += 1
    return None


def _nth_json_span(text: str, n: int) -> tuple[int, int] | None:
    """Return the ``(start, end)`` character offsets of the *n*-th JSON object."""
    decoder = json.JSONDecoder()
    idx = 0
    length = len(text)
    record_num = 0
    while idx < length:
        while idx < length and text[idx] in " \t\n\r":
            idx += 1
        if idx >= length:
            break
        start = idx
        try:
            _, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            break
        if record_num == n:
            return start, end
        idx = end
        record_num += 1
    return None


def _walk(root: Path):
    """rglob, but pruning IGNORED_DIRS so we don't peek into bulky asset dirs."""
    stack: list[Path] = [root]
    while stack:
        current = stack.pop()
        try:
            for child in current.iterdir():
                if child.is_dir():
                    if child.name in IGNORED_DIRS:
                        continue
                    stack.append(child)
                elif child.is_file():
                    yield child
        except OSError:
            continue


# Classified-entry cache keyed by path → (mtime, size, entry). A trace file is
# immutable once written (the only mutating file is a *running* eval's live
# trajectory, whose mtime/size change and so miss the cache and get re-read), so
# caching by (mtime, size) turns the every-2.5s poll from a full ~10 GB re-read
# into a cheap stat() per file. Without this, each poll re-classified every file
# — a 23 s scan that backed up behind itself and froze the UI.
_SCAN_CACHE: dict[str, tuple[float, int, dict]] = {}


def list_traces(scan_dir: Path, project_root: Path) -> list[dict]:
    if not scan_dir.exists() or not scan_dir.is_dir():
        return []
    results: list[dict] = []
    seen_paths: set[str] = set()
    for path in _walk(scan_dir):
        if path.suffix.lower() not in SCANNED_EXTENSIONS:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        path_key = str(path)
        seen_paths.add(path_key)
        cached = _SCAN_CACHE.get(path_key)
        if (
            cached is not None
            and cached[0] == stat.st_mtime
            and cached[1] == stat.st_size
        ):
            results.append(cached[2])
            continue
        first, fingerprint, peek_error = peek_first_record(path)
        if isinstance(first, dict):
            kind = detect_record_kind(first)
            schema = first.get("schema")
            trace_id = first.get("trace_id")
            task = first.get("task")
            instance_id = first.get("instance_id")
        elif first is not None:
            # Parsed, but the top-level record is a list/scalar, not an object
            # (e.g. a JSON-array file like the harness's patches.json). It's not
            # a viewer record; fall back to the text fingerprint for metadata.
            kind = detect_record_kind(first)
            schema = fingerprint.get("schema")
            trace_id = fingerprint.get("trace_id")
            task = fingerprint.get("task")
            instance_id = fingerprint.get("instance_id")
        else:
            head_text = _safe_head_text(path)
            kind = _detect_kind_by_text(head_text)
            schema = fingerprint.get("schema")
            trace_id = fingerprint.get("trace_id")
            task = fingerprint.get("task")
            instance_id = fingerprint.get("instance_id")
        record_count = count_records(path) if path.suffix.lower() == ".jsonl" else 1

        rel = _safe_relative(path, project_root)
        group_rel = _safe_relative(path.parent, scan_dir)
        # Group label: stay above the file's own directory so users see
        # "<run-name>/<instance-id>" rather than ten flat files all under "out/"
        group_label = str(group_rel) if str(group_rel) != "." else "(root)"
        run_id = _derive_run_id(path, scan_dir)
        # SWE-bench trajectories don't repeat instance_id inside the first
        # record, so fall back to the directory name when the peek missed it.
        # Only for the per-instance ``out/`` artifact — a sub-trace's parent is
        # ``sub`` (its instance dir is two up), and we WANT it to keep a None
        # instance_id so the sidebar labels it by filename (e.g. 00_worker.jsonl)
        # rather than repeating the instance under its run group.
        if not instance_id and run_id is not None and path.parent.name == "out":
            instance_id = path.parents[1].name

        entry = {
            "path": str(path),
            "rel_path": str(rel),
            "name": path.name,
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "mtime_iso": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(timespec="seconds"),
            "kind": kind,
            "schema": schema,
            "trace_id": trace_id,
            "task": task,
            "instance_id": instance_id,
            "record_count": record_count,
            "group": group_label,
            "run_id": run_id,
            "peek_error": peek_error,
        }
        _SCAN_CACHE[path_key] = (stat.st_mtime, stat.st_size, entry)
        results.append(entry)
    # Drop cache entries for files that have vanished since the last scan.
    if len(_SCAN_CACHE) > len(seen_paths):
        for stale in [k for k in _SCAN_CACHE if k not in seen_paths]:
            del _SCAN_CACHE[stale]
    results.sort(key=lambda e: (-e["mtime"], e["rel_path"]))
    return results


def _safe_relative(path: Path, base: Path) -> Path:
    try:
        return path.relative_to(base)
    except ValueError:
        return path


# Trajectories produced by the containerized SWE-bench runner land at
# ``<scan>/<...>/<run_id>/<instance_id>/out/{trajectory,trace}.jsonl`` (the run
# directory layout from ``simple_agent_lab.evals.runner``). Extract ``run_id``
# from that shape so the viewer can aggregate the per-instance files back into a
# single experiment row.
_RUN_ARTIFACT_NAMES = {"trajectory.jsonl", "trace.jsonl"}
# A run's aggregate verdicts land at ``<run_id>/eval_results.jsonl`` (written by
# evaluate_predictions.run_scoped_eval_results_path), so the viewer can bind
# each judge to the run that produced it and join by (run_id, instance_id).
_RUN_EVAL_NAME = "eval_results.jsonl"


def _derive_run_id(path: Path, scan_dir: Path) -> str | None:
    """Return the run_id for a per-run artifact, or None if the path doesn't fit.

    Trajectories live at ``<run_id>/<instance_id>/out/{trajectory,trace}.jsonl``;
    workflow sub-agent traces live one level deeper at
    ``<run_id>/<instance_id>/out/sub/*.jsonl``; the run's aggregate verdicts live
    at ``<run_id>/eval_results.jsonl``. All resolve to the same run_id (the run
    directory name) so the viewer groups them together — in particular the
    sub-traces nest under their run instead of a stray ``…/out/sub`` group.
    Legacy suite-level ``*_eval_results.jsonl`` files don't match and return None.
    """
    parents = path.parents
    if path.name in _RUN_ARTIFACT_NAMES:
        # Need ``<run_id>/<instance_id>/out/`` above the file.
        if len(parents) < 3 or parents[0].name != "out":
            return None
        run_dir = parents[2]
    elif parents and parents[0].name == "sub":
        # Sub-agent trace: ``<run_id>/<instance_id>/out/sub/<name>.jsonl``.
        if len(parents) < 4 or parents[1].name != "out":
            return None
        run_dir = parents[3]
    elif path.name == _RUN_EVAL_NAME:
        # Need ``<run_id>/`` directly above the file.
        if len(parents) < 1:
            return None
        run_dir = parents[0]
    else:
        return None
    try:
        run_dir.relative_to(scan_dir)
    except ValueError:
        return None
    if run_dir == scan_dir:
        return None
    return run_dir.name


def _safe_head_text(path: Path) -> str:
    try:
        with path.open("rb") as f:
            return f.read(MAX_PEEK_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return ""


def read_trace_record(path: Path, line_index: int = 0) -> dict | None:
    """Return one parsed record from path.

    For plain .json: the whole file (line_index ignored).
    For .jsonl: the record at line_index (0-based).  Supports both
    single-line JSONL and pretty-printed (``indent=2``) records.
    """
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    text = path.read_text(encoding="utf-8")
    return _nth_json_object(text, line_index)


def read_all_records(path: Path, limit: int = 5000) -> list[dict]:
    """Return every top-level JSON object in a .json/.jsonl file (capped).

    Used by /api/records for small aggregate files (eval results, predictions)
    where the viewer wants the whole set in one request rather than walking
    line indices. The cap is a runaway guard, not an expected boundary.
    """
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [r for r in data[:limit] if isinstance(r, dict)]
        return [data] if isinstance(data, dict) else []
    text = path.read_text(encoding="utf-8")
    out: list[dict] = []
    decoder = json.JSONDecoder()
    idx, length = 0, len(text)
    while idx < length and len(out) < limit:
        while idx < length and text[idx] in " \t\n\r":
            idx += 1
        if idx >= length:
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            out.append(obj)
        idx = end
    return out


def read_trace_record_bytes(path: Path, line_index: int = 0) -> bytes | None:
    """Fast path: return one record as raw JSON bytes.

    Uses ``raw_decode`` to locate the byte boundaries of the *n*-th
    JSON object, then slices the original text — skipping a full
    ``json.dumps`` round-trip.  Works for both single-line JSONL and
    pretty-printed records.
    """
    if path.suffix.lower() == ".json":
        try:
            return path.read_bytes()
        except OSError:
            return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    # A v5 trajectory is a header line + one line per event — the whole file is
    # one trace, so return all of it. Only an explicit line_index > 0 (selecting
    # one record in a rare multi-record file) slices to a single record.
    if line_index == 0:
        return text.encode("utf-8")
    span = _nth_json_span(text, line_index)
    if span is None:
        return None
    start, end = span
    return text[start:end].encode("utf-8")


CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".jsonl": "application/x-ndjson; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".md": "text/markdown; charset=utf-8",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
}


class TraceViewerHandler(BaseHTTPRequestHandler):
    project_root: Path = PROJECT_ROOT
    scan_dir: Path = DEFAULT_SCAN_DIR
    static_dir: Path = SCRIPT_DIR
    server_version = "Observatory/1.0"
    # HTTP/1.1 enables persistent connections + larger TCP send windows so
    # browsers don't pay an HTTP/1.0-tax on big bodies.
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        sys.stderr.write(
            f"{self.address_string()} - [{self.log_date_time_string()}] "
            f"{format % args}\n"
        )

    def _client_accepts_gzip(self) -> bool:
        accept = self.headers.get("Accept-Encoding") or ""
        # Token-list match without parsing q-values — good enough for browsers.
        return "gzip" in accept.lower()

    def _maybe_gzip(self, body: bytes, ctype: str) -> tuple[bytes, bool]:
        """Return (body, used_gzip). Compress only when the client opted in,
        the payload is large enough to matter, and the type is text-ish."""
        if len(body) < GZIP_MIN_BYTES:
            return body, False
        if not self._client_accepts_gzip():
            return body, False
        if not any(ctype.startswith(p) for p in GZIPPABLE_PREFIXES):
            return body, False
        return gzip.compress(body, compresslevel=GZIP_LEVEL), True

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        ctype: str,
        *,
        cache: str = "no-store",
        extra_headers: list[tuple[str, str]] | None = None,
    ) -> None:
        """Send a raw byte body, applying gzip when the client supports it.

        Centralizes the response shape so /api/trace, /api/traces, and the
        static handler all get the same compression behavior. ``extra_headers``
        lets callers attach response-specific headers (e.g. ``X-Trace-Mtime``
        on ``/api/trace``) without re-implementing the gzip/length plumbing.
        """
        encoded, gz = self._maybe_gzip(body, ctype)
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        if gz:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", cache)
        if extra_headers:
            # Make CORS-style readers see the custom headers we set. Without
            # this the browser's response object hides X-Trace-Mtime.
            self.send_header(
                "Access-Control-Expose-Headers",
                ", ".join(name for name, _ in extra_headers),
            )
            for name, value in extra_headers:
                self.send_header(name, value)
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, status: int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _send_static(self, rel_path: str) -> None:
        if rel_path in ("", "/"):
            rel_path = "index.html"
        candidate = (self.static_dir / rel_path.lstrip("/")).resolve()
        static_root = self.static_dir.resolve()
        try:
            candidate.relative_to(static_root)
        except ValueError:
            self.send_error(403, "Forbidden")
            return
        if not candidate.exists() or not candidate.is_file():
            self.send_error(404, "Not found")
            return
        ext = candidate.suffix.lower()
        ctype = CONTENT_TYPES.get(ext, "application/octet-stream")
        try:
            data = candidate.read_bytes()
        except OSError as e:
            self.send_error(500, f"read error: {e}")
            return
        self._send_bytes(200, data, ctype)

    def do_GET(self) -> None:  # noqa: N802 (HTTP handler API)
        url = urlparse(self.path)
        path = url.path

        if path == "/api/info":
            return self._send_json(
                200,
                {
                    "project_root": str(self.project_root),
                    "scan_dir": str(self.scan_dir),
                    "static_dir": str(self.static_dir),
                    "scan_dir_exists": self.scan_dir.exists(),
                },
            )

        if path == "/api/traces":
            return self._send_json(
                200,
                {
                    "scan_dir": str(self.scan_dir),
                    "scanned_at": datetime.now(tz=timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    "traces": list_traces(self.scan_dir, self.project_root),
                },
            )

        if path == "/api/trace":
            query = parse_qs(url.query)
            raw_path = unquote(query.get("path", [""])[0])
            try:
                line = int(query.get("line", ["0"])[0])
            except ValueError:
                line = 0
            if not raw_path:
                return self._send_json(400, {"error": "missing path"})
            target = Path(raw_path).resolve()
            allowed = self.project_root.resolve()
            try:
                target.relative_to(allowed)
            except ValueError:
                return self._send_json(403, {"error": "path outside project root"})
            if not target.exists() or not target.is_file():
                return self._send_json(404, {"error": "not found"})
            try:
                raw_bytes = read_trace_record_bytes(target, line_index=line)
            except OSError as e:
                return self._send_json(400, {"error": f"failed to read record: {e}"})
            if raw_bytes is None:
                return self._send_json(
                    404, {"error": f"line {line} not found in {raw_path}"}
                )
            # Attach the file's current mtime so the viewer can decide whether
            # a live re-poll actually needs to re-render. This is the same
            # mtime exposed by /api/traces; keeping both endpoints in sync
            # means the client never sees stale freshness from a cached
            # /api/traces row.
            try:
                mtime_value = target.stat().st_mtime
                mtime_iso = datetime.fromtimestamp(
                    mtime_value, tz=timezone.utc
                ).isoformat(timespec="seconds")
                extra = [
                    ("X-Trace-Mtime", f"{mtime_value:.6f}"),
                    ("X-Trace-Mtime-Iso", mtime_iso),
                ]
            except OSError:
                extra = None
            # Fast path: hand the raw JSON bytes to _send_bytes, which gzips
            # when the client supports it (15MB JSON → ~1.5MB on the wire).
            # We skip json.loads/dumps entirely; the viewer already trusts
            # this file (it appears in /api/traces with a detected shape).
            self._send_bytes(
                200,
                raw_bytes,
                "application/json; charset=utf-8",
                extra_headers=extra,
            )
            return

        if path == "/api/raw":
            # The sibling provider raw pool for a trajectory, so the viewer can
            # auto-resolve `{raw_ref}` pointers (real request/response) without
            # the user hand-loading `*.raw.jsonl`. Missing sibling (e.g. a
            # fake/oracle run with no real calls) is not an error — empty pool.
            query = parse_qs(url.query)
            raw_path = unquote(query.get("path", [""])[0])
            if not raw_path:
                return self._send_json(400, {"error": "missing path"})
            sidecar = Path(raw_path + ".raw.jsonl").resolve()
            allowed = self.project_root.resolve()
            try:
                sidecar.relative_to(allowed)
            except ValueError:
                return self._send_json(403, {"error": "path outside project root"})
            if not sidecar.exists() or not sidecar.is_file():
                return self._send_json(200, {"pool": []})
            try:
                pool = read_all_records(sidecar)
            except (OSError, ValueError, json.JSONDecodeError) as e:
                return self._send_json(400, {"error": f"failed to read raw pool: {e}"})
            return self._send_json(200, {"pool": pool})

        if path == "/api/records":
            query = parse_qs(url.query)
            raw_path = unquote(query.get("path", [""])[0])
            if not raw_path:
                return self._send_json(400, {"error": "missing path"})
            target = Path(raw_path).resolve()
            allowed = self.project_root.resolve()
            try:
                target.relative_to(allowed)
            except ValueError:
                return self._send_json(403, {"error": "path outside project root"})
            if not target.exists() or not target.is_file():
                return self._send_json(404, {"error": "not found"})
            try:
                records = read_all_records(target)
                mtime_value = target.stat().st_mtime
            except (OSError, ValueError, json.JSONDecodeError) as e:
                return self._send_json(400, {"error": f"failed to read records: {e}"})
            return self._send_json(
                200,
                {
                    "path": raw_path,
                    "count": len(records),
                    "mtime": mtime_value,
                    "records": records,
                },
            )

        return self._send_static(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the Observatory trace viewer with a live eval-outputs API.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument(
        "--dir",
        default=str(DEFAULT_SCAN_DIR),
        help=f"Directory to scan recursively for eval outputs "
        f"(default: {DEFAULT_SCAN_DIR})",
    )
    parser.add_argument(
        "--root",
        default=str(PROJECT_ROOT),
        help=f"Allowed root for /api/trace path lookups (default: {PROJECT_ROOT})",
    )
    args = parser.parse_args()

    scan_dir = Path(args.dir).resolve()
    project_root = Path(args.root).resolve()

    TraceViewerHandler.scan_dir = scan_dir
    TraceViewerHandler.project_root = project_root

    httpd = ThreadingHTTPServer((args.host, args.port), TraceViewerHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"observatory · serving  {SCRIPT_DIR}", file=sys.stderr)
    print(f"observatory · scanning {scan_dir}", file=sys.stderr)
    print(f"observatory · open     {url}", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nobservatory · shutting down", file=sys.stderr)
        httpd.shutdown()


if __name__ == "__main__":
    main()
