"""GDPVal solver web tools.

These are small Simple Agent Lab-native equivalents of the web tools used by
GDPVal solvers: Jina Search for search and Jina Reader for known-URL fetches.
They intentionally do not depend on SWALM.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from simple_agent_lab.tools import AgentTool, ToolResult, text_result

_JINA_SEARCH_DEFAULT_ENDPOINT = "https://s.jina.ai/"
_JINA_SEARCH_DEFAULT_MAX_RESULTS = 5
_JINA_SEARCH_MAX_RESULTS = 10
_JINA_SEARCH_CONTENT_PREVIEW_CHARS = 2000
_JINA_SEARCH_TIMEOUT_SECONDS = 60
_JINA_SEARCH_MAX_ATTEMPTS = 3
_JINA_DEFAULT_ENDPOINT = "https://r.jina.ai/"
_WEBFETCH_ARTIFACT_ROOT = ".webfetch_cache"
_WEBFETCH_SCHEMA_VERSION = 1
_WEBFETCH_DEFAULT_PREVIEW_CHARS = 20_000
_WEBFETCH_MAX_PREVIEW_CHARS = 100_000
_WEBFETCH_DEFAULT_TIMEOUT_SECONDS = 30


def make_gdpval_web_tools(*, workdir: str | Path) -> tuple[AgentTool, ...]:
    """Return optional GDPVal solver web tools."""

    workspace = Path(workdir).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    return (
        _make_web_search_tool(),
        _make_web_fetch_tool(workspace),
    )


def _make_web_search_tool() -> AgentTool:
    def execute(call_id: str, args: dict[str, Any], abort, on_update) -> ToolResult:
        del call_id, on_update
        if abort():
            return text_result(_json_result(ok=False, error="WebSearch aborted"))
        query = str(args.get("query") or "").strip()
        if not query:
            return text_result(
                _json_result(ok=False, error="query must not be empty."),
                is_error=True,
            )
        requested_limit = args.get("max_results")
        if requested_limit is None:
            requested_limit = args.get("num_results")
        limit = _bounded_int(
            requested_limit,
            default=_JINA_SEARCH_DEFAULT_MAX_RESULTS,
            minimum=1,
            maximum=_JINA_SEARCH_MAX_RESULTS,
        )
        allowed_domains = _string_list(args.get("allowed_domains"))
        blocked_domains = _string_list(args.get("blocked_domains"))
        output = _jina_search(
            query=query,
            limit=limit,
            gl=_optional_string(args.get("gl")),
            hl=_optional_string(args.get("hl")),
            location=_optional_string(args.get("location")),
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
        )
        return text_result(output)

    return AgentTool(
        name="WebSearch",
        description=(
            "Search the web using Jina Search and return concise structured "
            "results with source URLs and short extracted-content previews. "
            "Avoid over-constraining queries with many exact-quoted terms; if "
            "a search returns no results, retry with fewer quotes, fewer "
            "required terms, or a targeted site:domain query."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query.",
                    "minLength": 2,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of organic results to return. Defaults to 5, capped at 10.",
                    "default": 5,
                },
                "num_results": {
                    "type": "integer",
                    "description": "Alias for max_results.",
                },
                "gl": {
                    "type": "string",
                    "description": "Optional country code, such as us or cn.",
                },
                "hl": {
                    "type": "string",
                    "description": "Optional language code, such as en or zh-cn.",
                },
                "location": {
                    "type": "string",
                    "description": "Optional search location.",
                },
                "allowed_domains": {
                    "type": "array",
                    "description": 'Optional domains to prefer/include, e.g. ["python.org"].',
                    "items": {"type": "string"},
                },
                "blocked_domains": {
                    "type": "array",
                    "description": "Optional domains to filter out of returned results.",
                    "items": {"type": "string"},
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        execute=execute,
        timeout_seconds=200.0,
    )


def _make_web_fetch_tool(workspace: Path) -> AgentTool:
    def execute(call_id: str, args: dict[str, Any], abort, on_update) -> ToolResult:
        del call_id, on_update
        if abort():
            return text_result(_json_result(ok=False, error="WebFetch aborted"))
        url = str(args.get("url") or "").strip()
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return text_result(
                _json_result(
                    ok=False,
                    error="url must be an absolute http or https URL.",
                ),
                is_error=True,
            )
        limit = _bounded_int(
            args.get("max_chars"),
            default=_WEBFETCH_DEFAULT_PREVIEW_CHARS,
            minimum=1000,
            maximum=_WEBFETCH_MAX_PREVIEW_CHARS,
        )
        timeout = _bounded_int(
            args.get("timeout_seconds"),
            default=_WEBFETCH_DEFAULT_TIMEOUT_SECONDS,
            minimum=1,
            maximum=120,
        )
        try:
            result = _jina_fetch_and_write_artifact(
                url,
                workspace=workspace,
                timeout=timeout,
                preview_limit=limit,
            )
        except Exception as exc:
            return text_result(
                _json_result(ok=False, error=f"{type(exc).__name__}: {exc}"),
                is_error=True,
            )
        return text_result(_json_result(**result))

    return AgentTool(
        name="WebFetch",
        description=(
            "Fetch a public web URL through Jina Reader, save the full extracted "
            "content as a workspace artifact, and return metadata, artifact "
            "paths, and a bounded preview."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "HTTP or HTTPS URL to fetch.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum preview characters to return. Full content is saved as an artifact.",
                    "default": _WEBFETCH_DEFAULT_PREVIEW_CHARS,
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Optional request timeout in seconds.",
                },
            },
            "required": ["url"],
            "additionalProperties": False,
        },
        execute=execute,
        timeout_seconds=125.0,
    )


def _jina_search(
    *,
    query: str,
    limit: int,
    gl: str | None,
    hl: str | None,
    location: str | None,
    allowed_domains: list[str],
    blocked_domains: list[str],
) -> str:
    del gl, hl
    api_key = os.environ.get("JINA_API_KEY", "").strip()
    if not api_key:
        return _json_result(
            ok=False,
            error="Missing JINA_API_KEY. Jina Search requires it.",
        )

    base_query = f"{query} {location}".strip() if location else query
    search_query = base_query
    if allowed_domains:
        domain_filter = " OR ".join(f"site:{domain}" for domain in allowed_domains)
        if domain_filter:
            search_query = f"{base_query} ({domain_filter})"

    endpoint = os.environ.get(
        "JINA_SEARCH_ENDPOINT",
        _JINA_SEARCH_DEFAULT_ENDPOINT,
    ).strip()
    search_url = endpoint.rstrip("/") + "/" + urllib.parse.quote(search_query, safe="")
    request = urllib.request.Request(
        search_url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "curl/8.7.1",
        },
        method="GET",
    )
    raw = ""
    last_timeout_error: BaseException | None = None
    for attempt in range(1, _JINA_SEARCH_MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(
                request,
                timeout=_JINA_SEARCH_TIMEOUT_SECONDS,
            ) as response:
                raw = response.read().decode("utf-8", errors="replace")
            break
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            return _json_result(
                ok=False,
                status=exc.code,
                error=_redact_secrets(message[:2000]),
            )
        except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
            last_timeout_error = exc
            if attempt >= _JINA_SEARCH_MAX_ATTEMPTS:
                return _json_result(
                    ok=False,
                    attempts=attempt,
                    error=(
                        "Jina Search request timed out or failed after "
                        f"{_JINA_SEARCH_TIMEOUT_SECONDS}s "
                        f"after {attempt} attempts: {exc}."
                    ),
                )
            time.sleep(min(attempt, 3))
        except Exception as exc:
            return _json_result(ok=False, error=f"{type(exc).__name__}: {exc}")
    if not raw and last_timeout_error is not None:
        return _json_result(
            ok=False,
            attempts=_JINA_SEARCH_MAX_ATTEMPTS,
            error=(
                "Jina Search request timed out or failed after "
                f"{_JINA_SEARCH_TIMEOUT_SECONDS}s "
                f"after {_JINA_SEARCH_MAX_ATTEMPTS} attempts: "
                f"{last_timeout_error}."
            ),
        )

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _json_result(
            ok=False,
            error="Jina Search response was not valid JSON.",
            response_preview=raw[:1000],
        )

    result = _normalize_jina_search_response(data, limit)
    if blocked_domains:
        result["organic"] = [
            item
            for item in result.get("organic", [])
            if not _domain_matches(str(item.get("link") or ""), blocked_domains)
        ][:limit]
        result["news"] = [
            item
            for item in result.get("news", [])
            if not _domain_matches(str(item.get("link") or ""), blocked_domains)
        ][: min(limit, 5)]
    if allowed_domains:
        result["organic"] = [
            item
            for item in result.get("organic", [])
            if _domain_matches(str(item.get("link") or ""), allowed_domains)
        ][:limit]
        result["news"] = [
            item
            for item in result.get("news", [])
            if _domain_matches(str(item.get("link") or ""), allowed_domains)
        ][: min(limit, 5)]
    result["ok"] = True
    result["query"] = query
    result["provider"] = "jina"
    result["search_url"] = search_url
    return _json_result(**result)


def _normalize_jina_search_response(data: Any, limit: int) -> dict[str, Any]:
    items = data.get("data", []) if isinstance(data, dict) else data
    if not isinstance(items, list):
        items = []

    organic_results: list[dict[str, Any]] = []
    for item in items[:limit]:
        if not isinstance(item, dict):
            continue
        link = str(item.get("url") or item.get("link") or "")
        content = str(item.get("content") or "")
        content_preview, content_truncated = _truncate_text(
            content,
            _JINA_SEARCH_CONTENT_PREVIEW_CHARS,
        )
        description = str(item.get("description") or item.get("snippet") or "")
        organic_results.append(
            {
                "title": str(item.get("title") or ""),
                "link": link,
                "snippet": description or _first_nonempty_line(content_preview),
                "date": str(item.get("date") or ""),
                "source": str(item.get("source") or _hostname(link)),
                "content_preview": content_preview,
                "content_chars": len(content),
                "content_truncated": content_truncated,
            }
        )

    normalized: dict[str, Any] = {"organic": organic_results, "news": []}
    if isinstance(data, dict):
        normalized["status"] = data.get("status")
        if data.get("code") is not None:
            normalized["code"] = data.get("code")
    return normalized


def _jina_fetch_and_write_artifact(
    url: str,
    *,
    workspace: Path,
    timeout: int,
    preview_limit: int,
) -> dict[str, Any]:
    endpoint = os.environ.get("JINA_ENDPOINT", _JINA_DEFAULT_ENDPOINT).strip()
    fetch_url = endpoint.rstrip("/") + "/" + url
    headers = {
        "Accept": "text/markdown, text/plain;q=0.9, */*;q=0.1",
        "User-Agent": "curl/8.7.1",
        "X-Return-Format": "markdown",
    }
    api_key = os.environ.get("JINA_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        request = urllib.request.Request(fetch_url, headers=headers, method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            status = getattr(response, "status", 200)
            response_content_type = _response_header(response, "Content-Type")
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Jina Reader HTTP {exc.code}: {_redact_secrets(message[:2000])}"
        ) from exc

    title, source_url, content = _parse_jina_reader_text(raw, fallback_url=url)
    artifact_id = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    artifact_dir = workspace / _WEBFETCH_ARTIFACT_ROOT / artifact_id
    raw_jina_path = artifact_dir / "jina_response.txt"
    content_path = artifact_dir / "content.md"
    metadata_path = artifact_dir / "metadata.json"

    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw_jina_path.write_text(raw, encoding="utf-8")
    content_path.write_text(content, encoding="utf-8")

    preview, was_truncated = _truncate_text(content, preview_limit)
    metadata = {
        "schema_version": _WEBFETCH_SCHEMA_VERSION,
        "id": artifact_id,
        "provider": "jina",
        "url": url,
        "source_url": source_url,
        "title": title,
        "status": status,
        "response_content_type": response_content_type,
        "fetched_at": int(time.time()),
        "content_chars": len(content),
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "artifact_path": _relative_to(workspace, artifact_dir),
        "content_path": _relative_to(workspace, content_path),
        "raw_jina_path": _relative_to(workspace, raw_jina_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = (
        f"Fetched {len(content)} characters from {source_url}. "
        f"Full content is saved at {_relative_to(workspace, content_path)}."
    )
    return {
        "ok": True,
        "status": status,
        "url": source_url,
        "title": title,
        "summary": summary,
        "content": preview,
        "truncated": was_truncated,
        "content_chars": len(content),
        "preview_chars": len(preview),
        "provider": "jina",
        "artifact": {
            "id": artifact_id,
            "path": _relative_to(workspace, artifact_dir),
            "content_path": _relative_to(workspace, content_path),
            "metadata_path": _relative_to(workspace, metadata_path),
            "raw_jina_path": _relative_to(workspace, raw_jina_path),
        },
        "content_path": str(content_path),
        "metadata_path": str(metadata_path),
    }


def _parse_jina_reader_text(raw: str, *, fallback_url: str) -> tuple[str, str, str]:
    title = ""
    source_url = fallback_url
    lines = raw.splitlines()
    content_start = 0
    for index, line in enumerate(lines[:20]):
        if line.startswith("Title:"):
            title = line.removeprefix("Title:").strip()
            content_start = max(content_start, index + 1)
        elif line.startswith("URL Source:"):
            source_url = line.removeprefix("URL Source:").strip() or source_url
            content_start = max(content_start, index + 1)
        elif line.startswith("Markdown Content:"):
            content_start = index + 1
            break
    content = "\n".join(lines[content_start:]).strip() if content_start else raw.strip()
    if not title:
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                title = stripped.lstrip("#").strip()
                break
    return title, source_url, content


def _response_header(response: Any, name: str) -> str:
    headers = getattr(response, "headers", None)
    if headers is not None:
        value = headers.get(name)
        if value:
            return str(value)
    info = getattr(response, "info", None)
    if callable(info):
        value = info().get(name)
        if value:
            return str(value)
    return ""


def _domain_matches(url: str, domains: list[str]) -> bool:
    if not domains:
        return True
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)


def _hostname(url: str) -> str:
    return (urllib.parse.urlparse(url).hostname or "").lower()


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:500]
    return ""


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        str(item).strip().lower().lstrip(".")
        for item in value
        if isinstance(item, str) and item.strip()
    ]


def _optional_string(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, minimum), maximum)


def _truncate_text(text: str, limit: int) -> tuple[str, bool]:
    text = text or ""
    if len(text) <= limit:
        return text, False
    marker = f"\n...[truncated {len(text) - limit} chars from middle]...\n"
    keep = max(limit - len(marker), 0)
    head = keep // 2
    tail = keep - head
    return text[:head] + marker + (text[-tail:] if tail else ""), True


def _relative_to(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _redact_secrets(text: str) -> str:
    redacted = text or ""
    for name in ("JINA_API_KEY",):
        value = os.environ.get(name, "")
        if value:
            redacted = redacted.replace(value, f"<redacted {name}>")
    return redacted


def _json_result(**kwargs: Any) -> str:
    return json.dumps(kwargs, ensure_ascii=False, indent=2, default=str)
