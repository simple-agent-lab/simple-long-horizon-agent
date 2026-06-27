"""Jina tools: web search and URL fetch via Jina's free reader endpoints.

Two small tools that give an agent eyes on the open web without adding a
heavyweight HTTP dependency:

* ``jina_search`` queries ``https://s.jina.ai`` and returns the top results as
  clean markdown (title, url, and a content snippet per hit).
* ``jina_fetch`` reads one URL through ``https://r.jina.ai`` (the Jina Reader)
  and returns the page rendered as clean markdown — the boilerplate stripped,
  JavaScript executed server-side — which is far cheaper to put in context than
  raw HTML.

Both endpoints work anonymously (subject to a low rate limit). Set the
``JINA_API_KEY`` environment variable, or pass ``api_key=`` to the factory, to
authenticate with a higher quota; the factory argument wins over the env var.

Design mirrors the other tools in this package: a ``make_*_tool(...)`` factory
returns an ``AgentTool``, the pure request-building and response-shaping helpers
stay free of I/O so they unit-test without a network, and the one impure step —
the actual HTTP GET — sits behind an injectable ``http`` seam (default:
stdlib ``urllib``) so tests pass a fake instead of hitting the network.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import quote, urlencode, urlsplit

from . import (
    AbortFlag,
    AgentTool,
    ToolExecutionMode,
    ToolResult,
    ToolUpdateFn,
    coerce_int,
    text_result,
)


JINA_SEARCH_TOOL_NAME = "jina_search"
JINA_FETCH_TOOL_NAME = "jina_fetch"

JINA_SEARCH_ENDPOINT = "https://s.jina.ai/"
JINA_READER_ENDPOINT = "https://r.jina.ai/"

# The reader/search responses can be large (a full page rendered to markdown).
# Cap what reaches the model so one fetch cannot blow the context budget; the
# tail carries an actionable note so the model knows the view was cut short.
DEFAULT_MAX_OUTPUT_CHARS = 20_000
DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_TIMEOUT_SECONDS = 120.0

# Jina's search default is ~5 hits; allow the model to ask for fewer/more.
DEFAULT_SEARCH_RESULTS = 5
MAX_SEARCH_RESULTS = 20

_USER_AGENT = "simple-agent-lab/jina-tool"


@dataclass(frozen=True)
class HttpResponse:
    """A minimal HTTP response: status code plus the decoded text body.

    This is the contract of the injectable ``http`` seam, deliberately tiny so
    a test fake is a one-liner and the default urllib implementation maps onto
    it without leaking ``urllib`` types into the tool logic.
    """

    status: int
    text: str


# (url, headers, timeout_seconds) -> HttpResponse
JinaHttp = Callable[[str, dict[str, str], float], HttpResponse]


def make_jina_search_tool(
    *,
    api_key: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    default_results: int = DEFAULT_SEARCH_RESULTS,
    http: JinaHttp | None = None,
    execution_mode: ToolExecutionMode = "parallel",
) -> AgentTool:
    """Return an ``AgentTool`` that web-searches via ``s.jina.ai``.

    ``api_key`` (or, if omitted, the ``JINA_API_KEY`` env var read at call time)
    authenticates for a higher quota. ``http`` injects the HTTP seam for tests;
    the default uses stdlib ``urllib``.
    """

    _validate_common(timeout_seconds, max_output_chars)
    if not 1 <= default_results <= MAX_SEARCH_RESULTS:
        raise ValueError(
            f"default_results must be in 1..{MAX_SEARCH_RESULTS}, got {default_results}"
        )
    do_http = http or _urllib_http

    def execute(
        call_id: str,
        args: dict[str, Any],
        abort: AbortFlag,
        on_update: ToolUpdateFn | None,
    ) -> ToolResult:
        del call_id, on_update
        if abort():
            return text_result("Jina search aborted before start.", is_error=True)

        query = str(args.get("query", "")).strip()
        if not query:
            return text_result(
                "Missing required jina_search argument: query.", is_error=True
            )
        try:
            num_results = _coerce_optional_count(
                "num_results",
                args.get("num_results"),
                default=default_results,
                maximum=MAX_SEARCH_RESULTS,
            )
        except ValueError as exc:
            return text_result(f"Invalid jina_search argument: {exc}", is_error=True)

        url, headers = build_search_request(
            query, api_key=_resolve_api_key(api_key), num_results=num_results
        )
        return _run_request(
            do_http,
            url=url,
            headers=headers,
            timeout_seconds=timeout_seconds,
            max_output_chars=max_output_chars,
            label=f"search {query!r}",
            details={"query": query, "num_results": num_results},
        )

    return AgentTool(
        name=JINA_SEARCH_TOOL_NAME,
        description=(
            "Search the web and get the top results as clean markdown (title, "
            "URL, and a content snippet per result). Use this to find current "
            "information or relevant pages; follow up with `jina_fetch` to read "
            "a result in full."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "num_results": {
                    "type": "number",
                    "description": (
                        "Optional number of results to return "
                        f"(1..{MAX_SEARCH_RESULTS}, default {default_results})."
                    ),
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        execute=execute,
        execution_mode=execution_mode,
        timeout_seconds=timeout_seconds + 5,
    )


def make_jina_fetch_tool(
    *,
    api_key: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    http: JinaHttp | None = None,
    execution_mode: ToolExecutionMode = "parallel",
) -> AgentTool:
    """Return an ``AgentTool`` that fetches a URL as markdown via ``r.jina.ai``.

    ``api_key`` (or the ``JINA_API_KEY`` env var read at call time) authenticates
    for a higher quota. ``http`` injects the HTTP seam for tests.
    """

    _validate_common(timeout_seconds, max_output_chars)
    do_http = http or _urllib_http

    def execute(
        call_id: str,
        args: dict[str, Any],
        abort: AbortFlag,
        on_update: ToolUpdateFn | None,
    ) -> ToolResult:
        del call_id, on_update
        if abort():
            return text_result("Jina fetch aborted before start.", is_error=True)

        target = str(args.get("url", "")).strip()
        if not target:
            return text_result(
                "Missing required jina_fetch argument: url.", is_error=True
            )
        try:
            url, headers = build_fetch_request(
                target, api_key=_resolve_api_key(api_key)
            )
        except ValueError as exc:
            return text_result(f"Invalid jina_fetch argument: {exc}", is_error=True)

        return _run_request(
            do_http,
            url=url,
            headers=headers,
            timeout_seconds=timeout_seconds,
            max_output_chars=max_output_chars,
            label=f"fetch {target}",
            details={"url": target},
        )

    return AgentTool(
        name=JINA_FETCH_TOOL_NAME,
        description=(
            "Fetch a single web page and return its main content as clean "
            "markdown (boilerplate stripped, JavaScript rendered). Prefer this "
            "over a raw HTTP GET when you need to read a page's text. Pass a "
            "full http(s):// URL."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The http(s):// URL of the page to read.",
                },
            },
            "required": ["url"],
            "additionalProperties": False,
        },
        execute=execute,
        execution_mode=execution_mode,
        timeout_seconds=timeout_seconds + 5,
    )


def build_search_request(
    query: str, *, api_key: str | None, num_results: int = DEFAULT_SEARCH_RESULTS
) -> tuple[str, dict[str, str]]:
    """Build the ``(url, headers)`` for a Jina search. Pure; no I/O."""

    url = JINA_SEARCH_ENDPOINT + "?" + urlencode({"q": query})
    headers = _base_headers(api_key)
    # Ask the search endpoint for a bounded number of hits.
    headers["X-Num-Results"] = str(num_results)
    return url, headers


def build_fetch_request(
    target_url: str, *, api_key: str | None
) -> tuple[str, dict[str, str]]:
    """Build the ``(url, headers)`` for a Jina reader fetch. Pure; no I/O.

    Validates that ``target_url`` is an absolute http(s) URL — the reader
    expects a full address appended to its endpoint — and raises ``ValueError``
    with the offending value otherwise.
    """

    parts = urlsplit(target_url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(
            f"url must be an absolute http(s):// URL, got {target_url!r}"
        )
    # The reader takes the target URL as a path suffix; keep ':/?&=#%' etc. so
    # the address is passed through intact rather than percent-mangled.
    url = JINA_READER_ENDPOINT + quote(target_url, safe=":/?&=#%+@!$,;'()*~")
    return url, _base_headers(api_key)


def _run_request(
    do_http: JinaHttp,
    *,
    url: str,
    headers: dict[str, str],
    timeout_seconds: float,
    max_output_chars: int,
    label: str,
    details: dict[str, Any],
) -> ToolResult:
    """Run one Jina HTTP GET and shape the response into a ``ToolResult``."""

    try:
        response = do_http(url, headers, timeout_seconds)
    except JinaHttpError as exc:
        return text_result(
            f"Jina {label} failed: {exc}",
            details={**details, "error": str(exc)},
            is_error=True,
        )

    body, truncated = cap_text(response.text, max_output_chars)
    result_details = {
        **details,
        "status": response.status,
        "truncated": truncated,
    }
    if response.status >= 400:
        return text_result(
            f"Jina {label} returned HTTP {response.status}:\n{body}",
            details=result_details,
            is_error=True,
        )
    if not body.strip():
        return text_result(
            f"Jina {label} returned an empty response (HTTP {response.status}).",
            details=result_details,
        )
    if truncated:
        body += (
            f"\n\n[Truncated to {max_output_chars} chars. Narrow the query or "
            "fetch a more specific URL for the rest.]"
        )
    return text_result(body, details=result_details)


def cap_text(content: str, max_chars: int) -> tuple[str, bool]:
    """Keep the head of ``content`` under ``max_chars``; flag if it was cut.

    Search/fetch results lead with the most relevant material (top hits, the
    page's main content), so a head cap keeps the useful part — unlike bash
    output where the tail (the error) matters too.
    """

    if len(content) <= max_chars:
        return content, False
    return content[:max_chars], True


def _urllib_http(url: str, headers: dict[str, str], timeout: float) -> HttpResponse:
    """Default HTTP seam: a stdlib ``urllib`` GET returning ``HttpResponse``.

    Translates transport failures into ``JinaHttpError`` so the tool layer never
    sees a raw ``urllib`` exception. An HTTP error *status* (4xx/5xx) is returned
    as a normal ``HttpResponse`` — the caller decides how to surface it — while a
    connection/timeout failure raises.
    """

    req = urllib_request.Request(url, headers=headers, method="GET")
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            text = resp.read().decode(charset, errors="replace")
            return HttpResponse(status=resp.status, text=text)
    except urllib_error.HTTPError as exc:
        charset = exc.headers.get_content_charset() if exc.headers else None
        body = exc.read().decode(charset or "utf-8", errors="replace")
        return HttpResponse(status=exc.code, text=body)
    except urllib_error.URLError as exc:
        raise JinaHttpError(f"could not reach {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise JinaHttpError(f"request to {url} timed out after {timeout:g}s") from exc


class JinaHttpError(Exception):
    """A transport-level failure (DNS, connection, timeout) from the HTTP seam."""


def _base_headers(api_key: str | None) -> dict[str, str]:
    headers = {
        "Accept": "text/plain",
        "User-Agent": _USER_AGENT,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _resolve_api_key(api_key: str | None) -> str | None:
    """Factory-provided key wins; otherwise read ``JINA_API_KEY`` at call time."""

    if api_key is not None:
        return api_key
    key = os.environ.get("JINA_API_KEY", "").strip()
    return key or None


def _coerce_optional_count(
    name: str, value: Any, *, default: int, maximum: int
) -> int:
    """Coerce an optional positive count argument, capped at ``maximum``."""

    if value is None or value == "":
        return default
    number = coerce_int(name, value, minimum=1)
    return min(number, maximum)


def _validate_common(timeout_seconds: float, max_output_chars: int) -> None:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")
    if timeout_seconds > MAX_TIMEOUT_SECONDS:
        raise ValueError(
            f"timeout_seconds must be <= {MAX_TIMEOUT_SECONDS:g}, got {timeout_seconds:g}"
        )
    if max_output_chars <= 0:
        raise ValueError("max_output_chars must be > 0")
