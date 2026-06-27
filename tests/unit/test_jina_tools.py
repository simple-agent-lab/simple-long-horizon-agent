from __future__ import annotations

import unittest
from typing import cast

from simple_agent_lab import (
    AgentTool,
    ToolResult,
    make_jina_fetch_tool,
    make_jina_search_tool,
    tool_result_text,
)
from simple_agent_lab.tools.jina import (
    JINA_FETCH_TOOL_NAME,
    JINA_READER_ENDPOINT,
    JINA_SEARCH_ENDPOINT,
    JINA_SEARCH_TOOL_NAME,
    HttpResponse,
    JinaHttpError,
    build_fetch_request,
    build_search_request,
    cap_text,
)


class FakeHttp:
    """A recording HTTP seam: returns a canned response, captures the call."""

    def __init__(self, response: HttpResponse | None = None) -> None:
        self.response = response or HttpResponse(status=200, text="ok")
        self.calls: list[tuple[str, dict[str, str], float]] = []

    def __call__(
        self, url: str, headers: dict[str, str], timeout: float
    ) -> HttpResponse:
        self.calls.append((url, headers, timeout))
        return self.response


class RaisingHttp:
    def __call__(
        self, url: str, headers: dict[str, str], timeout: float
    ) -> HttpResponse:
        raise JinaHttpError("could not reach host")


class JinaSearchToolTest(unittest.TestCase):
    def test_returns_markdown_results(self) -> None:
        http = FakeHttp(HttpResponse(200, "1. Title\nURL: https://example.com"))
        tool = make_jina_search_tool(http=http)
        result = _execute(tool, {"query": "claude code"})

        self.assertFalse(result.is_error)
        self.assertIn("Title", tool_result_text(result))
        url, headers, _timeout = http.calls[0]
        self.assertTrue(url.startswith(JINA_SEARCH_ENDPOINT))
        self.assertIn("q=claude+code", url)
        self.assertEqual(headers["X-Num-Results"], "5")

    def test_num_results_is_forwarded_and_capped(self) -> None:
        http = FakeHttp()
        tool = make_jina_search_tool(http=http)
        _execute(tool, {"query": "x", "num_results": 999})

        _url, headers, _timeout = http.calls[0]
        self.assertEqual(headers["X-Num-Results"], "20")

    def test_missing_query_is_an_error(self) -> None:
        http = FakeHttp()
        result = _execute(make_jina_search_tool(http=http), {"query": "   "})

        self.assertTrue(result.is_error)
        self.assertIn("Missing required jina_search argument", tool_result_text(result))
        self.assertEqual(http.calls, [])

    def test_api_key_sets_authorization_header(self) -> None:
        http = FakeHttp()
        tool = make_jina_search_tool(http=http, api_key="secret-key")
        _execute(tool, {"query": "x"})

        _url, headers, _timeout = http.calls[0]
        self.assertEqual(headers["Authorization"], "Bearer secret-key")

    def test_no_api_key_omits_authorization(self) -> None:
        http = FakeHttp()
        tool = make_jina_search_tool(http=http)
        _execute(tool, {"query": "x"})

        _url, headers, _timeout = http.calls[0]
        self.assertNotIn("Authorization", headers)

    def test_transport_failure_is_a_clean_error(self) -> None:
        tool = make_jina_search_tool(http=RaisingHttp())
        result = _execute(tool, {"query": "x"})

        self.assertTrue(result.is_error)
        self.assertIn("could not reach host", tool_result_text(result))

    def test_http_error_status_is_surfaced(self) -> None:
        http = FakeHttp(HttpResponse(429, "rate limited"))
        result = _execute(make_jina_search_tool(http=http), {"query": "x"})

        self.assertTrue(result.is_error)
        self.assertIn("HTTP 429", tool_result_text(result))

    def test_schema_is_strict(self) -> None:
        tool = make_jina_search_tool()
        self.assertEqual(tool.name, JINA_SEARCH_TOOL_NAME)
        self.assertEqual(tool.parameters["required"], ["query"])
        self.assertFalse(tool.parameters["additionalProperties"])


class JinaFetchToolTest(unittest.TestCase):
    def test_fetches_url_as_markdown(self) -> None:
        http = FakeHttp(HttpResponse(200, "# Heading\n\nbody text"))
        tool = make_jina_fetch_tool(http=http)
        result = _execute(tool, {"url": "https://example.com/page"})

        self.assertFalse(result.is_error)
        self.assertIn("Heading", tool_result_text(result))
        url, _headers, _timeout = http.calls[0]
        self.assertTrue(url.startswith(JINA_READER_ENDPOINT))
        self.assertIn("https://example.com/page", url)

    def test_non_http_url_is_rejected_without_a_request(self) -> None:
        http = FakeHttp()
        result = _execute(make_jina_fetch_tool(http=http), {"url": "ftp://nope"})

        self.assertTrue(result.is_error)
        self.assertIn("absolute http(s)", tool_result_text(result))
        self.assertEqual(http.calls, [])

    def test_missing_url_is_an_error(self) -> None:
        http = FakeHttp()
        result = _execute(make_jina_fetch_tool(http=http), {})

        self.assertTrue(result.is_error)
        self.assertIn("Missing required jina_fetch argument", tool_result_text(result))
        self.assertEqual(http.calls, [])

    def test_output_is_truncated_with_a_note(self) -> None:
        http = FakeHttp(HttpResponse(200, "A" * 100))
        tool = make_jina_fetch_tool(http=http, max_output_chars=10)
        result = _execute(tool, {"url": "https://example.com"})

        self.assertFalse(result.is_error)
        self.assertIn("Truncated to 10 chars", tool_result_text(result))
        self.assertTrue(result.details["truncated"])

    def test_empty_response_is_reported_not_silent(self) -> None:
        http = FakeHttp(HttpResponse(200, "   "))
        result = _execute(make_jina_fetch_tool(http=http), {"url": "https://example.com"})

        self.assertFalse(result.is_error)
        self.assertIn("empty response", tool_result_text(result))

    def test_schema_is_strict(self) -> None:
        tool = make_jina_fetch_tool()
        self.assertEqual(tool.name, JINA_FETCH_TOOL_NAME)
        self.assertEqual(tool.parameters["required"], ["url"])
        self.assertFalse(tool.parameters["additionalProperties"])


class JinaHelpersTest(unittest.TestCase):
    def test_build_search_request_encodes_query(self) -> None:
        url, headers = build_search_request("a b&c", api_key=None, num_results=3)
        self.assertIn("q=a+b%26c", url)
        self.assertEqual(headers["X-Num-Results"], "3")
        self.assertNotIn("Authorization", headers)

    def test_build_fetch_request_rejects_relative_url(self) -> None:
        with self.assertRaises(ValueError):
            build_fetch_request("/just/a/path", api_key=None)

    def test_build_fetch_request_preserves_target_url(self) -> None:
        url, _headers = build_fetch_request(
            "https://example.com/a?x=1&y=2", api_key="k"
        )
        self.assertIn("https://example.com/a?x=1&y=2", url)

    def test_cap_text_keeps_head_and_flags_truncation(self) -> None:
        body, truncated = cap_text("abcdef", 3)
        self.assertEqual(body, "abc")
        self.assertTrue(truncated)

        body, truncated = cap_text("abc", 10)
        self.assertEqual(body, "abc")
        self.assertFalse(truncated)

    def test_env_var_is_used_when_no_explicit_key(self) -> None:
        import os

        http = FakeHttp()
        tool = make_jina_search_tool(http=http)
        previous = os.environ.get("JINA_API_KEY")
        os.environ["JINA_API_KEY"] = "env-key"
        try:
            _execute(tool, {"query": "x"})
        finally:
            if previous is None:
                del os.environ["JINA_API_KEY"]
            else:
                os.environ["JINA_API_KEY"] = previous

        _url, headers, _timeout = http.calls[0]
        self.assertEqual(headers["Authorization"], "Bearer env-key")

    def test_invalid_factory_config_raises(self) -> None:
        with self.assertRaises(ValueError):
            make_jina_search_tool(timeout_seconds=0)
        with self.assertRaises(ValueError):
            make_jina_fetch_tool(max_output_chars=0)


def _execute(tool: AgentTool, args: dict[str, object]) -> ToolResult:
    execute = cast(object, tool.execute)
    if not callable(execute):
        raise AssertionError("jina tool has no execute function")
    return execute("call_1", args, lambda: False, None)


if __name__ == "__main__":
    unittest.main()
