"""Verifier for plan-004: API Design from PRD."""

import json
import re
from pathlib import Path

import pytest

VALID_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}
WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
CRUD_METHODS = {"GET", "POST", "PUT", "DELETE"}


@pytest.fixture
def workspace(request):
    return Path(request.config.getoption("--workspace"))


@pytest.fixture
def api_design(workspace):
    path = workspace / "api_design.json"
    assert path.exists(), "api_design.json not found in workspace"
    return json.loads(path.read_text())


@pytest.fixture
def endpoints(api_design):
    assert "endpoints" in api_design, "api_design.json missing 'endpoints' key"
    return api_design["endpoints"]


# ── Core checks (weight 3) ──────────────────────────────────────────────────


@pytest.mark.weight(3)
def test_api_design_file_exists(workspace):
    assert (workspace / "api_design.json").exists(), "api_design.json not found"


@pytest.mark.weight(3)
def test_has_endpoints_key(api_design):
    assert "endpoints" in api_design, "api_design.json must have 'endpoints' key"


@pytest.mark.weight(3)
def test_endpoints_is_array(endpoints):
    assert isinstance(endpoints, list), "'endpoints' must be a JSON array"


@pytest.mark.weight(3)
def test_minimum_endpoint_count(endpoints):
    assert len(endpoints) >= 8, f"Need at least 8 endpoints, found {len(endpoints)}"


@pytest.mark.weight(3)
def test_endpoints_have_method(endpoints):
    for i, ep in enumerate(endpoints):
        assert "method" in ep, f"Endpoint {i} missing 'method'"


@pytest.mark.weight(3)
def test_endpoints_have_valid_methods(endpoints):
    for i, ep in enumerate(endpoints):
        assert ep["method"] in VALID_METHODS, (
            f"Endpoint {i}: invalid method '{ep['method']}', must be one of {VALID_METHODS}"
        )


@pytest.mark.weight(3)
def test_endpoints_have_path(endpoints):
    for i, ep in enumerate(endpoints):
        assert "path" in ep and isinstance(ep["path"], str), (
            f"Endpoint {i} missing or invalid 'path'"
        )


@pytest.mark.weight(3)
def test_paths_start_with_api(endpoints):
    for i, ep in enumerate(endpoints):
        assert ep["path"].startswith("/api"), (
            f"Endpoint {i}: path '{ep['path']}' must start with '/api'"
        )


@pytest.mark.weight(3)
def test_endpoints_have_description(endpoints):
    for i, ep in enumerate(endpoints):
        assert "description" in ep and isinstance(ep["description"], str), (
            f"Endpoint {i} missing 'description'"
        )
        assert len(ep["description"]) > 5, (
            f"Endpoint {i}: description too short"
        )


@pytest.mark.weight(3)
def test_endpoints_have_auth_required(endpoints):
    for i, ep in enumerate(endpoints):
        assert "auth_required" in ep, f"Endpoint {i} missing 'auth_required'"
        assert isinstance(ep["auth_required"], bool), (
            f"Endpoint {i}: 'auth_required' must be a boolean"
        )


@pytest.mark.weight(3)
def test_endpoints_have_response(endpoints):
    for i, ep in enumerate(endpoints):
        assert "response" in ep, f"Endpoint {i} missing 'response'"


@pytest.mark.weight(3)
def test_endpoints_have_request_body(endpoints):
    for i, ep in enumerate(endpoints):
        assert "request_body" in ep, f"Endpoint {i} missing 'request_body'"


@pytest.mark.weight(3)
def test_write_ops_require_auth(endpoints):
    for i, ep in enumerate(endpoints):
        if ep["method"] in WRITE_METHODS:
            path_lower = ep["path"].lower()
            if "register" in path_lower or "login" in path_lower:
                continue
            assert ep["auth_required"] is True, (
                f"Endpoint {i} ({ep['method']} {ep['path']}): "
                "write operations must require authentication"
            )


@pytest.mark.weight(3)
def test_crud_coverage(endpoints):
    methods_used = {ep["method"] for ep in endpoints}
    for method in CRUD_METHODS:
        assert method in methods_used, (
            f"Missing {method} endpoint — CRUD coverage incomplete"
        )


# ── Bonus checks (weight 1) ──────────────────────────────────────────────────


@pytest.mark.weight(1)
def test_path_format_lowercase(endpoints):
    for i, ep in enumerate(endpoints):
        path_parts = ep["path"].split("/")
        for part in path_parts:
            if part.startswith(":") or part == "":
                continue
            assert part == part.lower(), (
                f"Endpoint {i}: path segment '{part}' should be lowercase"
            )


@pytest.mark.weight(1)
def test_get_endpoints_no_request_body(endpoints):
    for i, ep in enumerate(endpoints):
        if ep["method"] == "GET":
            assert ep["request_body"] is None, (
                f"Endpoint {i} (GET {ep['path']}): GET should not have request body"
            )


@pytest.mark.weight(1)
def test_has_book_endpoints(endpoints):
    book_paths = [ep for ep in endpoints if "/book" in ep["path"].lower()]
    assert len(book_paths) >= 2, "Should have at least 2 book-related endpoints"


@pytest.mark.weight(1)
def test_has_order_endpoints(endpoints):
    order_paths = [ep for ep in endpoints if "/order" in ep["path"].lower()]
    assert len(order_paths) >= 2, "Should have at least 2 order-related endpoints"


@pytest.mark.weight(1)
def test_has_review_endpoints(endpoints):
    review_paths = [ep for ep in endpoints if "/review" in ep["path"].lower()]
    assert len(review_paths) >= 1, "Should have at least 1 review-related endpoint"


@pytest.mark.weight(1)
def test_has_auth_endpoints(endpoints):
    auth_paths = [ep for ep in endpoints
                  if "/auth" in ep["path"].lower()
                  or "/login" in ep["path"].lower()
                  or "/register" in ep["path"].lower()]
    assert len(auth_paths) >= 1, "Should have at least 1 auth-related endpoint"
