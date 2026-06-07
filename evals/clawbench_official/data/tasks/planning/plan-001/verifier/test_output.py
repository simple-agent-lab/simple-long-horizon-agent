"""Verifier for plan-001: Extract User Stories from Requirements."""

import json
from pathlib import Path

import pytest

VALID_ROLES = {"user", "admin", "system"}


@pytest.fixture
def workspace(request):
    return Path(request.config.getoption("--workspace"))


@pytest.fixture
def user_stories(workspace):
    path = workspace / "user_stories.json"
    assert path.exists(), "user_stories.json not found in workspace"
    return json.loads(path.read_text())


# ── Core checks (weight 3) ──────────────────────────────────────────────────


@pytest.mark.weight(3)
def test_user_stories_file_exists(workspace):
    assert (workspace / "user_stories.json").exists(), "user_stories.json not found"


@pytest.mark.weight(3)
def test_user_stories_is_array(user_stories):
    assert isinstance(user_stories, list), "user_stories.json must be a JSON array"


@pytest.mark.weight(3)
def test_minimum_story_count(user_stories):
    assert len(user_stories) >= 5, f"Need at least 5 user stories, found {len(user_stories)}"


@pytest.mark.weight(3)
def test_stories_have_role(user_stories):
    for i, story in enumerate(user_stories):
        assert "role" in story, f"Story {i} missing 'role' field"


@pytest.mark.weight(3)
def test_stories_have_action(user_stories):
    for i, story in enumerate(user_stories):
        assert "action" in story, f"Story {i} missing 'action' field"


@pytest.mark.weight(3)
def test_stories_have_benefit(user_stories):
    for i, story in enumerate(user_stories):
        assert "benefit" in story, f"Story {i} missing 'benefit' field"


@pytest.mark.weight(3)
def test_roles_are_valid(user_stories):
    for i, story in enumerate(user_stories):
        assert story["role"] in VALID_ROLES, (
            f"Story {i} has invalid role '{story['role']}', must be one of {VALID_ROLES}"
        )


@pytest.mark.weight(3)
def test_actions_are_non_empty_strings(user_stories):
    for i, story in enumerate(user_stories):
        assert isinstance(story["action"], str) and len(story["action"].strip()) > 5, (
            f"Story {i}: 'action' must be a meaningful string"
        )


@pytest.mark.weight(3)
def test_benefits_are_non_empty_strings(user_stories):
    for i, story in enumerate(user_stories):
        assert isinstance(story["benefit"], str) and len(story["benefit"].strip()) > 5, (
            f"Story {i}: 'benefit' must be a meaningful string"
        )


@pytest.mark.weight(3)
def test_has_user_role(user_stories):
    roles = {s["role"] for s in user_stories}
    assert "user" in roles, "At least one story must have the 'user' role"


# ── Bonus checks (weight 1) ──────────────────────────────────────────────────


@pytest.mark.weight(1)
def test_has_admin_role(user_stories):
    roles = {s["role"] for s in user_stories}
    assert "admin" in roles, "Should include at least one admin story"


@pytest.mark.weight(1)
def test_has_system_role(user_stories):
    roles = {s["role"] for s in user_stories}
    assert "system" in roles, "Should include at least one system story"


@pytest.mark.weight(1)
def test_no_duplicate_actions(user_stories):
    actions = [s["action"].strip().lower() for s in user_stories]
    assert len(actions) == len(set(actions)), "Duplicate actions found"


@pytest.mark.weight(1)
def test_stories_cover_task_management(user_stories):
    combined = " ".join(s["action"].lower() for s in user_stories)
    task_keywords = ["task", "create", "complete", "priority"]
    matches = sum(1 for kw in task_keywords if kw in combined)
    assert matches >= 2, "Stories should cover task management features"


@pytest.mark.weight(1)
def test_stories_cover_collaboration(user_stories):
    combined = " ".join(s["action"].lower() for s in user_stories)
    collab_keywords = ["share", "team", "collaborat", "comment", "assign"]
    matches = sum(1 for kw in collab_keywords if kw in combined)
    assert matches >= 1, "Stories should cover collaboration features"
