"""Verifier for plan-002: Project Milestone Timeline."""

import json
from datetime import date, datetime
from pathlib import Path

import pytest


PROJECT_START = date(2025, 1, 6)
PROJECT_END = date(2025, 4, 4)
REQUIRED_PHASES = 4


@pytest.fixture
def workspace(request):
    return Path(request.config.getoption("--workspace"))


@pytest.fixture
def milestones(workspace):
    path = workspace / "milestones.json"
    assert path.exists(), "milestones.json not found in workspace"
    return json.loads(path.read_text())


def parse_date(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d").date()


# ── Core checks (weight 3) ──────────────────────────────────────────────────


@pytest.mark.weight(3)
def test_milestones_file_exists(workspace):
    assert (workspace / "milestones.json").exists(), "milestones.json not found"


@pytest.mark.weight(3)
def test_milestones_is_array(milestones):
    assert isinstance(milestones, list), "milestones.json must be a JSON array"


@pytest.mark.weight(3)
def test_milestone_count(milestones):
    assert len(milestones) >= REQUIRED_PHASES, (
        f"Need at least {REQUIRED_PHASES} milestones, found {len(milestones)}"
    )


@pytest.mark.weight(3)
def test_milestones_have_name(milestones):
    for i, m in enumerate(milestones):
        assert "name" in m and isinstance(m["name"], str) and len(m["name"]) > 0, (
            f"Milestone {i} missing or empty 'name'"
        )


@pytest.mark.weight(3)
def test_milestones_have_start_date(milestones):
    for i, m in enumerate(milestones):
        assert "start_date" in m, f"Milestone {i} missing 'start_date'"
        try:
            parse_date(m["start_date"])
        except ValueError:
            pytest.fail(f"Milestone {i}: invalid start_date format '{m['start_date']}'")


@pytest.mark.weight(3)
def test_milestones_have_end_date(milestones):
    for i, m in enumerate(milestones):
        assert "end_date" in m, f"Milestone {i} missing 'end_date'"
        try:
            parse_date(m["end_date"])
        except ValueError:
            pytest.fail(f"Milestone {i}: invalid end_date format '{m['end_date']}'")


@pytest.mark.weight(3)
def test_milestones_have_deliverables(milestones):
    for i, m in enumerate(milestones):
        assert "deliverables" in m, f"Milestone {i} missing 'deliverables'"
        assert isinstance(m["deliverables"], list), f"Milestone {i}: 'deliverables' must be an array"


@pytest.mark.weight(3)
def test_milestones_have_dependencies(milestones):
    for i, m in enumerate(milestones):
        assert "dependencies" in m, f"Milestone {i} missing 'dependencies'"
        assert isinstance(m["dependencies"], list), f"Milestone {i}: 'dependencies' must be an array"


@pytest.mark.weight(3)
def test_start_before_end(milestones):
    for i, m in enumerate(milestones):
        start = parse_date(m["start_date"])
        end = parse_date(m["end_date"])
        assert start <= end, (
            f"Milestone {i} '{m['name']}': start_date {m['start_date']} > end_date {m['end_date']}"
        )


@pytest.mark.weight(3)
def test_chronological_order(milestones):
    for i in range(1, len(milestones)):
        prev_start = parse_date(milestones[i - 1]["start_date"])
        curr_start = parse_date(milestones[i]["start_date"])
        assert curr_start >= prev_start, (
            f"Milestone {i} '{milestones[i]['name']}' starts before milestone {i-1} '{milestones[i-1]['name']}'"
        )


@pytest.mark.weight(3)
def test_no_overlapping_phases(milestones):
    for i in range(1, len(milestones)):
        prev_end = parse_date(milestones[i - 1]["end_date"])
        curr_start = parse_date(milestones[i]["start_date"])
        assert curr_start >= prev_end, (
            f"Milestone {i} '{milestones[i]['name']}' overlaps with '{milestones[i-1]['name']}': "
            f"starts {milestones[i]['start_date']} before previous ends {milestones[i-1]['end_date']}"
        )


@pytest.mark.weight(3)
def test_within_project_window(milestones):
    first_start = parse_date(milestones[0]["start_date"])
    last_end = parse_date(milestones[-1]["end_date"])
    assert first_start >= PROJECT_START, (
        f"First milestone starts {first_start} before project start {PROJECT_START}"
    )
    assert last_end <= PROJECT_END, (
        f"Last milestone ends {last_end} after project deadline {PROJECT_END}"
    )


# ── Bonus checks (weight 1) ──────────────────────────────────────────────────


@pytest.mark.weight(1)
def test_deliverables_minimum_per_milestone(milestones):
    for i, m in enumerate(milestones):
        assert len(m["deliverables"]) >= 2, (
            f"Milestone {i} '{m['name']}' has fewer than 2 deliverables"
        )


@pytest.mark.weight(1)
def test_first_milestone_no_dependencies(milestones):
    assert len(milestones[0]["dependencies"]) == 0, (
        "First milestone should have no dependencies"
    )


@pytest.mark.weight(1)
def test_dependencies_reference_valid_names(milestones):
    names = {m["name"] for m in milestones}
    for i, m in enumerate(milestones):
        for dep in m["dependencies"]:
            assert dep in names, (
                f"Milestone {i} '{m['name']}' depends on unknown milestone '{dep}'"
            )


@pytest.mark.weight(1)
def test_dependencies_reference_earlier_milestones(milestones):
    name_order = {m["name"]: idx for idx, m in enumerate(milestones)}
    for i, m in enumerate(milestones):
        for dep in m["dependencies"]:
            if dep in name_order:
                assert name_order[dep] < i, (
                    f"Milestone {i} '{m['name']}' depends on later milestone '{dep}'"
                )


@pytest.mark.weight(1)
def test_milestone_names_unique(milestones):
    names = [m["name"] for m in milestones]
    assert len(names) == len(set(names)), "Milestone names must be unique"
