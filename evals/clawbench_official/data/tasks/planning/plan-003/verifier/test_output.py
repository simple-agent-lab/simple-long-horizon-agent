"""Verifier for plan-003: Technology Selection Analysis."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def workspace(request):
    return Path(request.config.getoption("--workspace"))


@pytest.fixture
def constraints(workspace):
    path = workspace / "constraints.json"
    assert path.exists(), "constraints.json not found in workspace"
    return json.loads(path.read_text())


@pytest.fixture
def analysis(workspace):
    path = workspace / "analysis.json"
    assert path.exists(), "analysis.json not found in workspace"
    return json.loads(path.read_text())


@pytest.fixture
def options(analysis):
    assert "options" in analysis, "analysis.json missing 'options' key"
    return analysis["options"]


# ── Core checks (weight 3) ──────────────────────────────────────────────────


@pytest.mark.weight(3)
def test_analysis_file_exists(workspace):
    assert (workspace / "analysis.json").exists(), "analysis.json not found"


@pytest.mark.weight(3)
def test_analysis_has_options_key(analysis):
    assert "options" in analysis, "analysis.json must have an 'options' key"


@pytest.mark.weight(3)
def test_options_is_array(options):
    assert isinstance(options, list), "'options' must be a JSON array"


@pytest.mark.weight(3)
def test_minimum_option_count(options):
    assert len(options) >= 3, f"Need at least 3 options, found {len(options)}"


@pytest.mark.weight(3)
def test_options_have_name(options):
    for i, opt in enumerate(options):
        assert "name" in opt and isinstance(opt["name"], str) and len(opt["name"]) > 0, (
            f"Option {i} missing or empty 'name'"
        )


@pytest.mark.weight(3)
def test_options_have_pros(options):
    for i, opt in enumerate(options):
        assert "pros" in opt and isinstance(opt["pros"], list), (
            f"Option {i} missing 'pros' array"
        )
        assert len(opt["pros"]) >= 2, f"Option {i} needs at least 2 pros"


@pytest.mark.weight(3)
def test_options_have_cons(options):
    for i, opt in enumerate(options):
        assert "cons" in opt and isinstance(opt["cons"], list), (
            f"Option {i} missing 'cons' array"
        )
        assert len(opt["cons"]) >= 1, f"Option {i} needs at least 1 con"


@pytest.mark.weight(3)
def test_options_have_cost_estimate(options):
    for i, opt in enumerate(options):
        assert "cost_estimate" in opt, f"Option {i} missing 'cost_estimate'"
        assert isinstance(opt["cost_estimate"], (int, float)), (
            f"Option {i}: 'cost_estimate' must be a number"
        )


@pytest.mark.weight(3)
def test_options_have_recommendation_score(options):
    for i, opt in enumerate(options):
        assert "recommendation_score" in opt, f"Option {i} missing 'recommendation_score'"
        score = opt["recommendation_score"]
        assert isinstance(score, (int, float)) and 1 <= score <= 10, (
            f"Option {i}: 'recommendation_score' must be between 1 and 10, got {score}"
        )


@pytest.mark.weight(3)
def test_exactly_one_recommended(options):
    recommended = [opt for opt in options if opt.get("recommended") is True]
    assert len(recommended) == 1, (
        f"Exactly one option must have recommended=true, found {len(recommended)}"
    )


@pytest.mark.weight(3)
def test_recommended_has_highest_score(options):
    recommended = [opt for opt in options if opt.get("recommended") is True]
    assert len(recommended) == 1, "No recommended option found"
    rec_score = recommended[0]["recommendation_score"]
    max_score = max(opt["recommendation_score"] for opt in options)
    assert rec_score == max_score, (
        f"Recommended option has score {rec_score} but max score is {max_score}"
    )


# ── Bonus checks (weight 1) ──────────────────────────────────────────────────


@pytest.mark.weight(1)
def test_costs_within_budget(options, constraints):
    budget = constraints.get("budget", {}).get("total", float("inf"))
    for i, opt in enumerate(options):
        assert opt["cost_estimate"] <= budget, (
            f"Option {i} '{opt['name']}' cost {opt['cost_estimate']} exceeds budget {budget}"
        )


@pytest.mark.weight(1)
def test_option_names_unique(options):
    names = [opt["name"] for opt in options]
    assert len(names) == len(set(names)), "Option names must be unique"


@pytest.mark.weight(1)
def test_scores_are_integers(options):
    for i, opt in enumerate(options):
        score = opt["recommendation_score"]
        assert isinstance(score, int), (
            f"Option {i}: 'recommendation_score' should be an integer, got {type(score).__name__}"
        )


@pytest.mark.weight(1)
def test_costs_are_positive(options):
    for i, opt in enumerate(options):
        assert opt["cost_estimate"] > 0, (
            f"Option {i}: 'cost_estimate' must be positive"
        )


@pytest.mark.weight(1)
def test_recommended_has_true_boolean(options):
    for i, opt in enumerate(options):
        assert "recommended" in opt, f"Option {i} missing 'recommended' field"
        assert isinstance(opt["recommended"], bool), (
            f"Option {i}: 'recommended' must be a boolean, got {type(opt['recommended']).__name__}"
        )
