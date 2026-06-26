"""Tests for the JSON run-profile loader (ADR run-profile-file)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from simple_agent_lab.evals.profile import (
    apply_profile_env,
    load_run_profile,
    parse_with_profile,
    profile_run_argv,
)


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_coerces_env_scalars_and_keeps_run(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {
            "_comment": "ignored",
            "env": {"AGENT_FLAVOR": "pdr", "SWE_PDR_WIDTH": 3, "FLAG": True},
            "run": {"max-turns": 200, "prepare-wheelhouse": True},
        },
    )
    profile = load_run_profile(path)
    # env values become environment strings; a JSON bool is lowercased.
    assert profile.env == {
        "AGENT_FLAVOR": "pdr",
        "SWE_PDR_WIDTH": "3",
        "FLAG": "true",
    }
    # run values keep their JSON type for argv rendering.
    assert profile.run == {"max-turns": 200, "prepare-wheelhouse": True}


def test_underscore_keys_are_comments(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {"env": {"_note": "doc", "X": "1"}, "run": {"_note": "doc", "y": 2}},
    )
    profile = load_run_profile(path)
    assert profile.env == {"X": "1"}
    assert profile.run == {"y": 2}


def test_apply_profile_env_fills_gaps_without_override() -> None:
    environ = {"EXISTING": "keep"}
    apply_profile_env({"EXISTING": "no", "NEW": "yes"}, environ=environ)
    assert environ == {"EXISTING": "keep", "NEW": "yes"}


def test_profile_run_argv_rendering() -> None:
    argv = profile_run_argv(
        {"max-turns": 200, "network-mode": "host", "on": True, "off": False}
    )
    assert argv == ["--max-turns", "200", "--network-mode", "host", "--on"]


def test_parse_with_profile_injects_and_cli_overrides(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {"env": {"AGENT_FLAVOR": "pdr"}, "run": {"max-turns": 200, "flag": True}},
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("instance_id")
    parser.add_argument("--profile")
    parser.add_argument("--max-turns", type=int, default=1)
    parser.add_argument("--flag", action="store_true")
    environ: dict[str, str] = {}

    # Profile supplies max-turns=200 and --flag; explicit --max-turns 50 wins.
    args = parse_with_profile(
        parser,
        argv=["case-1", "--profile", str(path), "--max-turns", "50"],
        environ=environ,
    )
    assert args.instance_id == "case-1"
    assert args.max_turns == 50
    assert args.flag is True
    assert environ == {"AGENT_FLAVOR": "pdr"}


def test_parse_with_profile_noop_without_flag() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("instance_id")
    parser.add_argument("--profile")
    parser.add_argument("--max-turns", type=int, default=1)
    args = parse_with_profile(parser, argv=["case-1"], environ={})
    assert args.instance_id == "case-1"
    assert args.max_turns == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"bogus": {}},
        {"env": ["not", "a", "dict"]},
        {"run": {"k": {"nested": 1}}},
    ],
)
def test_invalid_shapes_raise(tmp_path: Path, payload: dict) -> None:
    path = _write(tmp_path, payload)
    with pytest.raises(SystemExit):
        load_run_profile(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        load_run_profile(tmp_path / "nope.json")


def test_bad_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_run_profile(path)
