"""Tests for the JSON run-profile loader (ADR run-profile-file)."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.evals.profile import (
    apply_profile_env,
    load_run_profile,
    parse_with_profile,
    profile_run_argv,
)


def _write(directory: Path, payload: dict) -> Path:
    path = directory / "profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class RunProfileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_load_coerces_env_scalars_and_keeps_run(self) -> None:
        path = _write(
            self.tmp,
            {
                "_comment": "ignored",
                "env": {
                    "AGENT_FLAVOR": "pdr",
                    "SAL_WORKFLOW_PDR_WIDTH": 3,
                    "FLAG": True,
                },
                "run": {"max-turns": 200, "prepare-wheelhouse": True},
            },
        )
        profile = load_run_profile(path)
        # env values become environment strings; a JSON bool is lowercased.
        self.assertEqual(
            profile.env,
            {"AGENT_FLAVOR": "pdr", "SAL_WORKFLOW_PDR_WIDTH": "3", "FLAG": "true"},
        )
        # run values keep their JSON type for argv rendering.
        self.assertEqual(profile.run, {"max-turns": 200, "prepare-wheelhouse": True})

    def test_underscore_keys_are_comments(self) -> None:
        path = _write(
            self.tmp,
            {"env": {"_note": "doc", "X": "1"}, "run": {"_note": "doc", "y": 2}},
        )
        profile = load_run_profile(path)
        self.assertEqual(profile.env, {"X": "1"})
        self.assertEqual(profile.run, {"y": 2})

    def test_apply_profile_env_fills_gaps_without_override(self) -> None:
        environ = {"EXISTING": "keep"}
        apply_profile_env({"EXISTING": "no", "NEW": "yes"}, environ=environ)
        self.assertEqual(environ, {"EXISTING": "keep", "NEW": "yes"})

    def test_profile_run_argv_rendering(self) -> None:
        argv = profile_run_argv(
            {"max-turns": 200, "network-mode": "host", "on": True, "off": False}
        )
        self.assertEqual(argv, ["--max-turns", "200", "--network-mode", "host", "--on"])

    def test_parse_with_profile_injects_and_cli_overrides(self) -> None:
        path = _write(
            self.tmp,
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
        self.assertEqual(args.instance_id, "case-1")
        self.assertEqual(args.max_turns, 50)
        self.assertTrue(args.flag)
        self.assertEqual(environ, {"AGENT_FLAVOR": "pdr"})

    def test_parse_with_profile_noop_without_flag(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("instance_id")
        parser.add_argument("--profile")
        parser.add_argument("--max-turns", type=int, default=1)
        args = parse_with_profile(parser, argv=["case-1"], environ={})
        self.assertEqual(args.instance_id, "case-1")
        self.assertEqual(args.max_turns, 1)

    def test_invalid_shapes_raise(self) -> None:
        invalid_payloads = [
            {"bogus": {}},
            {"env": ["not", "a", "dict"]},
            {"run": {"k": {"nested": 1}}},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                path = _write(self.tmp, payload)
                with self.assertRaises(SystemExit):
                    load_run_profile(path)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(SystemExit):
            load_run_profile(self.tmp / "nope.json")

    def test_bad_json_raises(self) -> None:
        path = self.tmp / "profile.json"
        path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(SystemExit):
            load_run_profile(path)


if __name__ == "__main__":
    unittest.main()
