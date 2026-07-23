"""Tests for the unified benchmark entry (runs/run_bench.py)."""

from __future__ import annotations

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from tests.unit._support import load_module

ROOT = Path(__file__).resolve().parents[2]


def _load_run_bench():
    # run_bench.py sets up sys.path itself; ensure runs/ is importable for the
    # bench modules it pulls in.
    for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "runs")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return load_module(ROOT / "runs/run_bench.py", "sal_run_bench")


run_bench = _load_run_bench()

EXPECTED_BENCHES = {
    "swebench",
    "programbench",
    "onemillion",
    "harbor",
}


class RunBenchRegistryTest(unittest.TestCase):
    def test_registry_lists_every_bench_with_a_run_api(self) -> None:
        self.assertEqual(set(run_bench.BENCHES), EXPECTED_BENCHES)
        for bench in run_bench.BENCHES.values():
            self.assertTrue(callable(bench.module.run))
            self.assertTrue(callable(bench.module._build_parser))
            self.assertTrue(bench.description)

    def test_docker_benches_are_flagged(self) -> None:
        self.assertTrue(run_bench.BENCHES["swebench"].needs_docker)
        self.assertTrue(run_bench.BENCHES["programbench"].needs_docker)
        self.assertTrue(run_bench.BENCHES["harbor"].needs_docker)
        self.assertFalse(run_bench.BENCHES["onemillion"].needs_docker)

    def test_benches_dir_and_registry_are_in_sync(self) -> None:
        """Every runs/_benches/<x>.py is registered, and vice versa.

        A bench can't be added "in a weird place": the only way in is a module
        under runs/_benches/ that exposes NAME / _build_parser / run AND is
        wired into run_bench.BENCHES under that NAME.
        """
        modules = {
            p.stem
            for p in (ROOT / "runs/_benches").glob("*.py")
            if p.stem != "__init__"
        }
        self.assertEqual(
            modules,
            set(run_bench.BENCHES),
            "runs/_benches/<x>.py modules must match run_bench.BENCHES exactly "
            "(register a new bench, or move stray modules out of _benches/).",
        )
        for name, bench in run_bench.BENCHES.items():
            with self.subTest(bench=name):
                self.assertEqual(bench.module.NAME, name)
                self.assertTrue(bench.module.DESCRIPTION)
                self.assertTrue(callable(bench.module.run))
                self.assertTrue(callable(bench.module._build_parser))


class RunBenchCliTest(unittest.TestCase):
    def _run_json(self, argv: list[str]) -> dict:
        buf = io.StringIO()
        with redirect_stdout(buf):
            run_bench.main(argv)
        return json.loads(buf.getvalue().strip().splitlines()[-1])

    def test_list_json_reports_all_benches(self) -> None:
        payload = self._run_json(["list", "--json"])
        names = {b["name"] for b in payload["benches"]}
        self.assertEqual(names, EXPECTED_BENCHES)

    def test_setup_json_has_probe_shape(self) -> None:
        payload = self._run_json(["setup", "--json"])
        self.assertIn("checks", payload)
        self.assertIn("benches", payload)
        self.assertIn("ok", payload)
        # Every selected bench gets a readiness verdict.
        self.assertEqual({b["name"] for b in payload["benches"]}, EXPECTED_BENCHES)

    def test_setup_one_bench_scopes_the_report(self) -> None:
        payload = self._run_json(["setup", "onemillion", "--json"])
        self.assertEqual([b["name"] for b in payload["benches"]], ["onemillion"])

    def test_unknown_command_returns_error_code(self) -> None:
        self.assertEqual(run_bench.main(["nope"]), 2)

    def test_batch_dispatches_to_bench_batch_api(self) -> None:
        module = run_bench.BENCHES["swebench"].module
        with mock.patch.object(
            module,
            "run_batch",
            return_value={"bench": "swebench", "status_code": 0},
        ) as runner:
            code = run_bench.main(["batch", "swebench", "--variant", "pro"])

        self.assertEqual(code, 0)
        self.assertEqual(runner.call_args.args[0].variant, "pro")


class RunBenchHelpersTest(unittest.TestCase):
    def test_parse_last_json_picks_trailing_object(self) -> None:
        text = "noise line\nrun progress...\n" + json.dumps(
            {"bench": "x", "status_code": 0}
        )
        self.assertEqual(
            run_bench._parse_last_json(text), {"bench": "x", "status_code": 0}
        )

    def test_parse_last_json_none_when_absent(self) -> None:
        self.assertIsNone(run_bench._parse_last_json("just logs\nno json here"))

    def test_provider_creds_detected_from_env_file(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / ".env"
            env.write_text("OPENAI_MODEL=m\nOPENAI_AUTH_TOKEN=t\n", encoding="utf-8")
            self.assertTrue(run_bench._provider_creds_present(env))
            env.write_text("OPENAI_MODEL=m\n", encoding="utf-8")
            # Missing token (and assuming it isn't exported in this process).
            import os

            if not os.environ.get("OPENAI_AUTH_TOKEN"):
                self.assertFalse(run_bench._provider_creds_present(env))


class RunBenchScoreOracleTest(unittest.TestCase):
    def _capture(self, argv: list[str]) -> tuple[int, str]:
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = run_bench.main(argv)
        return code, buf.getvalue()

    def test_score_inline_bench_is_a_noop(self) -> None:
        code, out = self._capture(["score", "onemillion", "--json"])
        payload = json.loads(out.strip().splitlines()[-1])
        self.assertEqual(code, 0)
        self.assertTrue(payload["inline"])
        self.assertEqual(payload["bench"], "onemillion")

    def test_score_and_oracle_need_a_known_bench(self) -> None:
        self.assertEqual(run_bench.main(["score"]), 2)
        self.assertEqual(run_bench.main(["score", "nope"]), 2)
        self.assertEqual(run_bench.main(["oracle", "nope"]), 2)

    def test_oracle_unsupported_bench_errors(self) -> None:
        # ProgramBench has no apply_oracle / --provider oracle.
        self.assertEqual(run_bench.main(["oracle", "programbench"]), 2)

    def test_oracle_supported_where_provider_accepts_it(self) -> None:
        accepts = run_bench._provider_accepts_oracle
        self.assertTrue(accepts(run_bench.BENCHES["swebench"].module._build_parser()))
        self.assertTrue(accepts(run_bench.BENCHES["onemillion"].module._build_parser()))
        self.assertFalse(
            accepts(run_bench.BENCHES["programbench"].module._build_parser())
        )

    def test_benches_with_a_scorer_point_at_a_real_file(self) -> None:
        for name, bench in run_bench.BENCHES.items():
            scorer = getattr(bench.module, "SCORER", None)
            if scorer is None:
                continue
            with self.subTest(bench=name):
                self.assertTrue((ROOT / scorer[0]).exists(), scorer)
