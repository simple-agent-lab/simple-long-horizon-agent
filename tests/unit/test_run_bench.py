"""Tests for the unified benchmark entry (runs/run_bench.py)."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_run_bench():
    # run_bench.py sets up sys.path itself; ensure runs/ is importable for the
    # bench modules it pulls in.
    for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "runs")):
        if p not in sys.path:
            sys.path.insert(0, p)
    path = ROOT / "runs/run_bench.py"
    spec = importlib.util.spec_from_file_location("sal_run_bench", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_bench = _load_run_bench()

EXPECTED_BENCHES = {
    "swebench",
    "programbench",
    "onemillion",
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
        self.assertFalse(run_bench.BENCHES["onemillion"].needs_docker)


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


if __name__ == "__main__":
    unittest.main()
