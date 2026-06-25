import json
import tempfile
import unittest
from pathlib import Path

from recipes.dgm import swebench
from simple_agent_lab.evolution.types import Run


class DgmSwebenchDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def _run(
        self,
        name: str,
        result: dict | None = None,
        failure: dict | None = None,
    ) -> Run:
        root = self.tmp / name
        out = root / "out"
        out.mkdir(parents=True)
        if result is not None:
            (out / "result.json").write_text(
                json.dumps(result) + "\n", encoding="utf-8"
            )
        if failure is not None:
            (out / "failure.json").write_text(
                json.dumps(failure) + "\n", encoding="utf-8"
            )
        return Run(root)

    def test_completed_result_is_completed(self):
        run = self._run("case", {"resolved": True, "score": 1.0})

        diagnostic = swebench.run_diagnostic(run)

        self.assertEqual(diagnostic["status"], "completed")
        self.assertEqual(diagnostic["reward"], 1.0)

    def test_agent_build_failure_is_distinct_from_container_failure(self):
        run = self._run(
            "case",
            {
                "resolved": False,
                "score": 0.0,
                "agent_package": {
                    "loaded": True,
                    "used_fallback": True,
                    "error": "RuntimeError: boom",
                },
            },
        )

        diagnostic = swebench.run_diagnostic(run)

        self.assertEqual(diagnostic["status"], "agent_build_failed")

    def test_agent_load_failure_is_distinct(self):
        run = self._run(
            "case",
            {
                "resolved": False,
                "score": 0.0,
                "agent_package": {
                    "loaded": False,
                    "used_fallback": True,
                    "error": "SyntaxError: invalid syntax",
                },
            },
        )

        diagnostic = swebench.run_diagnostic(run)

        self.assertEqual(diagnostic["status"], "agent_load_failed")

    def test_container_failure_is_distinct_when_result_exists_only_as_fallback_marker(
        self,
    ):
        run = self._run(
            "case",
            {
                "resolved": False,
                "score": 0.0,
                "agent_package": {
                    "loaded": False,
                    "used_fallback": True,
                    "error": "container exited with status 1 before writing a result",
                },
            },
        )

        diagnostic = swebench.run_diagnostic(run)

        self.assertEqual(diagnostic["status"], "container_failed")

    def test_missing_result_reads_failure_json(self):
        run = self._run("case", failure={"likely_reason": "container oom"})

        diagnostic = swebench.run_diagnostic(run)

        self.assertEqual(diagnostic["status"], "missing_result")
        self.assertIn("container oom", diagnostic["message"])

    def test_scoring_failure_is_distinct(self):
        run = self._run(
            "case",
            {"resolved": False, "score": 0.0, "scoring_error": "bad log"},
        )

        diagnostic = swebench.run_diagnostic(run)

        self.assertEqual(diagnostic["status"], "scoring_failed")
        self.assertIn("bad log", diagnostic["message"])

    def test_candidate_summary_marks_isolated_container_failure_valid(self):
        runs = [
            self._run("ok1", {"resolved": True, "score": 1.0}),
            self._run("ok2", {"resolved": False, "score": 0.0}),
            self._run(
                "container",
                {
                    "resolved": False,
                    "score": 0.0,
                    "agent_package": {
                        "loaded": False,
                        "used_fallback": True,
                        "error": "container exited with status 1 before writing a result",
                    },
                },
            ),
        ]

        summary = swebench.candidate_diagnostics(runs, min_complete_fraction=0.5)

        self.assertTrue(summary["valid_parent"])
        self.assertEqual(summary["container_failed"], 1)
        self.assertEqual(summary["completed"], 2)

    def test_candidate_summary_rejects_agent_build_failure(self):
        runs = [
            self._run(
                "bad",
                {
                    "resolved": False,
                    "score": 0.0,
                    "agent_package": {
                        "loaded": True,
                        "used_fallback": True,
                        "error": "ValueError: bad build",
                    },
                },
            )
        ]

        summary = swebench.candidate_diagnostics(runs)

        self.assertFalse(summary["valid_parent"])
        self.assertEqual(summary["agent_build_failed"], 1)

    def test_swebench_reward_includes_diagnostic_dimensions(self):
        run = self._run(
            "bad",
            {
                "resolved": False,
                "score": 0.0,
                "agent_package": {
                    "loaded": True,
                    "used_fallback": True,
                    "error": "ValueError: bad build",
                },
            },
        )

        reward = swebench.swebench_reward(run)

        self.assertEqual(reward["reward"], 0.0)
        self.assertEqual(reward["agent_build_failed"], 1.0)
        self.assertEqual(reward["valid_parent"], 0.0)


if __name__ == "__main__":
    unittest.main()
