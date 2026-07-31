"""Tests for Harbor result summarization helpers."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from simple_long_horizon_agent.evals.harbor.results import (
    find_latest_job_dir,
    summarize_result_file,
)


class HarborResultsTest(unittest.TestCase):
    def test_summarize_result_file_extracts_stable_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            path.write_text(
                json.dumps(
                    {
                        "job_name": "hb-sal",
                        "n_total_trials": 2,
                        "stats": {
                            "n_completed": 2,
                            "n_errors": 1,
                            "evals": {
                                "simple-long-horizon-agent__gpt-test__demo": {
                                    "n_trials": 2,
                                    "n_errors": 1,
                                    "metrics": [{"reward": 0.5}],
                                }
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            summary = summarize_result_file(path)

        self.assertEqual(summary["result_path"], str(path))
        self.assertEqual(summary["job_name"], "hb-sal")
        self.assertEqual(summary["n_total_trials"], 2)
        self.assertEqual(summary["stats"]["n_completed"], 2)
        self.assertEqual(summary["stats"]["n_errors"], 1)

    def test_find_latest_job_dir_uses_newest_result_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "old"
            new = root / "new"
            old.mkdir()
            new.mkdir()
            (old / "result.json").write_text("{}", encoding="utf-8")
            (new / "result.json").write_text("{}", encoding="utf-8")
            os.utime(old / "result.json", (1000, 1000))
            os.utime(new / "result.json", (2000, 2000))

            self.assertEqual(find_latest_job_dir(root), new)
