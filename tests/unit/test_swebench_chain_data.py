from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.swebench import analyze_swebench_pro_chains as analyze
from scripts.swebench import export_swebench_pro_chain_experiments as export
from scripts.swebench import rebuild_deep_chains as rebuild


class SwebenchChainAnalysisTest(unittest.TestCase):
    def test_relation_file_extraction_and_noise_filtering(self) -> None:
        patch = (
            "diff --git a/src/widget.py b/src/widget.py\n"
            "diff --git a/tests/test_widget.py b/tests/test_widget.py\n"
        )

        files = analyze.patch_files(patch)

        self.assertEqual(files, {"src/widget.py", "tests/test_widget.py"})
        self.assertEqual(
            analyze.filter_relation_files(files, "python"), {"src/widget.py"}
        )
        self.assertEqual(
            analyze.interface_files("Changes `src/api.py` and `src/types.py`."),
            {"src/api.py", "src/types.py"},
        )

    def test_deep_relink_ignores_hot_files_and_sorts_chronologically(self) -> None:
        issues = [
            {
                "instance_id": "third",
                "commit_time": "2024-03-01T00:00:00+00:00",
                "files": ["hot.py"],
            },
            {
                "instance_id": "second",
                "commit_time": "2024-02-01T00:00:00+00:00",
                "files": ["hot.py", "pair.py"],
            },
            {
                "instance_id": "first",
                "commit_time": "2024-01-01T00:00:00+00:00",
                "files": ["hot.py", "pair.py"],
            },
        ]

        without_cutoff = rebuild.link_chains(
            issues, min_chain_size=2, ignore_file_freq=0
        )
        with_cutoff = rebuild.link_chains(issues, min_chain_size=2, ignore_file_freq=2)

        self.assertEqual(
            [issue["instance_id"] for issue in without_cutoff[0]],
            ["first", "second", "third"],
        )
        self.assertEqual(
            [issue["instance_id"] for issue in with_cutoff[0]],
            ["first", "second"],
        )


class SwebenchChainExportTest(unittest.TestCase):
    def test_export_records_prior_nodes_and_jsonish_test_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chains_path = root / "chains.json"
            dataset_path = root / "dataset.json"
            output_json = root / "out" / "experiments.json"
            output_jsonl = root / "out" / "nodes.jsonl"
            issues = [
                self._issue("first", "2024-01-01T00:00:00+00:00", ["shared.py"]),
                self._issue(
                    "second",
                    "2024-02-01T00:00:00+00:00",
                    ["shared.py", "second.py"],
                ),
            ]
            chains_path.write_text(
                json.dumps(
                    {
                        "repos": [
                            {
                                "repo": "acme/widgets",
                                "chains": [
                                    {
                                        "chain_id": "acme__widgets-0001",
                                        "issue_count": 2,
                                        "shared_files": ["shared.py"],
                                        "files": ["second.py", "shared.py"],
                                        "issues": issues,
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            dataset_path.write_text(
                json.dumps(
                    [
                        self._dataset_row("first"),
                        self._dataset_row("second"),
                    ]
                ),
                encoding="utf-8",
            )

            manifest = export.export_manifests(
                chains_path, dataset_path, output_json, output_jsonl
            )
            nodes = [
                json.loads(line)
                for line in output_jsonl.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(manifest["experiment_count"], 1)
        self.assertEqual(manifest["node_count"], 2)
        self.assertEqual(nodes[0]["prior_instance_ids"], [])
        self.assertEqual(nodes[1]["prior_instance_ids"], ["first"])
        self.assertEqual(
            nodes[1]["direct_predecessors"],
            [
                {
                    "step_index": 1,
                    "instance_id": "first",
                    "overlap_files": ["shared.py"],
                }
            ],
        )
        self.assertEqual(nodes[1]["fail_to_pass"], ["tests/test_widget.py"])
        self.assertEqual(nodes[1]["selected_test_files_to_run"], ["tests"])

    @staticmethod
    def _issue(instance_id: str, commit_time: str, files: list[str]) -> dict:
        return {
            "instance_id": instance_id,
            "base_commit": f"base-{instance_id}",
            "commit_time": commit_time,
            "files_source": "patch",
            "files": files,
            "raw_files": files,
        }

    @staticmethod
    def _dataset_row(instance_id: str) -> dict:
        return {
            "repo": "acme/widgets",
            "repo_language": "python",
            "instance_id": instance_id,
            "dockerhub_tag": "acme/widgets:test",
            "selected_test_files_to_run": '["tests"]',
            "fail_to_pass": '["tests/test_widget.py"]',
            "pass_to_pass": "[]",
        }


if __name__ == "__main__":
    unittest.main()
