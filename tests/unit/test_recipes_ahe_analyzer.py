from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.evolution.kernel import store
from simple_agent_lab.evolution.types import Decision, Run
from simple_agent_lab.llm import Provider

from recipes.ahe.analyzer import analyze_runs


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class AheAnalyzerTest(unittest.TestCase):
    def test_analyze_runs_writes_overview_details_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            analysis_dir = root / "analysis"
            version = store.stage(workspace, base=None, edits={"prompt.md": "hello"})
            runs = [
                self._make_run(root, "round_001", "i1", reward=0, message="pytest failed"),
                self._make_run(root, "round_001", "i2", reward=1, message="tests passed"),
            ]
            decisions = [
                Decision(
                    id="dec-1",
                    ts="2026-06-24T00:00:00Z",
                    baseline={},
                    candidate={},
                    slice={},
                    accepted=False,
                    reason="baseline issue",
                )
            ]

            result = analyze_runs(
                Provider(id="fake", api="fake", model="fake-model"),
                runs,
                version,
                decisions,
                analysis_dir,
                knowledge=("tool usage is flaky",),
                complete_fn=lambda req: FakeResponse(
                    json.dumps(
                        {
                            "overview": "# Overview\nModel summary for the run set.",
                            "details": {"i1": "# i1\nRoot cause: shell mismatch."},
                            "patterns": [
                                {
                                    "id": "pat-1",
                                    "instances": ["i1"],
                                    "likely_component": "tool_implementation",
                                    "root_cause": "The shell tool did not surface test failure context.",
                                }
                            ],
                        }
                    )
                ),
            )

            overview_path = analysis_dir / "overview.md"
            detail_path = analysis_dir / "detail" / "i1.md"
            index_path = analysis_dir / "index.json"

            self.assertTrue(overview_path.is_file())
            self.assertIn("Model summary", overview_path.read_text(encoding="utf-8"))
            self.assertTrue(detail_path.is_file())
            self.assertIn("Root cause", detail_path.read_text(encoding="utf-8"))
            index = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(index["run_count"], 2)
            self.assertEqual(index["failed_count"], 1)
            self.assertEqual(index["patterns"][0]["id"], "pat-1")
            self.assertEqual(result.index["run_count"], 2)
            self.assertEqual(result.overview_path, overview_path)

    def test_analyze_runs_writes_fallback_detail_for_omitted_failed_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            analysis_dir = root / "analysis"
            version = store.stage(workspace, base=None, edits={"prompt.md": "hello"})
            runs = [self._make_run(root, "round_001", "i1", reward=0, message="pytest failed")]

            analyze_runs(
                provider=Provider(id="fake", api="fake", model="fake-model"),
                runs=runs,
                version=version,
                decisions=(),
                output_dir=analysis_dir,
                complete_fn=lambda req: FakeResponse(
                    json.dumps({"overview": "# Overview\nEmpty details.", "details": {}})
                ),
            )

            detail_path = analysis_dir / "detail" / "i1.md"
            detail_text = detail_path.read_text(encoding="utf-8")
            self.assertIn("Fallback analysis", detail_text)
            self.assertIn('"reward": 0', detail_text)

    def test_analyze_runs_sanitizes_detail_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            analysis_dir = root / "analysis"
            version = store.stage(workspace, base=None, edits={"prompt.md": "hello"})
            run = self._make_fake_run(
                instance_id="repo/name:case",
                run_id="round_001",
                reward=0,
                result={"reward": 0},
                events=[{"type": "tool", "message": "pytest failed"}],
            )

            analyze_runs(
                provider=Provider(id="fake", api="fake", model="fake-model"),
                runs=[run],
                version=version,
                decisions=(),
                output_dir=analysis_dir,
                complete_fn=lambda req: FakeResponse(
                    json.dumps({"overview": "# Overview\nSanitized.", "details": {}})
                ),
            )

            self.assertTrue((analysis_dir / "detail" / "repo_name_case.md").is_file())

    def _make_run(
        self, root: Path, run_id: str, instance_id: str, *, reward: float, message: str
    ) -> Run:
        run_dir = root / run_id / instance_id
        self._write_run_artifacts(run_dir, reward=reward, message=message)
        return Run(run_dir)

    def _make_fake_run(
        self,
        *,
        instance_id: str,
        run_id: str,
        reward: float,
        result: dict[str, object],
        events: list[dict[str, object]],
    ) -> object:
        class _FakeRun:
            def __init__(self) -> None:
                self.instance_id = instance_id
                self.run_id = run_id
                self.ref = f"{run_id}/{instance_id}"
                self.reward = reward
                self._result = result
                self._events = events

            @property
            def result(self) -> dict[str, object]:
                return self._result

            def events(self) -> tuple[dict[str, object], ...]:
                return tuple(self._events)

        return _FakeRun()

    def _write_run_artifacts(
        self, run_dir: Path, *, reward: float, message: str
    ) -> None:
        out_dir = run_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text(
            json.dumps({"reward": reward}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (out_dir / "trajectory.jsonl").write_text(
            json.dumps({"events": [{"type": "tool", "message": message}]}).strip()
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
