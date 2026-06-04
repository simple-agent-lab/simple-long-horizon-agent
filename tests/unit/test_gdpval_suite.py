from __future__ import annotations

import base64
import json
import shutil
import tempfile
import unittest
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from evals.gdpval.load_instances import load_instances
from evals.gdpval.judge_suite import GdpvalGsbJudgeSuite, GdpvalJudgeSuite
from evals.gdpval.suite import GdpvalSuite
from simple_agent_lab.evals import (
    RESULT_KEY,
    TRACE_KEY,
    LocalDirStore,
    LocalProcessBackend,
    RunArtifacts,
    run_suite_instance,
)
from simple_agent_lab.evals.suites.gdpval import (
    container,
    judge_container,
    judge_gsb_container,
)
from simple_agent_lab.evals.suites.gdpval.judge_scoring import (
    normalize_rubrics,
    parse_gsb_judge_payload,
    parse_judge_payload,
    score_gsb_judgment,
    score_judgment,
)
from simple_agent_lab.evals.suites.gdpval.prompts import GDPVAL_SYSTEM_PROMPT
from simple_agent_lab.evals.suites.gdpval.tools import make_gdpval_tools
from simple_agent_lab.llm.provider import Provider
from simple_agent_lab.tools import tool_result_text


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return None


class GdpvalSuiteTest(unittest.TestCase):
    def test_task_input_hides_gold_and_inlines_reference_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "refs").mkdir()
            (root / "refs" / "brief.txt").write_text("reference text", encoding="utf-8")

            suite = GdpvalSuite(reference_root=root)
            visible = suite.task_input(
                {
                    "task_id": "task-1",
                    "prompt": "Create a deliverable.",
                    "reference_files": ["refs/brief.txt"],
                    "deliverable_files": ["gold-answer.txt"],
                    "rubric_json": [{"score": 1, "criterion": "secret"}],
                }
            )

            self.assertEqual(visible["solver_agent_mode"], "tool-call-context-managed")
            self.assertNotIn("deliverable_files", visible)
            self.assertNotIn("rubric_json", visible)
            self.assertEqual(visible["reference_files"][0]["name"], "refs/brief.txt")
            self.assertNotIn("data_base64", visible["reference_files"][0])
            decoded = base64.b64decode(
                visible["reference_file_blobs"][0]["data_base64"]
            ).decode("utf-8")
            self.assertEqual(decoded, "reference text")

    def test_task_input_uses_official_reference_files_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task-1"
            task_dir.mkdir()
            (task_dir / "Population v2.xlsx").write_bytes(b"xlsx bytes")

            visible = GdpvalSuite(reference_root=root).task_input(
                {
                    "task_id": "task-1",
                    "prompt": "Create a deliverable.",
                    "reference_file_urls": [
                        "https://example.invalid/reference_files/hash/Population%20v2.xlsx"
                    ],
                    "reference_files": ["reference_files/hash/Population v2.xlsx"],
                }
            )

            blob = visible["reference_file_blobs"][0]
            self.assertFalse(blob.get("missing"))
            self.assertEqual(base64.b64decode(blob["data_base64"]), b"xlsx bytes")

    def test_task_input_downloads_reference_urls_without_reference_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Population v2.xlsx").write_bytes(b"downloaded bytes")
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                lambda *args, **kwargs: _QuietHandler(
                    *args, directory=str(root), **kwargs
                ),
            )
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                port = server.server_address[1]
                visible = GdpvalSuite().task_input(
                    {
                        "task_id": "task-1",
                        "prompt": "Create a deliverable.",
                        "reference_files": ["reference_files/hash/Population v2.xlsx"],
                        "reference_file_urls": [
                            f"http://127.0.0.1:{port}/Population%20v2.xlsx"
                        ],
                    }
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            blob = visible["reference_file_blobs"][0]
            self.assertFalse(blob.get("missing"))
            self.assertEqual(base64.b64decode(blob["data_base64"]), b"downloaded bytes")

    def test_load_instances_filters_empty_deliverable_files_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "task_id": "empty",
                                "prompt": "p",
                                "deliverable_files": [],
                            }
                        ),
                        json.dumps(
                            {
                                "task_id": "nonempty",
                                "prompt": "p",
                                "deliverable_files": ["deliverable_files/x/out.xlsx"],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rows = load_instances(path)

            self.assertEqual([row["task_id"] for row in rows], ["nonempty"])

    def test_load_instances_can_include_empty_deliverable_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "task_id": "empty",
                                "prompt": "p",
                                "deliverable_files": [],
                            }
                        ),
                        json.dumps(
                            {
                                "task_id": "nonempty",
                                "prompt": "p",
                                "deliverable_files": ["deliverable_files/x/out.xlsx"],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rows = load_instances(path, require_deliverables=False)

            self.assertEqual([row["task_id"] for row in rows], ["empty", "nonempty"])

    def test_prepare_writes_reference_files_and_task_mentions_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "task" / "workdir"
            blob = base64.b64encode(b"reference text").decode("ascii")
            instance = {
                "task_id": "task-1",
                "prompt": "Use the reference.",
                "reference_files": [{"name": "brief.txt", "source": "brief.txt"}],
                "reference_file_blobs": [
                    {"name": "brief.txt", "source": "brief.txt", "data_base64": blob}
                ],
            }

            context = container.prepare(workdir, instance)
            reference_dir = Path(context["reference_dir"])
            self.assertEqual(
                (reference_dir / "brief.txt").read_text(encoding="utf-8"),
                "reference text",
            )

            task = container.build_task(instance, workdir=str(workdir))
            self.assertIn(f"WORKDIR: {workdir}", task)
            self.assertIn(str(reference_dir / "brief.txt"), task)
            self.assertIn("<FINAL_ANSWER>", GDPVAL_SYSTEM_PROMPT)
            self.assertIn("</FINAL_ANSWER>", GDPVAL_SYSTEM_PROMPT)
            self.assertIn("execute_bash as the primary tool", GDPVAL_SYSTEM_PROMPT)
            self.assertIn("multi_edit_file", GDPVAL_SYSTEM_PROMPT)
            self.assertIn("view_image", GDPVAL_SYSTEM_PROMPT)
            self.assertIn("<FINAL_ANSWER>...</FINAL_ANSWER>", task)

    def test_gdpval_tools_cover_file_and_shell_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "workdir"
            reference_dir = Path(tmp) / "reference_task_id_files"
            reference_dir.mkdir()
            (reference_dir / "brief.txt").write_text(
                "alpha reference", encoding="utf-8"
            )
            tools = {
                tool.name: tool
                for tool in make_gdpval_tools(
                    workdir=workdir,
                    reference_dir=reference_dir,
                    output_head_chars=2000,
                    output_tail_chars=2000,
                )
            }

            def call(name: str, args: dict) -> str:
                result = tools[name].execute("call-1", args, lambda: False, None)
                self.assertFalse(result.is_error, tool_result_text(result))
                return tool_result_text(result)

            call("write_file", {"path": "answer.txt", "content": "alpha\nbeta\n"})
            self.assertIn("alpha", call("read_file", {"path": "answer.txt"}))
            call(
                "edit_file",
                {"path": "answer.txt", "old": "beta", "new": "gamma"},
            )
            self.assertIn("gamma", call("read_file", {"path": "answer.txt"}))
            self.assertIn(
                "alpha reference",
                call("read_file", {"path": str(reference_dir / "brief.txt")}),
            )
            self.assertIn("answer.txt", call("execute_bash", {"command": "ls"}))
            if shutil.which("rg"):
                self.assertIn("answer.txt", call("grep_files", {"pattern": "gamma"}))

    def test_gdpval_bash_fileops_profile_covers_multi_edit_and_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "workdir"
            reference_dir = Path(tmp) / "reference_task_id_files"
            tools = {
                tool.name: tool
                for tool in make_gdpval_tools(
                    workdir=workdir,
                    reference_dir=reference_dir,
                    profile="bash_fileops",
                    output_head_chars=2000,
                    output_tail_chars=2000,
                )
            }
            self.assertEqual(
                set(tools),
                {"execute_bash", "TodoWrite", "multi_edit_file", "view_image"},
            )

            def call(name: str, args: dict) -> str:
                result = tools[name].execute("call-1", args, lambda: False, None)
                self.assertFalse(result.is_error, tool_result_text(result))
                return tool_result_text(result)

            call(
                "execute_bash",
                {"command": "printf 'alpha\\nbeta\\n' > answer.txt"},
            )
            self.assertIn(
                "gamma",
                call(
                    "multi_edit_file",
                    {
                        "file_path": str(workdir / "answer.txt"),
                        "edits": [
                            {
                                "old_string": "beta\n",
                                "new_string": "gamma\n",
                            }
                        ],
                    },
                ),
            )
            self.assertEqual(
                (workdir / "answer.txt").read_text(encoding="utf-8"),
                "alpha\ngamma\n",
            )

            failed = tools["multi_edit_file"].execute(
                "call-2",
                {
                    "file_path": str(workdir / "answer.txt"),
                    "edits": [
                        {
                            "old_string": "missing",
                            "new_string": "should-not-write",
                        }
                    ],
                },
                lambda: False,
                None,
            )
            self.assertTrue(failed.is_error)
            self.assertEqual(
                (workdir / "answer.txt").read_text(encoding="utf-8"),
                "alpha\ngamma\n",
            )

            pixel_png = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
                "/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
            )
            (workdir / "pixel.png").write_bytes(pixel_png)
            self.assertIn("Image file:", call("view_image", {"path": "pixel.png"}))

    def test_extract_result_collects_manifest_and_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "workdir"
            workdir.mkdir()
            (workdir / "answer.txt").write_text("answer", encoding="utf-8")

            result = container.extract_result(workdir, {"task_id": "task-1"})

            self.assertEqual(result["status"], "solver_finished")
            self.assertEqual(result["files"][0]["relative_path"], "answer.txt")
            self.assertGreater(result["workspace_archive_bytes"], 0)
            self.assertTrue(Path(result["workspace_archive_path"]).is_file())

    def test_gdpval_agents_do_not_install_custom_context_policy(self) -> None:
        provider = Provider(id="fake", api="fake", model="fake-model")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            solver = container.build_agent(provider=provider, cwd=root / "solver")
            judge = judge_container.build_agent(provider=provider, cwd=root / "judge")
            gsb_judge = judge_gsb_container.build_agent(
                provider=provider, cwd=root / "gsb_judge"
            )

        self.assertIsNone(solver.context_policy)
        self.assertIsNone(judge.context_policy)
        self.assertIsNone(gsb_judge.context_policy)
        self.assertEqual(
            [tool.name for tool in solver.tools],
            ["execute_bash", "TodoWrite", "multi_edit_file", "view_image"],
        )
        self.assertNotIn("read_file", {tool.name for tool in solver.tools})
        self.assertNotIn("write_file", {tool.name for tool in solver.tools})
        self.assertIn("write_file", {tool.name for tool in judge.tools})
        self.assertIn("write_file", {tool.name for tool in gsb_judge.tools})

    def test_judge_scoring_normalizes_rubrics_and_weighted_score(self) -> None:
        rubrics = json.dumps(
            [
                {"score": 2, "criterion": "Has workbook"},
                {"weight": 3, "rubric_content": "Reconciles totals"},
            ]
        )
        payload = parse_judge_payload(
            """
            ```json
            {
              "rubric_results": [
                {"index": 0, "grade": 1, "explanation": "present"},
                {"index": 1, "grade": 0.5, "explanation": "partial"}
              ],
              "overall_explanation": "mixed"
            }
            ```
            """
        )

        result = score_judgment(payload, rubrics)

        self.assertEqual(
            [item["criterion"] for item in normalize_rubrics(rubrics)],
            ["Has workbook", "Reconciles totals"],
        )
        self.assertEqual(result["status"], "judged")
        self.assertAlmostEqual(result["earned_score"], 3.5)
        self.assertAlmostEqual(result["max_score"], 5.0)
        self.assertAlmostEqual(result["score"], 0.7)

    def test_gsb_judge_scoring_uses_forward_and_reverse_maps(self) -> None:
        rubrics = [{"score": 1, "criterion": "Candidate is better"}]
        payload = parse_gsb_judge_payload(
            """
            ```json
            {
              "reverse": {
                "rubrics_result": [
                  {
                    "index": 0,
                    "grade_A": 0,
                    "grade_B": 1,
                    "gsb": "A<B",
                    "grade_explanation": "candidate wins"
                  }
                ],
                "overall": {
                  "overall_explanation": "candidate wins",
                  "final_gsb": "A<B"
                }
              },
              "forward": {
                "rubrics_result": [
                  {
                    "index": 0,
                    "grade_A": 1,
                    "grade_B": 0,
                    "gsb": "A>B",
                    "grade_explanation": "candidate wins"
                  }
                ],
                "overall": {
                  "overall_explanation": "candidate wins",
                  "final_gsb": "A>B"
                }
              }
            }
            ```
            """
        )

        result = score_gsb_judgment(payload, rubrics)

        self.assertEqual(result["status"], "gsb_judged")
        self.assertAlmostEqual(result["rubrics_weighted_score_reverse"], 1.0)
        self.assertAlmostEqual(result["rubrics_weighted_score_forward"], 1.0)
        self.assertAlmostEqual(result["combined_weighted_score"], 1.0)
        self.assertAlmostEqual(result["llm_score"], 1.0)
        self.assertAlmostEqual(result["score_process"], 1.0)
        self.assertAlmostEqual(result["dcg_winrate"], 1.0)
        self.assertAlmostEqual(result["score"], 1.0)

    def test_gsb_judge_scoring_falls_back_to_raw_overall_when_no_rubrics(
        self,
    ) -> None:
        result = score_gsb_judgment(
            {
                "reverse": {
                    "rubrics_result": [],
                    "overall": {
                        "overall_explanation": "candidate is better",
                        "final_gsb": "A<B",
                    },
                }
            },
            [],
        )

        self.assertEqual(result["status"], "no_rubrics")
        self.assertEqual(result["score"], 0.75)

    def test_judge_prepare_stages_candidate_gold_and_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_workdir = root / "candidate_workdir"
            candidate_workdir.mkdir()
            (candidate_workdir / "answer.txt").write_text("answer", encoding="utf-8")
            candidate_result = container.extract_result(
                candidate_workdir, {"task_id": "task-1"}
            )
            deliverable_root = root / "deliverables"
            reference_root = root / "references"
            (deliverable_root / "gold").mkdir(parents=True)
            (reference_root / "refs").mkdir(parents=True)
            (deliverable_root / "gold" / "answer.txt").write_text(
                "gold", encoding="utf-8"
            )
            (reference_root / "refs" / "brief.txt").write_text(
                "brief", encoding="utf-8"
            )
            suite = GdpvalJudgeSuite(
                image="unused",
                deliverable_root=deliverable_root,
                reference_root=reference_root,
                network_mode=None,
            )
            judge_instance = suite.build_instance(
                {
                    "task_id": "task-1",
                    "prompt": "Make answer.",
                    "rubrics": [{"score": 1, "criterion": "Answer exists"}],
                    "deliverable_files": ["gold/answer.txt"],
                    "reference_files": ["refs/brief.txt"],
                },
                candidate_result=candidate_result,
                candidate_artifacts=RunArtifacts(
                    instance_id="task-1",
                    run_dir=root / "solver-run",
                    trajectory_path=root / "solver-run" / "out" / "trajectory.jsonl",
                    status_code=0,
                ),
            )

            judge_workdir = root / "judge" / "workdir"
            context = judge_container.prepare(judge_workdir, judge_instance)
            task = judge_container.build_task(
                judge_instance, workdir=str(judge_workdir)
            )

            candidate_dir = Path(context["candidate_dir"])
            gold_dir = Path(context["gold_dir"])
            reference_dir = Path(context["reference_dir"])
            self.assertEqual(
                (candidate_dir / "workdir" / "answer.txt").read_text(encoding="utf-8"),
                "answer",
            )
            self.assertEqual(
                (gold_dir / "gold" / "answer.txt").read_text(encoding="utf-8"),
                "gold",
            )
            self.assertEqual(
                (reference_dir / "refs" / "brief.txt").read_text(encoding="utf-8"),
                "brief",
            )
            self.assertIn(str(judge_workdir / judge_container.JUDGE_RESULT_FILE), task)

    def test_gsb_judge_prepare_and_task_use_gold_as_standard_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_workdir = root / "candidate_workdir"
            candidate_workdir.mkdir()
            (candidate_workdir / "answer.txt").write_text("answer", encoding="utf-8")
            candidate_result = container.extract_result(
                candidate_workdir, {"task_id": "task-1"}
            )
            deliverable_root = root / "deliverables"
            (deliverable_root / "gold").mkdir(parents=True)
            (deliverable_root / "gold" / "answer.txt").write_text(
                "gold", encoding="utf-8"
            )
            suite = GdpvalGsbJudgeSuite(
                image="unused",
                deliverable_root=deliverable_root,
                network_mode=None,
            )
            judge_instance = suite.build_instance(
                {
                    "task_id": "task-1",
                    "prompt": "Make answer.",
                    "rubrics": [{"score": 1, "criterion": "Answer quality"}],
                    "deliverable_files": ["gold/answer.txt"],
                },
                candidate_result=candidate_result,
                candidate_artifacts=RunArtifacts(
                    instance_id="task-1",
                    run_dir=root / "solver-run",
                    trajectory_path=root / "solver-run" / "out" / "trajectory.jsonl",
                    status_code=0,
                ),
            )

            judge_workdir = root / "judge" / "workdir"
            context = judge_gsb_container.prepare(judge_workdir, judge_instance)
            task = judge_gsb_container.build_task(
                judge_instance, workdir=str(judge_workdir)
            )

            self.assertEqual(
                (Path(context["gold_dir"]) / "gold" / "answer.txt").read_text(
                    encoding="utf-8"
                ),
                "gold",
            )
            self.assertIn("reverse: A is GOLD_DIR, B is CANDIDATE_DIR.", task)
            self.assertIn("forward: A is CANDIDATE_DIR, B is GOLD_DIR.", task)
            self.assertIn(
                str(judge_workdir / judge_gsb_container.JUDGE_RESULT_FILE), task
            )

    def test_judge_suite_local_process_oracle_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_workdir = root / "candidate_workdir"
            candidate_workdir.mkdir()
            (candidate_workdir / "answer.txt").write_text("answer", encoding="utf-8")
            candidate_result = container.extract_result(
                candidate_workdir, {"task_id": "task-1"}
            )
            suite = GdpvalJudgeSuite(image="unused", network_mode=None)
            judge_instance = suite.build_instance(
                {
                    "task_id": "task-1",
                    "prompt": "Make answer.",
                    "rubrics": [{"score": 2, "criterion": "Answer exists"}],
                },
                candidate_result=candidate_result,
                candidate_artifacts=RunArtifacts(
                    instance_id="task-1",
                    run_dir=root / "solver-run",
                    trajectory_path=root / "solver-run" / "out" / "trajectory.jsonl",
                    status_code=0,
                ),
            )

            artifacts = run_suite_instance(
                suite=suite,
                instance=judge_instance,
                backend=LocalProcessBackend(workspace=root / "judge_workdir"),
                store=LocalDirStore(root / "runs"),
                run_root=root / "runs",
                run_id="judge-smoke",
                provider="oracle",
                max_turns=1,
            )

            self.assertEqual(artifacts.status_code, 0)
            result = json.loads((artifacts.run_dir / RESULT_KEY).read_text())
            self.assertEqual(result["status"], "judged")
            self.assertEqual(result["score"], 1.0)
            self.assertEqual(result["earned_score"], 2.0)

    def test_gsb_judge_suite_local_process_oracle_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_workdir = root / "candidate_workdir"
            candidate_workdir.mkdir()
            (candidate_workdir / "answer.txt").write_text("answer", encoding="utf-8")
            candidate_result = container.extract_result(
                candidate_workdir, {"task_id": "task-1"}
            )
            deliverable_root = root / "deliverables"
            (deliverable_root / "gold").mkdir(parents=True)
            (deliverable_root / "gold" / "answer.txt").write_text(
                "gold", encoding="utf-8"
            )
            suite = GdpvalGsbJudgeSuite(
                image="unused",
                deliverable_root=deliverable_root,
                network_mode=None,
            )
            judge_instance = suite.build_instance(
                {
                    "task_id": "task-1",
                    "prompt": "Make answer.",
                    "rubrics": [{"score": 2, "criterion": "Answer exists"}],
                    "deliverable_files": ["gold/answer.txt"],
                },
                candidate_result=candidate_result,
                candidate_artifacts=RunArtifacts(
                    instance_id="task-1",
                    run_dir=root / "solver-run",
                    trajectory_path=root / "solver-run" / "out" / "trajectory.jsonl",
                    status_code=0,
                ),
            )

            artifacts = run_suite_instance(
                suite=suite,
                instance=judge_instance,
                backend=LocalProcessBackend(workspace=root / "judge_workdir"),
                store=LocalDirStore(root / "runs"),
                run_root=root / "runs",
                run_id="gsb-judge-smoke",
                provider="oracle",
                max_turns=1,
            )

            self.assertEqual(artifacts.status_code, 0)
            result = json.loads((artifacts.run_dir / RESULT_KEY).read_text())
            self.assertEqual(result["status"], "gsb_judged")
            self.assertEqual(result["judge_mode"], "gsb")
            self.assertEqual(result["score"], 0.5)
            self.assertEqual(result["combined_weighted_score"], 0.5)

    def test_run_suite_instance_local_process_fake_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            workdir = Path(tmp) / "workspace"
            suite = GdpvalSuite(
                image="unused",
                reference_root=Path(tmp),
                network_mode=None,
            )
            artifacts = run_suite_instance(
                suite=suite,
                instance={
                    "instance_id": "task-1",
                    "task_id": "task-1",
                    "prompt": "Create any file under WORKDIR.",
                },
                backend=LocalProcessBackend(workspace=workdir),
                store=LocalDirStore(root),
                run_root=root,
                run_id="smoke",
                provider="fake",
                max_turns=2,
            )

            self.assertEqual(artifacts.status_code, 0)
            result = json.loads((artifacts.run_dir / RESULT_KEY).read_text())
            self.assertEqual(result["status"], "solver_finished")
            self.assertFalse((artifacts.run_dir / "input" / "eval.json").exists())
            trace = json.loads((artifacts.run_dir / TRACE_KEY).read_text())
            self.assertEqual(trace["meta"]["suite"], "gdpval")


if __name__ == "__main__":
    unittest.main()
