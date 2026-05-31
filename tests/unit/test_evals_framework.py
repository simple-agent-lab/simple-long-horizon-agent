"""Unit-smoke for the generic containerized eval framework (ADR 0017).

Runs `run_suite_instance` end-to-end against the in-memory `FakeBackend` +
`BindMountTransport` — no Docker — to prove the Suite / Backend / Transport
seams compose, and checks the SWE-bench `Suite` driver maps a Pro instance onto
a `ContainerPlan` as data (no `is_pro` branch in the runner).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

from simple_agent_lab.evals import (
    BindMountTransport,
    ContainerPlan,
    FakeBackend,
    FileTraceSink,
    StagedFile,
    Suite,
    bootstrap_script,
    run_suite_instance,
)
from simple_agent_lab.evals.backends.fake import FakeContainerHandle


class _DemoSuite:
    """Minimal suite: the entire author-facing surface for a new benchmark."""

    name = "demo"
    container_module = "demo.container"

    def container_plan(self, instance: Mapping[str, Any]) -> ContainerPlan:
        return ContainerPlan(image="demo:latest", workdir="/work")

    def sanitize_instance(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in instance.items() if k != "gold"}

    def prediction_record(
        self, instance: Mapping[str, Any], *, model_name: str, result: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "instance_id": str(instance["instance_id"]),
            "model_name_or_path": model_name,
            "answer": result.get("answer", ""),
        }


class EvalFrameworkSmokeTest(unittest.TestCase):
    def test_demo_suite_satisfies_protocol(self) -> None:
        self.assertIsInstance(_DemoSuite(), Suite)

    def test_run_suite_instance_end_to_end(self) -> None:
        suite = _DemoSuite()
        instance = {"instance_id": "demo-1", "problem": "p", "gold": "SECRET"}

        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp).resolve()
            out_dir = run_root / "run-x" / "demo-1" / "out"

            def fake_container(handle: FakeContainerHandle) -> None:
                # Stand in for the in-container runner writing through the bind
                # mount, including a live trace push via the file sink.
                sink = FileTraceSink(out_dir / "trajectory.jsonl")
                sink.emit({"schema": "v3", "trace_id": "demo.demo-1"})
                sink.close()
                (out_dir / "prediction.jsonl").write_text(
                    json.dumps(
                        suite.prediction_record(
                            instance, model_name="m", result={"answer": "42"}
                        )
                    )
                    + "\n",
                    encoding="utf-8",
                )

            backend = FakeBackend(on_start=fake_container, log_text="ok\n")
            command = (
                "bash",
                "-lc",
                bootstrap_script(runner_argv=("/agent/run_demo.py", "--x")),
            )
            artifacts = run_suite_instance(
                suite=suite,
                instance=instance,
                backend=backend,
                transport=BindMountTransport(),
                command=command,
                run_root=run_root,
                run_id="run-x",
                env={"OPENAI_MODEL": "m"},
                extra_inputs=(
                    StagedFile(data=b"print()", container_path="/agent/run_demo.py"),
                ),
            )

            # Lifecycle ran and the container was cleaned up.
            handle = backend.created[0]
            self.assertTrue(handle.started)
            self.assertTrue(handle.removed)
            self.assertEqual(artifacts.status_code, 0)
            self.assertEqual(artifacts.logs, "ok\n")

            # Transport staged the out-of-tree input file.
            self.assertEqual(handle.staged[0].container_path, "/agent/run_demo.py")

            # Sanitized instance.json dropped the gold field.
            written = json.loads(
                (artifacts.run_dir / "input" / "instance.json").read_text()
            )
            self.assertNotIn("gold", written)
            self.assertEqual(written["problem"], "p")

            # Outputs landed where RunArtifacts says.
            self.assertTrue(artifacts.trajectory_path.exists())
            prediction = json.loads(artifacts.prediction_path.read_text())
            self.assertEqual(prediction["answer"], "42")

    def test_bootstrap_script_is_suite_agnostic(self) -> None:
        script = bootstrap_script(
            runner_argv=("/agent/runner.py", "--instance-id", "x y"),
            wheelhouse_mount="/agent/wheelhouse",
        )
        self.assertIn("AGENT_PYTHON", script)
        self.assertIn("--no-index --find-links /agent/wheelhouse", script)
        # The runner argv is quoted so a value with a space stays one arg.
        self.assertIn("'x y'", script)


class SwebenchSuiteDriverTest(unittest.TestCase):
    def test_pro_instance_plan_is_data(self) -> None:
        from evals.swebench.suite import SwebenchSuite

        suite = SwebenchSuite(dataset_name="SWE-bench_Pro")
        self.assertIsInstance(suite, Suite)
        instance = {
            "instance_id": "instance_acme__widget-abc123",
            "repo": "acme/widget",
            "dockerhub_tag": "acme.widget-abc123",
        }
        plan = suite.container_plan(instance)
        self.assertEqual(plan.workdir, "/app")
        self.assertEqual(plan.shell, ("/bin/sh", "-lc"))
        self.assertEqual(plan.entrypoint, "")
        self.assertIn("sweap-images", plan.image)


if __name__ == "__main__":
    unittest.main()
