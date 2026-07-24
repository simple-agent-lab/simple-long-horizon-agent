from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.onemillion.suite import OneMillionDynamicWorkflowSuite  # noqa: E402
from simple_agent_lab.evals import (  # noqa: E402
    LocalDirStore,
    LocalProcessBackend,
    run_suite_instance,
)


class OneMillionDynamicWorkflowBenchTest(unittest.TestCase):
    def test_fake_provider_runs_dynamic_workflow_through_suite(self) -> None:
        instance = {
            "instance_id": "case_dyn",
            "case_id": "dyn",
            "prompt": "Explain decorators in Python.",
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "runs"
            artifacts = run_suite_instance(
                suite=OneMillionDynamicWorkflowSuite(in_env_scoring=False),
                instance=instance,
                backend=LocalProcessBackend(),
                store=LocalDirStore(run_root),
                run_root=run_root,
                run_id="dyn-smoke",
                provider="fake",
                max_turns=1,
            )
            self.assertEqual(artifacts.status_code, 0, artifacts.logs)
            result_path = artifacts.run_dir / "out" / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertIn("model_response", result)
        workflow = result.get("dynamic_workflow") or {}
        self.assertIn("workflow_js", workflow)
        self.assertIn("await agent", workflow["workflow_js"])
        self.assertGreaterEqual(len(workflow.get("agent_calls") or []), 3)
        self.assertIn("trace", workflow["agent_calls"][0])
        self.assertEqual(
            len(workflow.get("subagent_traces") or {}),
            len(workflow.get("agent_calls") or []),
        )
        kinds = {record.get("kind") for record in workflow.get("journal") or []}
        self.assertIn("workflow_completed", kinds)
        self.assertIn("agent_completed", kinds)


if __name__ == "__main__":
    unittest.main()
