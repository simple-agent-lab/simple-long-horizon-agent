from __future__ import annotations

import base64
import io
import os
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evals.programbench.suite import ProgrambenchDynamicWorkflowSuite
from simple_agent_lab.evals.suites.programbench import dynamic_workflow_container
from simple_agent_lab.llm.env import FAKE_PROVIDER


class ProgrambenchDynamicWorkflowTest(unittest.TestCase):
    def test_suite_uses_dynamic_container_half(self) -> None:
        suite = ProgrambenchDynamicWorkflowSuite()

        self.assertEqual(suite.name, "programbench")
        self.assertEqual(
            suite.container_module,
            "simple_agent_lab.evals.suites.programbench.dynamic_workflow_container",
        )

    def test_node_runtime_reuses_the_sealed_network_prefix(self) -> None:
        sealed_prefix = (
            "unshare",
            "--user",
            "--map-root-user",
            "--net",
            "--",
        )
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            runtime = mock.Mock()
            runtime.run.return_value = mock.Mock(output="done")
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        dynamic_workflow_container.PROGRAMBENCH_DYNAMIC_WORKFLOW_SOURCE_ENV: 'return "done";',
                        dynamic_workflow_container.PROGRAMBENCH_DYNAMIC_WORKFLOW_MAX_TURNS_ENV: "7",
                    },
                ),
                mock.patch.object(
                    dynamic_workflow_container,
                    "network_isolation_prefix",
                    return_value=sealed_prefix,
                ),
                mock.patch.object(
                    dynamic_workflow_container,
                    "DynamicWorkflowRuntime",
                    return_value=runtime,
                ) as runtime_type,
            ):
                agent = dynamic_workflow_container.build_agent(
                    provider=FAKE_PROVIDER,
                    cwd=workspace,
                )
                _state, events = agent.run("task", max_turns=1)
                for _ in events:
                    pass

        options = runtime_type.call_args.kwargs["options"]
        runner = runtime_type.call_args.kwargs["runner"]
        self.assertEqual(options.process_prefix, sealed_prefix)
        self.assertEqual(runner.max_turns_cap, 7)

    def test_dynamic_worker_edits_workspace_and_submission_excludes_artifacts(
        self,
    ) -> None:
        script = r"""
phase("implement");
const edit = await agent(`Run exactly this command:
<bash>printf '#!/bin/sh\necho built\n' > compile.sh && chmod +x compile.sh && printf 'int main(void){return 0;}\n' > main.c</bash>`, {
  name: "implementer",
  maxTurns: 2,
  cacheKey: "implement"
});
return edit.output;
"""
        instance = {
            "instance_id": "testorg__calculator.abc1234",
            "language": "c",
        }
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            (workspace / "executable").write_text("reference\n", encoding="utf-8")
            (workspace / "README.md").write_text("bundled docs\n", encoding="utf-8")
            context = dynamic_workflow_container.prepare(workspace, instance)
            task = dynamic_workflow_container.build_task(
                instance, workdir=str(workspace)
            )

            with (
                mock.patch.dict(
                    os.environ,
                    {
                        dynamic_workflow_container.REQUIRE_ISOLATION_ENV: "0",
                        dynamic_workflow_container.PROGRAMBENCH_DYNAMIC_WORKFLOW_SOURCE_ENV: script,
                        dynamic_workflow_container.PROGRAMBENCH_DYNAMIC_WORKFLOW_NODE_ENV: shutil.which(
                            "node"
                        )
                        or "node",
                        dynamic_workflow_container.PROGRAMBENCH_DYNAMIC_WORKFLOW_TIMEOUT_ENV: "30",
                    },
                ),
                mock.patch(
                    "simple_agent_lab.evals.suites.programbench.container._detect_network_isolation",
                    return_value=False,
                ),
            ):
                agent = dynamic_workflow_container.build_agent(
                    provider=FAKE_PROVIDER,
                    cwd=workspace,
                )
                _state, events = agent.run(task, max_turns=1)
                for _ in events:
                    pass

            artifact_dir = dynamic_workflow_container._artifact_dir(workspace)
            result = dynamic_workflow_container.extract_result(
                workspace, instance, context=context
            )
            raw = base64.b64decode(result["submission_tar_b64"])
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as bundle:
                names = set(bundle.getnames())

        self.assertFalse(result["network_isolated"])
        self.assertFalse(artifact_dir.is_relative_to(workspace))
        self.assertIn("./compile.sh", names)
        self.assertIn("./main.c", names)
        self.assertFalse(any(".simple-agent-lab" in name for name in names))
        workflow = result["dynamic_workflow"]
        self.assertIn('phase("implement")', workflow["workflow_js"])
        self.assertEqual(len(workflow["agent_calls"]), 1)
        self.assertNotIn("trace", workflow["agent_calls"][0])
        self.assertNotIn("trace", workflow["result"]["agent_calls"][0])
        self.assertEqual(sorted(workflow["subagent_traces"]), ["implement"])


if __name__ == "__main__":
    unittest.main()
