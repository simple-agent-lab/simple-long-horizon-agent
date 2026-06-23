from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evals.swebench.suite import SwebenchDynamicWorkflowSuite
from simple_agent_lab.evals.suites.swebench import dynamic_workflow_container
from simple_agent_lab.llm.env import FAKE_PROVIDER


class SwebenchDynamicWorkflowTest(unittest.TestCase):
    def test_suite_uses_dynamic_container_half(self) -> None:
        suite = SwebenchDynamicWorkflowSuite(dataset_name="ScaleAI/SWE-bench_Pro")

        self.assertEqual(suite.name, "swebench_pro")
        self.assertEqual(
            suite.container_module,
            "simple_agent_lab.evals.suites.swebench.dynamic_workflow_container",
        )

    def test_dynamic_container_runs_bash_subagent_and_collects_patch(self) -> None:
        script = """
phase("edit");
const edit = await agent(`Run exactly this command:
<bash>printf "patched\\\\n" > file.txt</bash>`, {
  name: "editor",
  maxTurns: 2,
  cacheKey: "edit"
});
return edit.output;
"""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp) / "repo")
            tracked = repo / "file.txt"
            tracked.write_text("base\n", encoding="utf-8")
            _git(repo, "add", ".")
            _git(repo, "commit", "-m", "base")

            script_path = Path(tmp) / "workflow.js"
            script_path.write_text(script, encoding="utf-8")
            instance = {
                "instance_id": "example__repo-1",
                "problem_statement": "Change file.txt to say patched.",
                "repo_language": "python",
            }
            context = dynamic_workflow_container.prepare(repo, instance)
            task = dynamic_workflow_container.build_task(instance, workdir=str(repo))

            with mock.patch.dict(
                os.environ,
                {
                    dynamic_workflow_container.SWE_DYNAMIC_WORKFLOW_SCRIPT_ENV: str(
                        script_path
                    ),
                    dynamic_workflow_container.SWE_DYNAMIC_WORKFLOW_TIMEOUT_ENV: "30",
                },
            ):
                agent = dynamic_workflow_container.build_agent(
                    provider=FAKE_PROVIDER,
                    cwd=repo,
                )
                _state, events = agent.run(task, max_turns=1)
                for _ in events:
                    pass

            result = dynamic_workflow_container.extract_result(
                repo, instance, context=context
            )

        self.assertIn("diff --git a/file.txt b/file.txt", result["model_patch"])
        self.assertIn("-base", result["model_patch"])
        self.assertIn("+patched", result["model_patch"])
        self.assertNotIn(".simple-agent-lab", result["model_patch"])
        self.assertNotIn("dynamic_workflow", result["model_patch"])
        workflow = result["dynamic_workflow"]
        self.assertIn('phase("edit")', workflow["workflow_js"])
        self.assertEqual(len(workflow["agent_calls"]), 1)
        self.assertEqual(sorted(workflow["subagent_traces"]), ["edit"])


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    _git(path, "init")
    _git(path, "config", "user.email", "test@example.invalid")
    _git(path, "config", "user.name", "Test User")
    _git(path, "config", "commit.gpgsign", "false")
    return path


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
