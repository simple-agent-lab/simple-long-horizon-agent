from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.evals.suites.swebench import container as swebench_container
from simple_agent_lab.evals.suites.swebench.patch import (
    git_diff,
    instance_base_commit,
    instance_language,
    prepare_baseline_commit,
)
from simple_agent_lab.messages import ToolResultBlock, tool_results_message
from simple_agent_lab.state import State


class SwebenchPatchExtractTest(unittest.TestCase):
    def test_build_task_states_submission_prompt_contract(self) -> None:
        task = swebench_container.build_task(
            {
                "problem_statement": "Fix a parser edge case.",
                "requirements": "Keep the public API stable.",
                "interface": "No new interface.",
            },
            workdir="/app",
        )

        self.assertIn(
            "Before each bash call, briefly state what you are checking or changing.",
            task,
        )
        self.assertIn("Modify configuration or project metadata only", task)
        self.assertIn(
            "plus any\n   config or metadata files that the issue explicitly requires.",
            task,
        )
        self.assertIn("If you modify\n`patch.txt` after inspecting it", task)
        self.assertIn("After submitting, do not continue", task)
        self.assertIn("<problem_statement>\nFix a parser edge case.", task)
        self.assertIn("</problem_statement>", task)
        self.assertIn("<requirements>\nKeep the public API stable.", task)
        self.assertIn("<interface>\nNo new interface.", task)

    def test_git_diff_excludes_generated_build_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp))
            (repo / "pkg").mkdir()
            (repo / "pkg" / "core.py").write_text("value = 1\n", encoding="utf-8")
            _git(repo, "add", ".")
            _git(repo, "commit", "-m", "base")

            (repo / "pkg" / "core.py").write_text("value = 2\n", encoding="utf-8")
            (repo / "pkg" / "extra.py").write_text("extra = True\n", encoding="utf-8")
            (repo / "build" / "lib").mkdir(parents=True)
            (repo / "build" / "lib" / "generated.py").write_text(
                "generated = True\n",
                encoding="utf-8",
            )

            patch = git_diff(repo, language="python")

        self.assertIn("diff --git a/pkg/core.py b/pkg/core.py", patch)
        self.assertIn("diff --git a/pkg/extra.py b/pkg/extra.py", patch)
        self.assertNotIn("build/lib/generated.py", patch)
        self.assertNotIn(".gitignore", patch)

    def test_git_diff_uses_language_specific_multilingual_ignores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp))
            (repo / "src").mkdir()
            (repo / "src" / "app.ts").write_text(
                "export const x = 1;\n", encoding="utf-8"
            )
            _git(repo, "add", ".")
            _git(repo, "commit", "-m", "base")

            (repo / "src" / "app.ts").write_text(
                "export const x = 2;\n", encoding="utf-8"
            )
            (repo / "dist").mkdir()
            (repo / "dist" / "app.js").write_text("compiled\n", encoding="utf-8")
            (repo / "src" / "app.d.ts").write_text(
                "export declare const x: number;\n",
                encoding="utf-8",
            )

            patch = git_diff(repo, language="typescript")

        self.assertIn("diff --git a/src/app.ts b/src/app.ts", patch)
        self.assertNotIn("dist/app.js", patch)
        self.assertNotIn("src/app.d.ts", patch)

    def test_git_diff_can_compare_against_base_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp))
            tracked = repo / "module.py"
            tracked.write_text("a = 1\n", encoding="utf-8")
            _git(repo, "add", ".")
            _git(repo, "commit", "-m", "base")
            base_commit = _git(repo, "rev-parse", "HEAD").stdout.strip()

            tracked.write_text("a = 2\n", encoding="utf-8")
            _git(repo, "add", ".")
            _git(repo, "commit", "-m", "intermediate")
            tracked.write_text("a = 3\n", encoding="utf-8")

            patch = git_diff(repo, language="python", commit=base_commit)

        self.assertIn("-a = 1", patch)
        self.assertIn("+a = 3", patch)

    def test_baseline_commit_excludes_pre_agent_environment_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp))
            (repo / "setup.py").write_text("deps = ['old']\n", encoding="utf-8")
            (repo / "pkg").mkdir()
            (repo / "pkg" / "core.py").write_text("value = 1\n", encoding="utf-8")
            _git(repo, "add", ".")
            _git(repo, "commit", "-m", "base")

            (repo / "setup.py").write_text("deps = ['env-pinned']\n", encoding="utf-8")
            (repo / "build").mkdir()
            (repo / "build" / "generated.py").write_text(
                "generated = True\n",
                encoding="utf-8",
            )

            baseline = prepare_baseline_commit(repo, language="python")
            (repo / "pkg" / "core.py").write_text("value = 2\n", encoding="utf-8")
            patch = git_diff(repo, language="python", commit=baseline)

        self.assertIn("diff --git a/pkg/core.py b/pkg/core.py", patch)
        self.assertNotIn("setup.py", patch)
        self.assertNotIn("build/generated.py", patch)

    def test_extract_result_keeps_collected_and_model_submitted_patches(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp))
            tracked = repo / "app.py"
            tracked.write_text("value = 1\n", encoding="utf-8")
            _git(repo, "add", ".")
            _git(repo, "commit", "-m", "base")
            baseline = _git(repo, "rev-parse", "HEAD").stdout.strip()

            tracked.write_text("value = 2\n", encoding="utf-8")
            state = State("task")
            state.record(
                tool_results_message(
                    [
                        ToolResultBlock(
                            tool_call_id="call-1",
                            tool_name="bash",
                        )
                    ],
                    target="swebench_agent",
                    sidecar={
                        "details": {
                            "call-1": {
                                "submission": (
                                    "diff --git a/app.py b/app.py\n"
                                    "--- a/app.py\n"
                                    "+++ b/app.py\n"
                                    "@@\n"
                                    "-value = 1\n"
                                    "+value = 2\n"
                                )
                            }
                        }
                    },
                )
            )

            result = swebench_container.extract_result(
                repo,
                {"repo": "acme/widgets"},
                context={"language": "python", "baseline_commit": baseline},
                state=state,
            )

        self.assertIn("diff --git a/app.py b/app.py", result["model_patch"])
        self.assertEqual(result["model_patch_source"], "collected_git_diff")
        self.assertIn("diff --git a/app.py b/app.py", result["model_submitted_patch"])
        self.assertEqual(result["model_submitted_patch_source"], "tool_submission")

    def test_extract_result_reads_patch_txt_when_no_submission_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp))
            (repo / "app.py").write_text("value = 1\n", encoding="utf-8")
            _git(repo, "add", ".")
            _git(repo, "commit", "-m", "base")
            (repo / "patch.txt").write_text(
                "diff --git a/app.py b/app.py\n", encoding="utf-8"
            )

            result = swebench_container.extract_result(
                repo,
                {"repo": "acme/widgets"},
                context={"language": "python"},
            )

        self.assertEqual(
            result["model_submitted_patch"], "diff --git a/app.py b/app.py\n"
        )
        self.assertEqual(result["model_submitted_patch_source"], "patch_txt")

    def test_instance_helpers_default_verified_to_python_and_read_multilingual(
        self,
    ) -> None:
        self.assertEqual(instance_language({}), "python")
        self.assertEqual(instance_language({"language": "TypeScript"}), "ts")
        self.assertEqual(instance_language({"repo_language": "JavaScript"}), "js")
        self.assertEqual(instance_language({"repo": "facebook/docusaurus"}), "ts")
        self.assertEqual(instance_language({"repo": "BABEL/BABEL"}), "ts")
        self.assertEqual(
            instance_language({"instance_id": "tokio-rs__tokio-12345"}), "rust"
        )
        self.assertEqual(
            instance_language(
                {
                    "language": "JavaScript",
                    "repo": "facebook/docusaurus",
                }
            ),
            "js",
        )
        self.assertEqual(instance_base_commit({"base_commit": "abc123"}), "abc123")
        self.assertEqual(instance_base_commit({"base": {"sha": "def456"}}), "def456")


def _init_repo(path: Path) -> Path:
    _git(path, "init")
    _git(path, "config", "user.email", "test@example.invalid")
    _git(path, "config", "user.name", "Test User")
    # Keep the throwaway repo hermetic: never invoke the host's commit
    # signer (some sandboxes route gpgsign through a server that rejects
    # these temp commits and aborts `git commit`).
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
