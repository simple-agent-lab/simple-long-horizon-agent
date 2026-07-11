from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import simple_agent_lab.config as config
from simple_agent_lab.evals.in_container import _memory_artifact_builder
from simple_agent_lab.evals.suites.swebench import container as swebench_container
from simple_agent_lab.evals.suites.swebench.patch import (
    git_diff,
    instance_base_commit,
    instance_language,
    prepare_baseline_commit,
)
from simple_agent_lab.memory import FilesystemMemory, MemoryContext
from simple_agent_lab.messages import user_message
from simple_agent_lab.state import State


class SwebenchPatchExtractTest(unittest.TestCase):
    def test_runtime_defaults_do_not_enable_submission_marker(self) -> None:
        marker_env = config.BASH_SUBMISSION_MARKER.name
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(marker_env, None)

            swebench_container._enable_swebench_runtime_defaults()

            self.assertNotIn(marker_env, os.environ)

    def test_build_task_focuses_on_workspace_solution(self) -> None:
        task = swebench_container.build_task(
            {
                "problem_statement": "Fix a parser edge case.",
                "requirements": "Keep the public API stable.",
                "interface": "No new interface.",
            },
            workdir="/app",
        )

        self.assertIn("<pr_description>\nConsider the following PR description:", task)
        self.assertIn("Fix a parser edge case.", task)
        self.assertIn("## Requirements\nKeep the public API stable.", task)
        self.assertIn("## Interface\nNo new interface.", task)
        self.assertIn("</pr_description>", task)
        self.assertIn("<instructions>\n# Task Instructions", task)
        self.assertIn("While work remains, each response should include", task)
        self.assertIn("THOUGHT text", task)
        self.assertIn("one or more bash tool calls", task)
        self.assertIn("DO NOT MODIFY: Tests, lockfiles", task)
        self.assertIn("project metadata", task)
        self.assertIn("## Completion", task)
        self.assertIn("stop using tools and give a concise final summary", task)
        self.assertIn("evaluation harness collects the workspace diff", task)
        self.assertNotIn("Keep the changes you make focused", task)
        self.assertNotIn("Remove any", task)
        self.assertNotIn("generated artifacts before finishing", task)
        self.assertNotIn("## Submission", task)
        self.assertNotIn("patch.txt", task)
        self.assertNotIn("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", task)

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

    def test_git_diff_preserves_trailing_blank_context_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp))
            tracked = repo / "module.py"
            tracked.write_text("first\nold\n\n", encoding="utf-8")
            _git(repo, "add", ".")
            _git(repo, "commit", "-m", "base")
            tracked.write_text("first\nnew\n\n", encoding="utf-8")

            patch = git_diff(repo, language="python")

        self.assertTrue(patch.endswith(" \n"), repr(patch[-20:]))
        parsed = subprocess.run(
            ["git", "apply", "--numstat", "--allow-empty", "-"],
            input=patch,
            capture_output=True,
            encoding="utf-8",
            check=False,
        )
        self.assertEqual(parsed.returncode, 0, parsed.stderr)

    def test_git_diff_raises_when_git_add_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / ".git" / "info").mkdir(parents=True)
            failed = subprocess.CompletedProcess(
                args=["git", "add", "-A"],
                returncode=1,
                stdout="",
                stderr="index is locked",
            )
            with mock.patch(
                "simple_agent_lab.evals.suites.swebench.patch.subprocess.run",
                return_value=failed,
            ):
                with self.assertRaisesRegex(RuntimeError, "git add"):
                    git_diff(repo)

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

    def test_extract_result_returns_only_collected_workspace_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp))
            tracked = repo / "app.py"
            tracked.write_text("value = 1\n", encoding="utf-8")
            _git(repo, "add", ".")
            _git(repo, "commit", "-m", "base")
            baseline = _git(repo, "rev-parse", "HEAD").stdout.strip()

            tracked.write_text("value = 2\n", encoding="utf-8")
            result = swebench_container.extract_result(
                repo,
                {"repo": "acme/widgets"},
                context={"language": "python", "baseline_commit": baseline},
            )

        self.assertIn("diff --git a/app.py b/app.py", result["model_patch"])
        self.assertEqual(result["model_patch_source"], "collected_git_diff")
        self.assertEqual(
            set(result),
            {"model_patch", "model_patch_source"},
        )

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


class SwebenchMemoryArtifactsTest(unittest.TestCase):
    def test_memory_artifacts_returns_collected_solution_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp))
            (repo / "app.py").write_text("value = 1\n", encoding="utf-8")
            _git(repo, "add", ".")
            _git(repo, "commit", "-m", "base")
            base = _git(repo, "rev-parse", "HEAD").stdout.strip()
            (repo / "app.py").write_text("value = 2\n", encoding="utf-8")

            artifacts = swebench_container.memory_artifacts(
                repo,
                {"repo": "acme/widgets"},
                context={"language": "python", "baseline_commit": base},
            )

        self.assertEqual(
            [artifact.name for artifact in artifacts], ["model_patch.diff"]
        )
        self.assertIn("diff --git a/app.py b/app.py", artifacts[0].content)
        self.assertIn("+value = 2", artifacts[0].content)

    def test_memory_artifacts_empty_when_workspace_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp))
            (repo / "app.py").write_text("value = 1\n", encoding="utf-8")
            _git(repo, "add", ".")
            _git(repo, "commit", "-m", "base")
            base = _git(repo, "rev-parse", "HEAD").stdout.strip()

            artifacts = swebench_container.memory_artifacts(
                repo,
                {"repo": "acme/widgets"},
                context={"language": "python", "baseline_commit": base},
            )

        self.assertEqual(artifacts, ())

    def test_solution_patch_lands_in_filesystem_memory_run(self) -> None:
        """End-to-end: the patch reaches runs/<id>/artifacts via the real hook.

        Uses the generic runner's ``_memory_artifact_builder`` (the same path
        ``in_container`` wires) so this locks that a memory-chain run persists the
        model's diff, not just a transcript.
        """

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            _init_repo(repo)
            (repo / "app.py").write_text("value = 1\n", encoding="utf-8")
            _git(repo, "add", ".")
            _git(repo, "commit", "-m", "base")
            base = _git(repo, "rev-parse", "HEAD").stdout.strip()
            (repo / "app.py").write_text("value = 2\n", encoding="utf-8")

            builder = _memory_artifact_builder(
                swebench_container,
                workdir=repo,
                instance={"repo": "acme/widgets"},
                context={"language": "python", "baseline_commit": base},
            )
            self.assertIsNotNone(builder)

            mem_root = Path(tmp) / "memory"
            memory = FilesystemMemory(
                root=mem_root, distiller=None, artifact_builder=builder
            )
            state = State("Fix the off-by-one bug.")
            state.record(
                user_message(
                    "Fix the off-by-one bug.",
                    target="swebench_agent",
                    kind="task",
                )
            )
            ctx = MemoryContext(
                agent="swebench_agent",
                task="Fix the off-by-one bug.",
                run_id="001_acme__widgets-a1",
                memory_name="acme-chain",
                state=state,
            )
            memory.finish(ctx)

            run_dir = mem_root / "acme-chain" / "runs" / "001_acme__widgets-a1"
            patch_file = run_dir / "artifacts" / "model_patch.diff"
            self.assertTrue(patch_file.is_file())
            self.assertIn("diff --git a/app.py b/app.py", patch_file.read_text())
            self.assertIn("model_patch.diff", (run_dir / "artifacts.md").read_text())


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
