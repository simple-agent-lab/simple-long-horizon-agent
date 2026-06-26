from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.evolution.source_tree import (
    CANDIDATE_PACKAGE,
    CANDIDATE_SOURCE_CONTAINER_SRC,
    CANDIDATE_SRC,
    SOURCE_ROOT,
    candidate_source_artifacts,
    cheap_validate_source_tree,
    source_tree_agent_surface,
    source_tree_surface,
    validate_source_tree_edits,
)


class SourceTreeEvolutionTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo_root = Path(tmp.name)
        package = self.repo_root / SOURCE_ROOT
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("VALUE = 1\n")
        (package / "core.py").write_text("def run() -> str:\n    return 'ok'\n")
        (package / "tools").mkdir()
        (package / "tools" / "bash.py").write_text(
            "def tool() -> str:\n    return 'bash'\n"
        )
        (package / "evolution").mkdir()
        (package / "evolution" / "kernel.py").write_text("KERNEL = True\n")
        (package / "README.md").write_text("# Package notes\n")
        (package / "__pycache__").mkdir()
        (package / "__pycache__" / "core.cpython-314.pyc").write_bytes(b"cache")

    def test_surface_includes_package_files(self) -> None:
        surface = source_tree_surface(self.repo_root)

        self.assertIn(SOURCE_ROOT + "/__init__.py", surface)
        self.assertIn("VALUE = 1", surface)
        self.assertIn(SOURCE_ROOT + "/core.py", surface)
        self.assertIn("def run() -> str:", surface)
        self.assertIn(SOURCE_ROOT + "/README.md", surface)
        self.assertNotIn("__pycache__", surface)

    def test_candidate_source_artifacts_overlay_valid_edit(self) -> None:
        artifacts = candidate_source_artifacts(
            self.repo_root,
            {SOURCE_ROOT + "/core.py": "def run() -> str:\n    return 'better'\n"},
        )

        self.assertEqual(CANDIDATE_PACKAGE, CANDIDATE_SRC + "/simple_agent_lab")
        self.assertEqual(
            artifacts[CANDIDATE_PACKAGE + "/core.py"],
            b"def run() -> str:\n    return 'better'\n",
        )
        self.assertEqual(artifacts[CANDIDATE_PACKAGE + "/__init__.py"], b"VALUE = 1\n")
        self.assertNotIn(
            CANDIDATE_PACKAGE + "/__pycache__/core.cpython-314.pyc", artifacts
        )

    def test_agent_surface_seeds_python_source_files(self) -> None:
        surface = source_tree_agent_surface(self.repo_root)

        self.assertEqual(surface.id, "source_tree")
        self.assertEqual(surface.artifact_key, "source_tree")
        self.assertEqual(
            CANDIDATE_SOURCE_CONTAINER_SRC,
            "/agent/run/input/source_tree/src",
        )
        self.assertEqual(
            surface.default_files[SOURCE_ROOT + "/__init__.py"], "VALUE = 1\n"
        )
        self.assertEqual(
            surface.default_files[SOURCE_ROOT + "/core.py"],
            "def run() -> str:\n    return 'ok'\n",
        )
        self.assertNotIn(SOURCE_ROOT + "/README.md", surface.default_files)
        self.assertEqual(surface.component("agent_runtime").id, "agent_runtime")
        self.assertEqual(surface.component("tools").id, "tools")
        self.assertEqual(surface.component("everything").id, "everything")

    def test_agent_surface_can_exclude_protected_source_paths(self) -> None:
        surface = source_tree_agent_surface(
            self.repo_root,
            exclude=(SOURCE_ROOT + "/evolution/**",),
        )

        self.assertNotIn(SOURCE_ROOT + "/evolution/kernel.py", surface.default_files)
        result = surface.validate_edits(
            {
                SOURCE_ROOT + "/core.py": "def run() -> str:\n    return 'new'\n",
                SOURCE_ROOT + "/evolution/kernel.py": "KERNEL = False\n",
            },
            components=("everything",),
        )

        self.assertIn(SOURCE_ROOT + "/core.py", result.edits)
        self.assertIn(SOURCE_ROOT + "/evolution/kernel.py", result.rejected)

    def test_candidate_artifacts_and_surface_skip_symlinked_files(self) -> None:
        outside = self.repo_root / "outside_secret.py"
        outside.write_text("SECRET = 'outside'\n")
        symlink = self.repo_root / SOURCE_ROOT / "linked_secret.py"
        try:
            symlink.symlink_to(outside)
        except (NotImplementedError, OSError) as exc:
            self.skipTest(f"symlinks are not supported here: {exc}")

        artifacts = candidate_source_artifacts(self.repo_root, {})
        surface = source_tree_surface(self.repo_root)

        self.assertNotIn(CANDIDATE_PACKAGE + "/linked_secret.py", artifacts)
        self.assertNotIn("linked_secret.py", surface)
        self.assertNotIn("SECRET = 'outside'", surface)

    def test_artifact_paths_stay_under_candidate_package(self) -> None:
        artifacts = candidate_source_artifacts(
            self.repo_root,
            {SOURCE_ROOT + "/new_module.py": "ANSWER = 42\n"},
        )

        for path in artifacts:
            self.assertTrue(
                path.startswith(CANDIDATE_PACKAGE + "/"),
                msg=f"artifact path escaped candidate package: {path}",
            )

    def test_validate_rejects_outside_path(self) -> None:
        errors = validate_source_tree_edits({"README.md": "not allowed\n"})

        self.assertTrue(
            any("outside src/simple_agent_lab" in error for error in errors)
        )

    def test_validate_rejects_traversal_and_absolute_paths(self) -> None:
        errors = validate_source_tree_edits(
            {
                SOURCE_ROOT + "/../escape.py": "x = 1\n",
                "/tmp/escape.py": "x = 1\n",
            }
        )

        self.assertTrue(any(".." in error for error in errors))
        self.assertTrue(any("absolute" in error for error in errors))

    def test_validate_rejects_non_python_and_cache_paths(self) -> None:
        errors = validate_source_tree_edits(
            {
                SOURCE_ROOT + "/notes.md": "# no\n",
                SOURCE_ROOT + "/__pycache__/core.py": "x = 1\n",
            }
        )

        self.assertTrue(any("only .py" in error for error in errors))
        self.assertTrue(any("__pycache__" in error for error in errors))

    def test_candidate_artifacts_raise_on_invalid_edit(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside src/simple_agent_lab"):
            candidate_source_artifacts(self.repo_root, {"outside.py": "x = 1\n"})

    def test_cheap_validate_source_tree_rejects_compile_failure(self) -> None:
        with self.assertRaisesRegex((ValueError, RuntimeError), "compile"):
            cheap_validate_source_tree(
                self.repo_root,
                {SOURCE_ROOT + "/broken.py": "def nope(:\n    pass\n"},
            )


if __name__ == "__main__":
    unittest.main()
