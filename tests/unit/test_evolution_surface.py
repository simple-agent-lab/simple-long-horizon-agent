from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.evals.protocols import AGENT_PACKAGE_KEY
from simple_agent_lab.evolution.kernel import store
from simple_agent_lab.evolution.surface import (
    AgentSurface,
    SurfaceComponent,
    python_agent_surface,
)
from simple_agent_lab.evolution.types import Manifest, Version


DEFAULT_FILES = {
    "agent_program.py": (
        "from simple_agent_lab.core import Agent\n\n"
        "def build_agent(*, provider, cwd, base_system_prompt) -> Agent:\n"
        "    raise RuntimeError('demo')\n"
    ),
    "prompts.py": "SYSTEM_PROMPT = 'demo'\n",
    "tool_policy.py": "MAX_RETRIES = 1\n",
}


class AgentSurfaceTest(unittest.TestCase):
    def test_seed_files_are_stored_under_version_root(self) -> None:
        surface = python_agent_surface(
            default_files=DEFAULT_FILES,
            artifact_key=AGENT_PACKAGE_KEY,
        )

        self.assertIsInstance(surface, AgentSurface)
        self.assertIsInstance(surface.component("prompts"), SurfaceComponent)
        self.assertIn("agent/agent_program.py", surface.seed_files())
        self.assertIn("agent/prompts.py", surface.seed_files())

    def test_prompt_brief_describes_selected_components(self) -> None:
        surface = python_agent_surface(
            default_files=DEFAULT_FILES,
            artifact_key=AGENT_PACKAGE_KEY,
        )

        brief = surface.prompt_brief(components=("prompts",))

        self.assertIn("Python agent package", brief)
        self.assertIn("prompts", brief)
        self.assertIn("system prompts", brief)
        self.assertIn("agent/agent_program.py:build_agent", brief)

    def test_validate_edits_rejects_paths_outside_selected_components(self) -> None:
        surface = python_agent_surface(
            default_files=DEFAULT_FILES,
            artifact_key=AGENT_PACKAGE_KEY,
        )

        result = surface.validate_edits(
            {
                "agent/prompts.py": "SYSTEM_PROMPT = 'better'\n",
                "agent/tool_policy.py": "MAX_RETRIES = 3\n",
                "../escape.py": "x = 1\n",
            },
            components=("prompts",),
        )

        self.assertIn("agent/prompts.py", result.edits)
        self.assertIn("agent/tool_policy.py", result.rejected)
        self.assertIn("../escape.py", result.rejected)

    def test_validate_edits_always_rejects_unsafe_paths(self) -> None:
        surface = AgentSurface(
            id="custom",
            name="Custom surface",
            description="A custom public surface.",
            entrypoint="safe.py:build_agent",
            default_files={},
            artifact_key=AGENT_PACKAGE_KEY,
            components=(
                SurfaceComponent(
                    id="custom_component",
                    name="Custom component",
                    description="Syntax-only validation.",
                    paths=("safe.py",),
                    validators=("python_syntax",),
                ),
            ),
        )

        result = surface.validate_edits(
            {
                "safe.py": "x = 1\n",
                "../escape.py": "x = 1\n",
                "/tmp/escape.py": "x = 1\n",
            },
            components=("custom_component",),
        )

        self.assertIn("safe.py", result.edits)
        self.assertIn("../escape.py", result.rejected)
        self.assertIn("/tmp/escape.py", result.rejected)

    def test_validate_edits_rejects_everything_without_selected_components(
        self,
    ) -> None:
        surface = python_agent_surface(
            default_files=DEFAULT_FILES,
            artifact_key=AGENT_PACKAGE_KEY,
        )

        result = surface.validate_edits(
            {
                "agent/prompts.py": "def (",
                "../escape.py": "x = 1\n",
            },
            components=(),
        )

        self.assertEqual(result.edits, {})
        self.assertIn("agent/prompts.py", result.rejected)
        self.assertIn("../escape.py", result.rejected)

    def test_everything_component_allows_whole_agent_package(self) -> None:
        surface = python_agent_surface(
            default_files=DEFAULT_FILES,
            artifact_key=AGENT_PACKAGE_KEY,
        )

        result = surface.validate_edits(
            {"agent/tool_policy.py": "MAX_RETRIES = 3\n"},
            components=("everything",),
        )

        self.assertEqual(result.edits["agent/tool_policy.py"], "MAX_RETRIES = 3\n")

    def test_entrypoint_deletion_is_rejected_when_entrypoint_validator_runs(
        self,
    ) -> None:
        surface = python_agent_surface(
            default_files=DEFAULT_FILES,
            artifact_key=AGENT_PACKAGE_KEY,
        )

        result = surface.validate_edits(
            {"agent/agent_program.py": None},
            components=("everything",),
        )

        self.assertEqual(result.edits, {})
        self.assertIn("agent/agent_program.py", result.rejected)

    def test_python_validator_rejects_invalid_python(self) -> None:
        surface = python_agent_surface(
            default_files=DEFAULT_FILES,
            artifact_key=AGENT_PACKAGE_KEY,
        )

        result = surface.validate_edits(
            {"agent/prompts.py": "def ("},
            components=("prompts",),
        )

        self.assertIn("agent/prompts.py", result.rejected)

    def test_artifacts_from_version_strip_version_root(self) -> None:
        surface = python_agent_surface(
            default_files=DEFAULT_FILES,
            artifact_key=AGENT_PACKAGE_KEY,
        )
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            version = store.stage(
                workspace,
                base=None,
                edits=surface.seed_files(),
                manifest=Manifest(producer="test"),
            )

            artifacts = surface.artifacts_from_version(Version(version.dir))

        payload = json.loads(artifacts[AGENT_PACKAGE_KEY].decode("utf-8"))
        self.assertIn("agent_program.py", payload)
        self.assertIn("prompts.py", payload)
        self.assertNotIn("agent/agent_program.py", payload)

    def test_python_agent_surface_rejects_unsafe_version_roots(self) -> None:
        for version_root in ("", "/tmp", "../agent"):
            with self.subTest(version_root=version_root):
                with self.assertRaises(ValueError):
                    python_agent_surface(
                        default_files=DEFAULT_FILES,
                        artifact_key=AGENT_PACKAGE_KEY,
                        version_root=version_root,
                    )


if __name__ == "__main__":
    unittest.main()
