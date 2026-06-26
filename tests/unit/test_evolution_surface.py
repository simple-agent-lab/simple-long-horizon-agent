from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.evolution.kernel import store
from simple_agent_lab.evolution.surface import (
    AgentSurface,
    SurfaceComponent,
)
from simple_agent_lab.evolution.types import Manifest, Version


ARTIFACT_KEY = "input/source.json"


class AgentSurfaceTest(unittest.TestCase):
    def test_seed_files_preserve_surface_paths(self) -> None:
        surface = AgentSurface(
            id="custom",
            name="Custom source",
            description="A custom source surface.",
            entrypoint="src/pkg/__init__.py",
            default_files={"src/pkg/__init__.py": "VALUE = 1\n"},
            artifact_key=ARTIFACT_KEY,
            components=(
                SurfaceComponent(
                    id="everything",
                    name="Everything",
                    description="All source files.",
                    paths=("src/pkg/**",),
                ),
            ),
        )

        self.assertIsInstance(surface, AgentSurface)
        self.assertIsInstance(surface.component("everything"), SurfaceComponent)
        self.assertIn("src/pkg/__init__.py", surface.seed_files())

    def test_prompt_brief_describes_selected_components(self) -> None:
        surface = AgentSurface(
            id="custom",
            name="Custom source",
            description="A custom source surface.",
            entrypoint="src/pkg/__init__.py",
            default_files={},
            artifact_key=ARTIFACT_KEY,
            components=(
                SurfaceComponent(
                    id="prompts",
                    name="Prompts",
                    description="system prompts, task framing, and response policy.",
                    paths=("src/pkg/prompts.py",),
                ),
            ),
        )

        brief = surface.prompt_brief(components=("prompts",))

        self.assertIn("Custom source", brief)
        self.assertIn("prompts", brief)
        self.assertIn("system prompts", brief)
        self.assertIn("src/pkg/__init__.py", brief)

    def test_validate_edits_rejects_paths_outside_selected_components(self) -> None:
        surface = AgentSurface(
            id="custom",
            name="Custom source",
            description="A custom source surface.",
            entrypoint="src/pkg/__init__.py",
            default_files={},
            artifact_key=ARTIFACT_KEY,
            components=(
                SurfaceComponent(
                    id="prompts",
                    name="Prompts",
                    description="Prompt files.",
                    paths=("src/pkg/prompts.py",),
                ),
            ),
        )

        result = surface.validate_edits(
            {
                "src/pkg/prompts.py": "SYSTEM_PROMPT = 'better'\n",
                "src/pkg/tool_policy.py": "MAX_RETRIES = 3\n",
                "../escape.py": "x = 1\n",
            },
            components=("prompts",),
        )

        self.assertIn("src/pkg/prompts.py", result.edits)
        self.assertIn("src/pkg/tool_policy.py", result.rejected)
        self.assertIn("../escape.py", result.rejected)

    def test_validate_edits_always_rejects_unsafe_paths(self) -> None:
        surface = AgentSurface(
            id="custom",
            name="Custom surface",
            description="A custom public surface.",
            entrypoint="safe.py:build_agent",
            default_files={},
            artifact_key=ARTIFACT_KEY,
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

    def test_validate_edits_always_rejects_unmatched_component_paths(self) -> None:
        surface = AgentSurface(
            id="custom",
            name="Custom surface",
            description="A custom public surface.",
            entrypoint="safe.py:build_agent",
            default_files={},
            artifact_key=ARTIFACT_KEY,
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
                "other.py": "x = 1\n",
            },
            components=("custom_component",),
        )

        self.assertIn("safe.py", result.edits)
        self.assertIn("other.py", result.rejected)

    def test_validate_edits_rejects_everything_without_selected_components(
        self,
    ) -> None:
        surface = AgentSurface(
            id="custom",
            name="Custom source",
            description="A custom source surface.",
            entrypoint="src/pkg/__init__.py",
            default_files={},
            artifact_key=ARTIFACT_KEY,
            components=(
                SurfaceComponent(
                    id="prompts",
                    name="Prompts",
                    description="Prompt files.",
                    paths=("src/pkg/prompts.py",),
                ),
            ),
        )

        result = surface.validate_edits(
            {
                "src/pkg/prompts.py": "def (",
                "../escape.py": "x = 1\n",
            },
            components=(),
        )

        self.assertEqual(result.edits, {})
        self.assertIn("src/pkg/prompts.py", result.rejected)
        self.assertIn("../escape.py", result.rejected)

    def test_everything_component_allows_whole_surface(self) -> None:
        surface = AgentSurface(
            id="custom",
            name="Custom source",
            description="A custom source surface.",
            entrypoint="src/pkg/__init__.py",
            default_files={},
            artifact_key=ARTIFACT_KEY,
            components=(
                SurfaceComponent(
                    id="everything",
                    name="Everything",
                    description="All files.",
                    paths=("src/pkg/**",),
                ),
            ),
        )

        result = surface.validate_edits(
            {"src/pkg/tool_policy.py": "MAX_RETRIES = 3\n"},
            components=("everything",),
        )

        self.assertEqual(result.edits["src/pkg/tool_policy.py"], "MAX_RETRIES = 3\n")

    def test_excluded_paths_reject_even_when_component_matches(self) -> None:
        surface = AgentSurface(
            id="custom",
            name="Custom source",
            description="A custom source surface.",
            entrypoint="src/pkg/__init__.py",
            default_files={},
            artifact_key=ARTIFACT_KEY,
            components=(
                SurfaceComponent(
                    id="everything",
                    name="Everything",
                    description="All files.",
                    paths=("src/pkg/**",),
                ),
            ),
            excluded_paths=("src/pkg/protected/**",),
        )

        result = surface.validate_edits(
            {
                "src/pkg/module.py": "x = 1\n",
                "src/pkg/protected/kernel.py": "x = 2\n",
            },
            components=("everything",),
        )

        self.assertIn("src/pkg/module.py", result.edits)
        self.assertIn("src/pkg/protected/kernel.py", result.rejected)

    def test_entrypoint_deletion_is_rejected_when_entrypoint_validator_runs(
        self,
    ) -> None:
        surface = AgentSurface(
            id="custom",
            name="Custom source",
            description="A custom source surface.",
            entrypoint="src/pkg/agent_program.py:build_agent",
            default_files={},
            artifact_key=ARTIFACT_KEY,
            components=(
                SurfaceComponent(
                    id="everything",
                    name="Everything",
                    description="All files.",
                    paths=("src/pkg/**",),
                    validators=("path_allowed", "python_syntax", "entrypoint_exists"),
                ),
            ),
        )

        result = surface.validate_edits(
            {"src/pkg/agent_program.py": None},
            components=("everything",),
        )

        self.assertEqual(result.edits, {})
        self.assertIn("src/pkg/agent_program.py", result.rejected)

    def test_entrypoint_replacement_without_symbol_is_rejected(self) -> None:
        surface = AgentSurface(
            id="custom",
            name="Custom source",
            description="A custom source surface.",
            entrypoint="src/pkg/agent_program.py:build_agent",
            default_files={},
            artifact_key=ARTIFACT_KEY,
            components=(
                SurfaceComponent(
                    id="everything",
                    name="Everything",
                    description="All files.",
                    paths=("src/pkg/**",),
                    validators=("path_allowed", "python_syntax", "entrypoint_exists"),
                ),
            ),
        )

        result = surface.validate_edits(
            {"src/pkg/agent_program.py": "def other() -> None:\n    pass\n"},
            components=("everything",),
        )

        self.assertEqual(result.edits, {})
        self.assertIn("src/pkg/agent_program.py", result.rejected)

    def test_python_validator_rejects_invalid_python(self) -> None:
        surface = AgentSurface(
            id="custom",
            name="Custom source",
            description="A custom source surface.",
            entrypoint="src/pkg/prompts.py",
            default_files={},
            artifact_key=ARTIFACT_KEY,
            components=(
                SurfaceComponent(
                    id="prompts",
                    name="Prompts",
                    description="Prompt files.",
                    paths=("src/pkg/prompts.py",),
                ),
            ),
        )

        result = surface.validate_edits(
            {"src/pkg/prompts.py": "def ("},
            components=("prompts",),
        )

        self.assertIn("src/pkg/prompts.py", result.rejected)

    def test_artifacts_from_version_keep_surface_paths(self) -> None:
        surface = AgentSurface(
            id="custom",
            name="Custom source",
            description="A custom source surface.",
            entrypoint="src/pkg/__init__.py",
            default_files={"src/pkg/__init__.py": "VALUE = 1\n"},
            artifact_key=ARTIFACT_KEY,
            components=(
                SurfaceComponent(
                    id="everything",
                    name="Everything",
                    description="All files.",
                    paths=("src/pkg/**",),
                ),
            ),
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

        payload = json.loads(artifacts[ARTIFACT_KEY].decode("utf-8"))
        self.assertIn("src/pkg/__init__.py", payload)

    def test_agent_surface_rejects_unsafe_artifact_keys(self) -> None:
        for artifact_key in ("", "/tmp/agent.json", "../agent.json"):
            with self.subTest(artifact_key=artifact_key):
                with self.assertRaises(ValueError):
                    AgentSurface(
                        id="custom",
                        name="Custom surface",
                        description="A custom public surface.",
                        entrypoint="safe.py:build_agent",
                        default_files={},
                        artifact_key=artifact_key,
                        components=(
                            SurfaceComponent(
                                id="custom_component",
                                name="Custom component",
                                description="Syntax-only validation.",
                                paths=("safe.py",),
                            ),
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
