from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

from simple_agent_lab import Agent
from simple_agent_lab.evals.protocols import AGENT_PACKAGE_KEY
from simple_agent_lab.evolution.agent_package import load_agent_package_result
from simple_agent_lab.evolution.kernel import store
from simple_agent_lab.evolution.types import Manifest, Version
from simple_agent_lab.llm import Provider

from recipes.ahe.surface import ahe_harness_surface, default_harness_files


class AheSurfaceTest(unittest.TestCase):
    def test_default_harness_has_required_files_and_entrypoint(self) -> None:
        files = default_harness_files()

        self.assertIn("agent_program.py", files)
        self.assertIn("code_agent.yaml", files)
        self.assertIn("systemprompt.md", files)
        self.assertIn("LongTermMEMORY.md", files)
        self.assertIn("ShortTermMEMORY.md", files)
        self.assertIn("tool_descriptions/bash.tool.md", files)
        self.assertIn("tools/bash.py", files)
        self.assertIn("middleware/README.md", files)
        self.assertIn("skills/README.md", files)
        self.assertIn("sub_agents/README.md", files)
        tree = ast.parse(files["agent_program.py"])
        self.assertTrue(
            any(getattr(node, "name", "") == "build_agent" for node in tree.body)
        )

    def test_surface_components_match_ahe_vocabulary(self) -> None:
        surface = ahe_harness_surface(artifact_key=AGENT_PACKAGE_KEY)

        self.assertEqual(surface.id, "ahe_harness_surface")
        self.assertEqual(surface.entrypoint, "harness/agent_program.py:build_agent")
        for component in (
            "agent_program",
            "system_prompt",
            "tool_descriptions",
            "tool_implementations",
            "middleware",
            "skills",
            "sub_agents",
            "long_term_memory",
            "everything",
        ):
            self.assertEqual(surface.component(component).id, component)

    def test_surface_rejects_paths_outside_selected_components(self) -> None:
        surface = ahe_harness_surface(artifact_key=AGENT_PACKAGE_KEY)

        result = surface.validate_edits(
            {
                "harness/systemprompt.md": "Be careful.\n",
                "harness/tools/bash.py": "x = 1\n",
                "../escape.py": "x = 1\n",
            },
            components=("system_prompt",),
        )

        self.assertIn("harness/systemprompt.md", result.edits)
        self.assertIn("harness/tools/bash.py", result.rejected)
        self.assertIn("../escape.py", result.rejected)

    def test_surface_rejects_entrypoint_deletion(self) -> None:
        surface = ahe_harness_surface(artifact_key=AGENT_PACKAGE_KEY)

        result = surface.validate_edits(
            {"harness/agent_program.py": None},
            components=("everything",),
        )

        self.assertEqual(result.edits, {})
        self.assertIn("harness/agent_program.py", result.rejected)

    def test_surface_rejects_invalid_python_syntax(self) -> None:
        surface = ahe_harness_surface(artifact_key=AGENT_PACKAGE_KEY)

        result = surface.validate_edits(
            {"harness/tools/bash.py": "def (\n"},
            components=("tool_implementations",),
        )

        self.assertEqual(result.edits, {})
        self.assertIn("harness/tools/bash.py", result.rejected)

    def test_surface_rejects_entrypoint_replacement(self) -> None:
        surface = ahe_harness_surface(artifact_key=AGENT_PACKAGE_KEY)

        result = surface.validate_edits(
            {
                "harness/agent_program.py": "def other(*, provider, cwd, base_system_prompt):\n    pass\n"
            },
            components=("everything",),
        )

        self.assertEqual(result.edits, {})
        self.assertIn("harness/agent_program.py", result.rejected)

    def test_artifact_payload_strips_harness_root(self) -> None:
        surface = ahe_harness_surface(artifact_key=AGENT_PACKAGE_KEY)
        with tempfile.TemporaryDirectory() as tmp:
            version = store.stage(
                Path(tmp),
                base=None,
                edits=surface.seed_files(),
                manifest=Manifest(producer="test"),
            )

            payload = json.loads(
                surface.artifacts_from_version(Version(version.dir))[
                    AGENT_PACKAGE_KEY
                ].decode("utf-8")
            )

        self.assertIn("agent_program.py", payload)
        self.assertNotIn("harness/agent_program.py", payload)
        self.assertIn("tools/bash.py", payload)

    def test_loaded_package_builds_agent_without_memory_files(self) -> None:
        files = default_harness_files()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded = load_agent_package_result(files, root=root)
            self.assertTrue(loaded.loaded)
            self.assertIsNotNone(loaded.builder)

            (root / "LongTermMEMORY.md").unlink()
            (root / "ShortTermMEMORY.md").unlink()

            agent = loaded.builder(
                provider=Provider(id="fake", api="fake", model="fake-model"),
                cwd=root,
                base_system_prompt="base prompt",
            )

        self.assertIsInstance(agent, Agent)
        self.assertTrue(callable(loaded.builder))

    def test_loaded_package_uses_editable_tool_and_prompt_component_files(
        self,
    ) -> None:
        files = default_harness_files()
        files["tool_descriptions/bash.tool.md"] = "# bash\n\nCUSTOM TOOL DESCRIPTION\n"
        files["middleware/README.md"] = "# Middleware\n\nCUSTOM MIDDLEWARE NOTE\n"
        files["skills/README.md"] = "# Skills\n\nCUSTOM SKILL NOTE\n"
        files["sub_agents/README.md"] = "# Sub Agents\n\nCUSTOM SUBAGENT NOTE\n"
        files["tools/bash.py"] = (
            "from __future__ import annotations\n\n"
            "from simple_agent_lab.tools import AgentTool, text_result\n\n\n"
            "def build_tool(*, cwd):\n"
            "    return AgentTool(\n"
            "        name='custom_bash',\n"
            "        description='custom executable bash replacement',\n"
            "        parameters={'type': 'object', 'properties': {}},\n"
            "        execute=lambda call_id, args, abort, update: text_result('custom'),\n"
            "    )\n"
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded = load_agent_package_result(files, root=root)
            assert loaded.builder is not None
            agent = loaded.builder(
                provider=Provider(id="fake", api="fake", model="fake-model"),
                cwd=root,
                base_system_prompt="base prompt",
            )

        self.assertEqual(agent.tools[0].name, "custom_bash")
        self.assertIn("CUSTOM TOOL DESCRIPTION", agent.system_prompt)
        self.assertIn("CUSTOM MIDDLEWARE NOTE", agent.system_prompt)
        self.assertIn("CUSTOM SKILL NOTE", agent.system_prompt)
        self.assertIn("CUSTOM SUBAGENT NOTE", agent.system_prompt)


if __name__ == "__main__":
    unittest.main()
