from pathlib import Path
import tempfile
import unittest

from simple_agent_lab.evolution import agent_package


ROOT = Path(__file__).resolve().parents[2]


class AgentPackageTest(unittest.TestCase):
    def test_default_package_has_entry_and_builder(self):
        files = agent_package.default_agent_package()
        self.assertIn(agent_package.ENTRY_MODULE_FILENAME, files)
        self.assertIn("def build_agent", files[agent_package.ENTRY_MODULE_FILENAME])
        self.assertNotIn("swebench", files[agent_package.ENTRY_MODULE_FILENAME].lower())

    def test_load_agent_package_returns_callable(self):
        with self.subTest("valid package"):
            with tempfile.TemporaryDirectory() as tmp:
                builder = agent_package.load_agent_package(
                    agent_package.default_agent_package(), root=Path(tmp)
                )
        self.assertTrue(callable(builder))

    def test_load_agent_package_supports_sibling_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = agent_package.load_agent_package(
                {
                    agent_package.ENTRY_MODULE_FILENAME: (
                        "from prompts import VALUE\n\n"
                        "def build_agent(**kwargs):\n"
                        "    return VALUE\n"
                    ),
                    "prompts.py": "VALUE = 123\n",
                },
                root=Path(tmp),
            )

        self.assertTrue(callable(builder))
        assert builder is not None
        self.assertEqual(builder(), 123)

    def test_load_agent_package_result_preserves_import_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = agent_package.load_agent_package_result(
                {
                    agent_package.ENTRY_MODULE_FILENAME: (
                        "from missing_helper import VALUE\n\n"
                        "def build_agent(**kwargs):\n"
                        "    return VALUE\n"
                    )
                },
                root=Path(tmp),
            )

        self.assertIsNone(result.builder)
        self.assertFalse(result.loaded)
        self.assertIn("ModuleNotFoundError", result.error)
        self.assertIn("missing_helper", result.error)

    def test_load_agent_package_returns_none_on_bad_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = {agent_package.ENTRY_MODULE_FILENAME: "def build_agent(:\n"}
            self.assertIsNone(agent_package.load_agent_package(bad, root=Path(tmp)))

    def test_load_agent_package_returns_none_when_builder_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = {agent_package.ENTRY_MODULE_FILENAME: "X = 1\n"}
            self.assertIsNone(agent_package.load_agent_package(files, root=Path(tmp)))

    def test_agent_package_helper_is_not_swebench_specific(self):
        self.assertFalse(
            (
                ROOT
                / "src"
                / "simple_agent_lab"
                / "evals"
                / "suites"
                / "swebench"
                / "agent_package.py"
            ).exists()
        )


if __name__ == "__main__":
    unittest.main()
