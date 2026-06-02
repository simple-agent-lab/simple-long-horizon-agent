from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import cast

from simple_agent_lab.tools import AgentTool, ToolResult, tool_result_text
from simple_agent_lab.tools.read import (
    DEFAULT_READ_MAX_CHARS,
    READ_TOOL_NAME,
    make_read_tool,
)


def _execute(tool: AgentTool, args: dict[str, object]) -> ToolResult:
    execute = cast(object, tool.execute)
    if not callable(execute):
        raise AssertionError("read tool has no execute function")
    return execute("call_1", args, lambda: False, None)


class ReadToolTest(unittest.TestCase):
    def test_reads_whole_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "note.md").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
            result = _execute(make_read_tool(cwd=tmp), {"path": "note.md"})
        self.assertFalse(result.is_error)
        self.assertIn("alpha", tool_result_text(result))
        self.assertIn("gamma", tool_result_text(result))

    def test_reads_line_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "f.txt").write_text("l1\nl2\nl3\nl4\n", encoding="utf-8")
            result = _execute(
                make_read_tool(cwd=tmp), {"path": "f.txt", "offset": 2, "limit": 2}
            )
        text = tool_result_text(result)
        self.assertIn("l2", text)
        self.assertIn("l3", text)
        self.assertNotIn("l4", text)

    def test_missing_path_is_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = _execute(make_read_tool(cwd=tmp), {})
        self.assertTrue(result.is_error)
        self.assertIn("Missing required read argument", tool_result_text(result))

    def test_missing_target_is_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = _execute(make_read_tool(cwd=tmp), {"path": "nope.txt"})
        self.assertTrue(result.is_error)
        self.assertIn("No such file or directory", tool_result_text(result))

    def test_reads_directory_listing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            skill = Path(tmp) / "s"
            (skill / "scripts").mkdir(parents=True)
            (skill / "SKILL.md").write_text("x", encoding="utf-8")
            (skill / "scripts" / "run.py").write_text("y", encoding="utf-8")
            result = _execute(make_read_tool(cwd=tmp), {"path": "s"})
        text = tool_result_text(result)
        self.assertFalse(result.is_error)
        self.assertIn("SKILL.md", text)
        self.assertIn("scripts/run.py", text)
        self.assertEqual(result.details["kind"], "directory")

    def test_budget_is_more_generous_than_bash(self) -> None:
        from simple_agent_lab.tools.bash import DEFAULT_BASH_MAX_OUTPUT_CHARS

        self.assertGreater(DEFAULT_READ_MAX_CHARS, DEFAULT_BASH_MAX_OUTPUT_CHARS)

    def test_large_file_truncates_with_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            big = "x" * (DEFAULT_READ_MAX_CHARS + 500)
            (Path(tmp) / "big.txt").write_text(big, encoding="utf-8")
            result = _execute(make_read_tool(cwd=tmp), {"path": "big.txt"})
        text = tool_result_text(result)
        self.assertIn("truncated", text)
        self.assertTrue(result.details["truncated"])

    def test_schema_is_strict(self) -> None:
        tool = make_read_tool(cwd=".")
        self.assertEqual(tool.name, READ_TOOL_NAME)
        self.assertEqual(tool.parameters["required"], ["path"])
        self.assertFalse(tool.parameters["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
