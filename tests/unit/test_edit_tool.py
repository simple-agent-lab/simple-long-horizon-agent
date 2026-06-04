from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import cast

from simple_agent_lab import AgentTool, ToolResult, tool_result_text
from simple_agent_lab.tools.edit import (
    EDIT_TOOL_NAME,
    edit_file,
    make_edit_tool,
)


class EditToolTest(unittest.TestCase):
    def test_replaces_a_unique_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "f.txt"
            target.write_text("alpha\nbeta\ngamma\n")
            result = _execute(
                make_edit_tool(cwd=tmp),
                {"path": "f.txt", "old_string": "beta", "new_string": "BETA"},
            )
            written = target.read_text()

        self.assertFalse(result.is_error)
        self.assertEqual(written, "alpha\nBETA\ngamma\n")
        self.assertIn("updated", tool_result_text(result))

    def test_replace_all_changes_every_occurrence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "f.txt"
            target.write_text("x = 1\ny = x\nz = x\n")
            result = _execute(
                make_edit_tool(cwd=tmp),
                {
                    "path": "f.txt",
                    "old_string": "x",
                    "new_string": "q",
                    "replace_all": True,
                },
            )
            written = target.read_text()

        self.assertFalse(result.is_error)
        self.assertEqual(written, "q = 1\ny = q\nz = q\n")
        self.assertEqual(result.details["replacements"], 3)

    def test_creates_a_new_file_with_empty_old_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = _execute(
                make_edit_tool(cwd=tmp),
                {"path": "new.txt", "old_string": "", "new_string": "hello\n"},
            )
            written = (Path(tmp) / "new.txt").read_text()

        self.assertFalse(result.is_error)
        self.assertEqual(written, "hello\n")
        self.assertTrue(result.details["created"])
        self.assertIn("created", tool_result_text(result))

    def test_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = _execute(
                make_edit_tool(cwd=tmp),
                {"path": "a/b/c.txt", "old_string": "", "new_string": "deep\n"},
            )
            written = (Path(tmp) / "a" / "b" / "c.txt").read_text()

        self.assertFalse(result.is_error)
        self.assertEqual(written, "deep\n")

    def test_identical_strings_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "f.txt").write_text("same\n")
            result = _execute(
                make_edit_tool(cwd=tmp),
                {"path": "f.txt", "old_string": "same", "new_string": "same"},
            )

        self.assertTrue(result.is_error)
        self.assertIn("No changes to make", tool_result_text(result))

    def test_string_not_found_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "f.txt").write_text("alpha\n")
            result = _execute(
                make_edit_tool(cwd=tmp),
                {"path": "f.txt", "old_string": "missing", "new_string": "x"},
            )

        self.assertTrue(result.is_error)
        self.assertIn("not found", tool_result_text(result))

    def test_multiple_matches_without_replace_all_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "f.txt").write_text("a\na\na\n")
            result = _execute(
                make_edit_tool(cwd=tmp),
                {"path": "f.txt", "old_string": "a", "new_string": "b"},
            )

        self.assertTrue(result.is_error)
        self.assertIn("Found 3 matches", tool_result_text(result))

    def test_missing_file_with_nonempty_old_string_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = _execute(
                make_edit_tool(cwd=tmp),
                {"path": "nope.txt", "old_string": "x", "new_string": "y"},
            )

        self.assertTrue(result.is_error)
        self.assertIn("does not exist", tool_result_text(result))

    def test_empty_old_string_on_existing_nonempty_file_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "f.txt").write_text("has content\n")
            result = _execute(
                make_edit_tool(cwd=tmp),
                {"path": "f.txt", "old_string": "", "new_string": "x"},
            )

        self.assertTrue(result.is_error)
        self.assertIn("already exists", tool_result_text(result))

    def test_empty_old_string_fills_an_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "empty.txt"
            target.write_text("")
            result = _execute(
                make_edit_tool(cwd=tmp),
                {"path": "empty.txt", "old_string": "", "new_string": "filled\n"},
            )
            written = target.read_text()

        self.assertFalse(result.is_error)
        self.assertEqual(written, "filled\n")

    def test_missing_path_argument_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = _execute(
                make_edit_tool(cwd=tmp),
                {"old_string": "a", "new_string": "b"},
            )

        self.assertTrue(result.is_error)
        self.assertIn("path", tool_result_text(result))

    def test_directory_path_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "sub").mkdir()
            result = _execute(
                make_edit_tool(cwd=tmp),
                {"path": "sub", "old_string": "a", "new_string": "b"},
            )

        self.assertTrue(result.is_error)

    def test_edit_tool_schema_is_strict(self) -> None:
        tool = make_edit_tool(cwd=".")

        self.assertEqual(tool.name, EDIT_TOOL_NAME)
        self.assertEqual(
            tool.parameters["required"], ["path", "old_string", "new_string"]
        )
        self.assertFalse(tool.parameters["additionalProperties"])
        self.assertEqual(tool.execution_mode, "sequential")


class EditFileTest(unittest.TestCase):
    def test_returns_replacement_count_in_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "f.txt"
            target.write_text("one two two\n")
            result = edit_file("f.txt", "two", "2", replace_all=True, root=tmp)
            written = target.read_text()

        self.assertFalse(result.is_error)
        self.assertEqual(result.details["replacements"], 2)
        self.assertEqual(written, "one 2 2\n")


def _execute(tool: AgentTool, args: dict[str, object]) -> ToolResult:
    execute = cast(object, tool.execute)
    if not callable(execute):
        raise AssertionError("edit tool has no execute function")
    return execute("call_1", args, lambda: False, None)


if __name__ == "__main__":
    unittest.main()
