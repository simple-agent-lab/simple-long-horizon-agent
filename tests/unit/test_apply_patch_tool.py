from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import cast

from simple_agent_lab import AgentTool, ToolResult, tool_result_text
from simple_agent_lab.tools.apply_patch import (
    APPLY_PATCH_TOOL_NAME,
    apply_patch,
    make_apply_patch_tool,
)


def _patch(body: str) -> str:
    """Wrap a dedented hunk body in the Begin/End Patch envelope."""
    return "*** Begin Patch\n" + textwrap.dedent(body).strip("\n") + "\n*** End Patch"


class ApplyPatchTest(unittest.TestCase):
    def test_update_replaces_a_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "f.txt"
            target.write_text("alpha\nbeta\ngamma\n")
            patch = _patch(
                """
                *** Update File: f.txt
                 alpha
                -beta
                +BETA
                 gamma
                """
            )
            result = apply_patch(patch, root=tmp)
            written = target.read_text()

        self.assertFalse(result.is_error, tool_result_text(result))
        self.assertEqual(written, "alpha\nBETA\ngamma\n")
        self.assertEqual(result.details["modified"], ["f.txt"])
        self.assertIn("M f.txt", tool_result_text(result))

    def test_add_file_creates_with_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            patch = _patch(
                """
                *** Add File: pkg/new.py
                +print("hi")
                +print("bye")
                """
            )
            result = apply_patch(patch, root=tmp)
            written = (Path(tmp) / "pkg" / "new.py").read_text()

        self.assertFalse(result.is_error, tool_result_text(result))
        self.assertEqual(written, 'print("hi")\nprint("bye")')
        self.assertEqual(result.details["added"], ["pkg/new.py"])
        self.assertIn("A pkg/new.py", tool_result_text(result))

    def test_delete_file_removes_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "gone.txt"
            target.write_text("bye\n")
            patch = _patch("*** Delete File: gone.txt")
            result = apply_patch(patch, root=tmp)

        self.assertFalse(result.is_error, tool_result_text(result))
        self.assertFalse(target.exists())
        self.assertEqual(result.details["deleted"], ["gone.txt"])

    def test_move_renames_and_patches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "old.txt"
            src.write_text("one\ntwo\n")
            patch = _patch(
                """
                *** Update File: old.txt
                *** Move to: new.txt
                 one
                -two
                +TWO
                """
            )
            result = apply_patch(patch, root=tmp)
            src_exists = src.exists()
            new_text = (Path(tmp) / "new.txt").read_text()

        self.assertFalse(result.is_error, tool_result_text(result))
        self.assertFalse(src_exists)
        self.assertEqual(new_text, "one\nTWO\n")
        self.assertIn("old.txt -> new.txt", result.details["modified"])

    def test_multiple_files_in_one_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.txt").write_text("aaa\n")
            (Path(tmp) / "b.txt").write_text("bbb\n")
            patch = _patch(
                """
                *** Update File: a.txt
                -aaa
                +AAA
                *** Add File: c.txt
                +ccc
                *** Delete File: b.txt
                """
            )
            result = apply_patch(patch, root=tmp)
            a_text = (Path(tmp) / "a.txt").read_text()
            c_text = (Path(tmp) / "c.txt").read_text()
            b_exists = (Path(tmp) / "b.txt").exists()

        self.assertFalse(result.is_error, tool_result_text(result))
        self.assertEqual(a_text, "AAA\n")
        self.assertEqual(c_text, "ccc")
        self.assertFalse(b_exists)

    def test_add_then_insert_at_end_of_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "f.txt"
            target.write_text("line1\nline2\n")
            patch = _patch(
                """
                *** Update File: f.txt
                 line2
                +line3
                *** End of File
                """
            )
            result = apply_patch(patch, root=tmp)
            written = target.read_text()

        self.assertFalse(result.is_error, tool_result_text(result))
        self.assertEqual(written, "line1\nline2\nline3\n")

    def test_fuzzy_context_matches_after_relaxing_whitespace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "f.py"
            target.write_text("def f():\n    return 1\n")
            # Model reproduces the removed line with the wrong indentation
            # (2 spaces vs the file's 4); fuzzy matching still locates it.
            patch = _patch(
                """
                *** Update File: f.py
                 def f():
                -  return 1
                +    return 2
                """
            )
            result = apply_patch(patch, root=tmp)
            written = target.read_text()

        self.assertFalse(result.is_error, tool_result_text(result))
        self.assertEqual(written, "def f():\n    return 2\n")
        self.assertGreater(result.details["fuzz"], 0)

    def test_missing_envelope_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = apply_patch("*** Update File: f.txt\n-a\n+b\n", root=tmp)
        self.assertTrue(result.is_error)
        self.assertIn("Begin Patch", tool_result_text(result))

    def test_update_missing_file_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            patch = _patch(
                """
                *** Update File: nope.txt
                -a
                +b
                """
            )
            result = apply_patch(patch, root=tmp)
        self.assertTrue(result.is_error)
        self.assertIn("missing file", tool_result_text(result))

    def test_add_existing_file_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "exists.txt").write_text("x\n")
            patch = _patch(
                """
                *** Add File: exists.txt
                +y
                """
            )
            result = apply_patch(patch, root=tmp)
        self.assertTrue(result.is_error)
        self.assertIn("already exists", tool_result_text(result))

    def test_unmatched_context_is_an_error_and_leaves_file_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "f.txt"
            target.write_text("alpha\nbeta\n")
            patch = _patch(
                """
                *** Update File: f.txt
                -does-not-exist
                +x
                """
            )
            result = apply_patch(patch, root=tmp)
            written = target.read_text()

        self.assertTrue(result.is_error)
        self.assertIn("context", tool_result_text(result))
        self.assertEqual(written, "alpha\nbeta\n")  # untouched


class ApplyPatchToolTest(unittest.TestCase):
    def test_schema_is_strict(self) -> None:
        tool = make_apply_patch_tool(cwd=".")
        self.assertEqual(tool.name, APPLY_PATCH_TOOL_NAME)
        self.assertEqual(tool.parameters["required"], ["input"])
        self.assertFalse(tool.parameters["additionalProperties"])
        self.assertEqual(tool.execution_mode, "sequential")

    def test_missing_input_is_an_error(self) -> None:
        result = _execute(make_apply_patch_tool(cwd="."), {})
        self.assertTrue(result.is_error)
        self.assertIn("input", tool_result_text(result))

    def test_tool_applies_a_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "f.txt").write_text("hello\n")
            patch = _patch(
                """
                *** Update File: f.txt
                -hello
                +goodbye
                """
            )
            result = _execute(make_apply_patch_tool(cwd=tmp), {"input": patch})
            written = (Path(tmp) / "f.txt").read_text()

        self.assertFalse(result.is_error, tool_result_text(result))
        self.assertEqual(written, "goodbye\n")


def _execute(tool: AgentTool, args: dict[str, object]) -> ToolResult:
    execute = cast(object, tool.execute)
    if not callable(execute):
        raise AssertionError("apply_patch tool has no execute function")
    return execute("call_1", args, lambda: False, None)


if __name__ == "__main__":
    unittest.main()
