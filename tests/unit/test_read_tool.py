from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from simple_long_horizon_agent import tool_result_text
from simple_long_horizon_agent.messages import ImageBlock
from simple_long_horizon_agent.tools.read import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    READ_TOOL_NAME,
    format_size,
    make_read_tool,
    truncate_head,
)
from tests.unit._support import execute_tool as _execute
from tests.unit._support import make_red_png as _make_red_png


class ReadToolTest(unittest.TestCase):
    def test_reads_full_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "f.txt").write_text("alpha\nbeta\ngamma\n")
            result = _execute(make_read_tool(cwd=tmp), {"path": "f.txt"})

        self.assertFalse(result.is_error)
        text = tool_result_text(result)
        self.assertIn("alpha", text)
        self.assertIn("gamma", text)

    def test_offset_and_limit_select_a_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "f.txt").write_text(
                "\n".join(f"line{i}" for i in range(1, 11)) + "\n"
            )
            result = _execute(
                make_read_tool(cwd=tmp), {"path": "f.txt", "offset": 3, "limit": 2}
            )

        text = tool_result_text(result)
        self.assertIn("line3", text)
        self.assertIn("line4", text)
        self.assertNotIn("line5\n", text)
        self.assertIn("Use offset=5 to continue.", text)

    def test_offset_beyond_end_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "f.txt").write_text("only one line\n")
            result = _execute(make_read_tool(cwd=tmp), {"path": "f.txt", "offset": 99})

        self.assertTrue(result.is_error)
        self.assertIn("beyond end of file", tool_result_text(result))

    def test_line_limit_truncation_points_to_next_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "big.txt").write_text("\n".join(f"L{i}" for i in range(5000)))
            tool = make_read_tool(cwd=tmp, max_lines=10)
            result = _execute(tool, {"path": "big.txt"})

        text = tool_result_text(result)
        self.assertIn("Showing lines 1-10 of 5000", text)
        self.assertIn("Use offset=11 to continue.", text)
        self.assertEqual(result.details["truncation"]["truncated_by"], "lines")

    def test_byte_limit_truncation_reports_the_byte_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "big.txt").write_text("\n".join(f"L{i}" for i in range(5000)))
            tool = make_read_tool(cwd=tmp, max_bytes=20)
            result = _execute(tool, {"path": "big.txt"})

        text = tool_result_text(result)
        self.assertIn("20B limit", text)
        self.assertEqual(result.details["truncation"]["truncated_by"], "bytes")

    def test_first_line_over_byte_cap_suggests_shell_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "huge.txt").write_text("x" * 100 + "\nshort\n")
            tool = make_read_tool(cwd=tmp, max_bytes=20)
            result = _execute(tool, {"path": "huge.txt"})

        text = tool_result_text(result)
        self.assertIn("Line 1 is 100B", text)
        self.assertIn("sed -n '1p'", text)
        self.assertTrue(result.details["truncation"]["first_line_exceeds_limit"])

    def test_missing_file_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = _execute(make_read_tool(cwd=tmp), {"path": "nope.txt"})

        self.assertTrue(result.is_error)
        self.assertIn("No such file or directory", tool_result_text(result))

    def test_directory_path_returns_a_listing(self) -> None:
        # Skills are read by browsing: reading a directory lists its files so
        # the model can see a skill's scripts/ and references/ before loading.
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

    def test_missing_path_argument_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = _execute(make_read_tool(cwd=tmp), {})

        self.assertTrue(result.is_error)
        self.assertIn("Missing required read argument", tool_result_text(result))

    def test_invalid_offset_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "f.txt").write_text("a\nb\n")
            result = _execute(make_read_tool(cwd=tmp), {"path": "f.txt", "offset": 0})

        self.assertTrue(result.is_error)
        self.assertIn("offset", tool_result_text(result))

    def test_trailing_newline_is_not_counted_as_an_extra_line(self) -> None:
        # A file ending in "\n" must report its real line count and must not
        # expose a phantom empty line one past the end via offset.
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "f.txt").write_text("only one line\n")
            ok = _execute(make_read_tool(cwd=tmp), {"path": "f.txt", "offset": 1})
            past = _execute(make_read_tool(cwd=tmp), {"path": "f.txt", "offset": 2})

        self.assertFalse(ok.is_error)
        self.assertIn("only one line", tool_result_text(ok))
        self.assertTrue(past.is_error)
        self.assertIn("1 lines total", tool_result_text(past))

    def test_fractional_offset_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "f.txt").write_text("a\nb\nc\n")
            result = _execute(make_read_tool(cwd=tmp), {"path": "f.txt", "offset": 1.9})

        self.assertTrue(result.is_error)
        self.assertIn("integer", tool_result_text(result))

    def test_image_file_returns_an_image_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "pic.png").write_bytes(_make_red_png())
            result = _execute(make_read_tool(cwd=tmp), {"path": "pic.png"})

        self.assertFalse(result.is_error)
        self.assertIn("Read image file [image/png]", tool_result_text(result))
        self.assertTrue(any(isinstance(b, ImageBlock) for b in result.content))

    def test_oversized_image_is_omitted_with_a_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "pic.png").write_bytes(_make_red_png())
            tool = make_read_tool(cwd=tmp, max_attach_bytes=10)
            result = _execute(tool, {"path": "pic.png"})

        self.assertFalse(result.is_error)
        self.assertIn("Image omitted", tool_result_text(result))
        self.assertFalse(any(isinstance(b, ImageBlock) for b in result.content))

    def test_budget_is_more_generous_than_bash(self) -> None:
        from simple_long_horizon_agent.tools.bash import DEFAULT_BASH_MAX_OUTPUT_CHARS

        # The read tool exists partly so a real SKILL.md is not mangled by the
        # bash tool's aggressive truncation, so its byte budget must be larger.
        self.assertGreater(DEFAULT_MAX_BYTES, DEFAULT_BASH_MAX_OUTPUT_CHARS)

    def test_read_tool_schema_is_strict(self) -> None:
        tool = make_read_tool(cwd=".")

        self.assertEqual(tool.name, READ_TOOL_NAME)
        self.assertEqual(tool.parameters["required"], ["path"])
        self.assertFalse(tool.parameters["additionalProperties"])
        self.assertEqual(tool.execution_mode, "parallel")


class TruncateHeadTest(unittest.TestCase):
    def test_short_content_is_not_truncated(self) -> None:
        result = truncate_head("a\nb\nc")

        self.assertFalse(result.truncated)
        self.assertIsNone(result.truncated_by)
        self.assertEqual(result.total_lines, 3)
        self.assertEqual(result.output_lines, 3)

    def test_trailing_newline_does_not_add_a_line(self) -> None:
        result = truncate_head("a\nb\n")

        self.assertEqual(result.total_lines, 2)

    def test_line_limit_keeps_whole_lines(self) -> None:
        result = truncate_head("\n".join(str(i) for i in range(100)), max_lines=5)

        self.assertTrue(result.truncated)
        self.assertEqual(result.truncated_by, "lines")
        self.assertEqual(result.output_lines, 5)

    def test_defaults_are_exposed(self) -> None:
        result = truncate_head("hello")

        self.assertEqual(result.max_lines, DEFAULT_MAX_LINES)
        self.assertEqual(result.max_bytes, DEFAULT_MAX_BYTES)


class FormatSizeTest(unittest.TestCase):
    def test_renders_human_readable_sizes(self) -> None:
        self.assertEqual(format_size(512), "512B")
        self.assertEqual(format_size(2048), "2.0KB")
        self.assertEqual(format_size(3 * 1024 * 1024), "3.0MB")
