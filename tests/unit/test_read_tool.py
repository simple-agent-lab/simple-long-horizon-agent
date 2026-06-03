from __future__ import annotations

import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from typing import cast

from simple_agent_lab import AgentTool, ToolResult, tool_result_text
from simple_agent_lab.messages import ImageBlock
from simple_agent_lab.tools.read import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    format_size,
    make_read_tool,
    truncate_head,
)


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
        self.assertIn("File not found", tool_result_text(result))

    def test_directory_path_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = _execute(make_read_tool(cwd=tmp), {"path": "."})

        self.assertTrue(result.is_error)
        self.assertIn("directory", tool_result_text(result))

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

    def test_read_tool_schema_is_strict(self) -> None:
        tool = make_read_tool(cwd=".")

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


def _execute(tool: AgentTool, args: dict[str, object]) -> ToolResult:
    execute = cast(object, tool.execute)
    if not callable(execute):
        raise AssertionError("read tool has no execute function")
    return execute("call_1", args, lambda: False, None)


def _make_red_png(side: int = 32) -> bytes:
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * side for _ in range(side))

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data))
        )

    ihdr = struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0)
    idat = zlib.compress(raw)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )


if __name__ == "__main__":
    unittest.main()
