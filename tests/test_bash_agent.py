from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from typing import cast

from simple_agent_lab import (
    AgentTool,
    ToolResult,
    last_message,
    message_text,
    tool_result_text,
    tool_results_of,
)
from simple_agent_lab.agents.bash import (
    DEFAULT_BASH_DEMO_COMMAND,
    MAX_BASH_TIMEOUT_SECONDS,
    bash_execution_to_tool_result,
    detect_blocked_sleep_pattern,
    interpret_command_result,
    make_bash_tool,
    run_bash,
    run_bash_agent_demo,
)
from simple_agent_lab.agents.bash.tool import _resolve_timeout


ROOT = Path(__file__).resolve().parents[1]


class BashToolTest(unittest.TestCase):
    def test_runs_command_and_returns_structured_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = make_bash_tool(cwd=tmp)
            result = _execute(tool, {"command": "printf 'hello\\n'"})

        self.assertFalse(result.is_error)
        self.assertIn("hello", tool_result_text(result))
        self.assertEqual(result.details["exit_code"], 0)
        self.assertEqual(result.details["raw_stdout"], "hello")

    def test_successful_empty_output_reports_done(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = _execute(make_bash_tool(cwd=tmp), {"command": "true"})

        self.assertFalse(result.is_error)
        self.assertIn("Done. Command completed with no output.", tool_result_text(result))

    def test_bash_tool_schema_is_strict_for_model_arguments(self) -> None:
        tool = make_bash_tool(cwd=ROOT)

        self.assertEqual(tool.parameters["required"], ["command", "description"])
        self.assertFalse(tool.parameters["additionalProperties"])

    def test_grep_no_match_is_observation_not_tool_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            execution = run_bash(
                "grep missing /dev/null",
                cwd=tmp,
                timeout_seconds=2,
            )
        result = bash_execution_to_tool_result(execution)

        self.assertFalse(result.is_error)
        self.assertEqual(
            interpret_command_result("grep missing /dev/null", 1).message,
            "No matches found",
        )
        self.assertIn("No matches found", tool_result_text(result))
        self.assertIn("exit_code: 1", tool_result_text(result))

    def test_blocks_long_leading_sleep_without_waiting(self) -> None:
        self.assertEqual(detect_blocked_sleep_pattern("sleep 2"), "standalone sleep 2")
        tool = make_bash_tool(cwd=ROOT)
        result = _execute(tool, {"command": "sleep 2"})

        self.assertTrue(result.is_error)
        self.assertIn("Blocked bash command", tool_result_text(result))

    def test_bash_agent_demo_runs_tool_then_finalizes(self) -> None:
        runtime = run_bash_agent_demo(
            command="printf 'demo ok\\n'",
            cwd=ROOT,
        )
        tool_result_msg = last_message(runtime.state, kind="tool_result")
        final = last_message(runtime.state, kind="final")

        blocks = tool_results_of(tool_result_msg.content)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].tool_name, "bash")
        self.assertIn("demo ok", message_text(tool_result_msg))
        self.assertEqual(final.sender, "bash_agent")
        self.assertIn("demo ok", message_text(final))
        self.assertTrue(
            any(event.kind == "tool_execution_start" for event in runtime.state.events)
        )
        self.assertTrue(
            any(event.kind == "model_request" for event in runtime.state.events)
        )

    def test_default_demo_command_is_read_only(self) -> None:
        self.assertIn("find src/simple_agent_lab", DEFAULT_BASH_DEMO_COMMAND)


class BashToolCrashSafetyTest(unittest.TestCase):
    """Defenses against non-UTF-8 output and NaN/zero/non-numeric timeouts."""

    def test_invalid_utf8_stdout_does_not_crash(self) -> None:
        execution = run_bash(r"printf '\xff\xfe\xfd'", cwd=ROOT, timeout_seconds=2)
        self.assertEqual(execution.exit_code, 0)
        # All three bytes are invalid UTF-8 starts → three replacement chars.
        self.assertEqual(execution.raw_stdout, "���")
        self.assertFalse(execution.is_error)

    def test_invalid_utf8_stderr_does_not_crash(self) -> None:
        execution = run_bash(r"printf '\xff' >&2", cwd=ROOT, timeout_seconds=2)
        self.assertEqual(execution.exit_code, 0)
        self.assertEqual(execution.raw_stderr, "�")
        # No exit-code failure, just unusual bytes.
        self.assertFalse(execution.is_error)

    def test_mixed_valid_and_invalid_utf8_preserves_valid_parts(self) -> None:
        execution = run_bash(r"printf 'ok\xffok'", cwd=ROOT, timeout_seconds=2)
        self.assertEqual(execution.raw_stdout, "ok�ok")

    def test_truncated_multibyte_sequence_is_replaced_not_raised(self) -> None:
        # 0xC3 starts a 2-byte UTF-8 sequence; 0x28 ('(') is not a valid
        # continuation byte. Must not raise.
        execution = run_bash(r"printf '\xc3\x28'", cwd=ROOT, timeout_seconds=2)
        self.assertEqual(execution.exit_code, 0)
        self.assertIn("�", execution.raw_stdout)

    def test_valid_utf8_emoji_passes_through(self) -> None:
        execution = run_bash("printf '🦀\\n'", cwd=ROOT, timeout_seconds=2)
        self.assertEqual(execution.raw_stdout, "🦀")
        self.assertFalse(execution.is_error)

    def test_binary_output_via_tool_returns_structured_result(self) -> None:
        tool = make_bash_tool(cwd=ROOT)
        result = _execute(tool, {"command": r"printf '\xff\xfe'"})
        self.assertFalse(result.is_error)
        self.assertEqual(result.details["exit_code"], 0)
        # Truncation note must NOT fire for tiny binary output.
        self.assertFalse(result.details["stdout_truncated"])
        self.assertEqual(result.details["raw_stdout"], "��")

    def test_resolve_timeout_rejects_nan(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _resolve_timeout(float("nan"), 5.0, 60.0)
        self.assertIn("NaN", str(ctx.exception))

    def test_resolve_timeout_rejects_string_nan(self) -> None:
        # `float("nan")` succeeds, so the string must be caught by the same path.
        with self.assertRaises(ValueError) as ctx:
            _resolve_timeout("nan", 5.0, 60.0)
        self.assertIn("NaN", str(ctx.exception))

    def test_resolve_timeout_rejects_zero(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_timeout(0, 5.0, 60.0)

    def test_resolve_timeout_rejects_negative(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_timeout(-1, 5.0, 60.0)

    def test_resolve_timeout_rejects_non_numeric(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_timeout("not-a-number", 5.0, 60.0)

    def test_resolve_timeout_clamps_infinity_to_max(self) -> None:
        self.assertEqual(_resolve_timeout(float("inf"), 5.0, 60.0), 60.0)

    def test_resolve_timeout_uses_default_when_missing(self) -> None:
        self.assertEqual(_resolve_timeout(None, 5.0, 60.0), 5.0)
        self.assertEqual(_resolve_timeout("", 5.0, 60.0), 5.0)

    def test_resolve_timeout_clamps_overshoot(self) -> None:
        self.assertEqual(_resolve_timeout(9999, 5.0, 60.0), 60.0)

    def test_nan_timeout_returns_structured_tool_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = make_bash_tool(cwd=tmp)
            result = _execute(
                tool,
                {"command": "echo hi", "timeout_seconds": float("nan")},
            )
        self.assertTrue(result.is_error)
        self.assertIn("Invalid bash timeout", tool_result_text(result))
        self.assertIn("NaN", tool_result_text(result))

    def test_zero_timeout_returns_structured_tool_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = make_bash_tool(cwd=tmp)
            result = _execute(tool, {"command": "echo hi", "timeout_seconds": 0})
        self.assertTrue(result.is_error)
        self.assertIn("Invalid bash timeout", tool_result_text(result))

    def test_non_numeric_timeout_returns_structured_tool_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = make_bash_tool(cwd=tmp)
            result = _execute(
                tool,
                {"command": "echo hi", "timeout_seconds": "soon"},
            )
        self.assertTrue(result.is_error)
        self.assertIn("Invalid bash timeout", tool_result_text(result))

    def test_partial_binary_output_before_timeout_does_not_crash(self) -> None:
        # Emit invalid UTF-8 then sleep past the timeout. With errors='replace'
        # the partial bytes captured by TimeoutExpired must decode cleanly.
        execution = run_bash(
            r"printf '\xff\xfe' && sleep 5",
            cwd=ROOT,
            timeout_seconds=1,
        )
        self.assertTrue(execution.timed_out)
        self.assertLess(execution.exit_code, 0)
        # Replacement chars survived, plus our timeout banner is appended.
        self.assertIn("�", execution.raw_stdout)
        self.assertIn("Timed out", execution.raw_stderr)

    def test_max_timeout_constant_matches_resolved_cap(self) -> None:
        # Sanity: the public default cap is what _resolve_timeout enforces.
        self.assertGreater(MAX_BASH_TIMEOUT_SECONDS, 0)
        self.assertTrue(math.isfinite(MAX_BASH_TIMEOUT_SECONDS))
        self.assertEqual(
            _resolve_timeout(MAX_BASH_TIMEOUT_SECONDS * 10, 5.0, MAX_BASH_TIMEOUT_SECONDS),
            MAX_BASH_TIMEOUT_SECONDS,
        )


def _execute(tool: AgentTool, args: dict[str, object]) -> ToolResult:
    execute = cast(object, tool.execute)
    if not callable(execute):
        raise AssertionError("bash tool has no execute function")
    return execute("call_1", args, lambda: False, None)


def _make_red_png(side: int = 32) -> bytes:
    import struct
    import zlib

    raw = b"".join(b"\x00" + b"\xff\x00\x00" * side for _ in range(side))

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data))
        )

    ihdr = struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw))
        + _chunk(b"IEND", b"")
    )


class BashAttachTest(unittest.TestCase):
    """`attach` inlines file paths as image content blocks on the ToolResult."""

    def test_attach_inlines_png_as_image_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            png = Path(tmp) / "red.png"
            png.write_bytes(_make_red_png())
            tool = make_bash_tool(cwd=tmp)
            result = _execute(
                tool,
                {"command": "ls", "description": "list dir", "attach": ["red.png"]},
            )
        image_blocks = [b for b in result.content if b.kind == "image"]
        self.assertEqual(len(image_blocks), 1)
        self.assertEqual(image_blocks[0].mime_type, "image/png")
        self.assertTrue(image_blocks[0].data)  # non-empty base64 payload

    def test_attach_resolves_paths_against_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            png_dir = Path(tmp) / "out"
            png_dir.mkdir()
            (png_dir / "shot.png").write_bytes(_make_red_png())
            tool = make_bash_tool(cwd=tmp)
            result = _execute(
                tool,
                {
                    "command": "true",
                    "description": "noop",
                    "attach": ["out/shot.png"],
                },
            )
        image_blocks = [b for b in result.content if b.kind == "image"]
        self.assertEqual(len(image_blocks), 1)

    def test_attach_records_note_for_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = make_bash_tool(cwd=tmp)
            result = _execute(
                tool,
                {
                    "command": "true",
                    "description": "noop",
                    "attach": ["missing.png", "also-missing.jpg"],
                },
            )
        image_blocks = [b for b in result.content if b.kind == "image"]
        self.assertEqual(len(image_blocks), 0)
        note_text = "\n".join(b.text for b in result.content if b.kind == "text")
        self.assertIn("missing.png", note_text)
        self.assertIn("not a file", note_text)

    def test_attach_rejects_oversize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            big = Path(tmp) / "big.png"
            big.write_bytes(_make_red_png())
            tool = make_bash_tool(cwd=tmp, max_attach_bytes=10)  # absurdly small cap
            result = _execute(
                tool,
                {
                    "command": "true",
                    "description": "noop",
                    "attach": ["big.png"],
                },
            )
        image_blocks = [b for b in result.content if b.kind == "image"]
        self.assertEqual(len(image_blocks), 0)
        note_text = "\n".join(b.text for b in result.content if b.kind == "text")
        self.assertIn("exceeds limit", note_text)


if __name__ == "__main__":
    unittest.main()
