from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from typing import cast

from simple_agent_lab import (
    DEFAULT_BASH_DEMO_COMMAND,
    AgentTool,
    ToolResult,
    bash_execution_to_tool_result,
    detect_blocked_sleep_pattern,
    interpret_command_result,
    last_message,
    make_bash_tool,
    message_text,
    run_bash,
    run_bash_agent_demo,
    tool_result_text,
)
from simple_agent_lab.bash_tool import (
    MAX_BASH_TIMEOUT_SECONDS,
    _resolve_timeout,
)


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
        tool_result = last_message(runtime.state, kind="tool_result")
        final = last_message(runtime.state, kind="final")

        self.assertEqual(tool_result.tool_name, "bash")
        self.assertIn("demo ok", message_text(tool_result))
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


if __name__ == "__main__":
    unittest.main()
