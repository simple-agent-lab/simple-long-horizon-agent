from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import cast

from simple_agent_lab import (
    Agent,
    AgentTool,
    ToolResult,
    assistant_message,
    event_record,
    message_text,
    run_trace_from_state,
    task_tool,
    tool_result_text,
    tool_results_of,
)
from simple_agent_lab.agents.bash import make_bash_agent
from simple_agent_lab.llm import Provider
from simple_agent_lab.messages import Message
from simple_agent_lab.trajectory import (
    Span,
    spans_from_events,
    trace_record,
)
from simple_agent_lab.trajectory.spans import _collect_sub_events, _tree_sort
from simple_agent_lab.tools.bash import (
    MAX_BASH_TIMEOUT_SECONDS,
    NON_INTERACTIVE_BASH_ENV,
    _resolve_timeout,
    bash_execution_to_tool_result,
    detect_blocked_sleep_pattern,
    interpret_command_result,
    make_bash_tool,
    run_bash,
)


ROOT = Path(__file__).resolve().parents[1]
FAKE_PROVIDER = Provider(id="fake", api="fake", model="fake-model")


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
        self.assertIn(
            "Done. Command completed with no output.", tool_result_text(result)
        )

    def test_bash_tool_schema_is_strict_for_model_arguments(self) -> None:
        tool = make_bash_tool(cwd=ROOT)

        self.assertEqual(tool.parameters["required"], ["command", "description"])
        self.assertFalse(tool.parameters["additionalProperties"])

    def test_bash_tool_defaults_to_parallel_execution(self) -> None:
        tool = make_bash_tool(cwd=ROOT)

        self.assertEqual(tool.execution_mode, "parallel")

    def test_bash_tool_can_be_forced_sequential(self) -> None:
        tool = make_bash_tool(cwd=ROOT, execution_mode="sequential")

        self.assertEqual(tool.execution_mode, "sequential")

    def test_bash_agent_uses_parallel_bash_tool(self) -> None:
        agent = make_bash_agent(provider=FAKE_PROVIDER, cwd=ROOT)

        self.assertEqual(agent.tools[0].execution_mode, "parallel")

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

    def test_injects_non_interactive_env_defaults(self) -> None:
        import os
        from unittest import mock

        # Clear any pre-existing values so the test exercises the injection path
        # regardless of the user's shell setup.
        scrubbed = {
            key: "" for key in ("PAGER", "MANPAGER", "TQDM_DISABLE", "PIP_PROGRESS_BAR")
        }
        with mock.patch.dict(os.environ, scrubbed, clear=False):
            for key in scrubbed:
                os.environ.pop(key, None)
            execution = run_bash(
                'printf "%s,%s,%s\\n" "$PAGER" "$TQDM_DISABLE" "$PIP_PROGRESS_BAR"',
                cwd=ROOT,
                timeout_seconds=2,
            )
        self.assertEqual(execution.exit_code, 0)
        self.assertEqual(execution.raw_stdout, "cat,1,off")

    def test_caller_env_overrides_non_interactive_defaults(self) -> None:
        import os
        from unittest import mock

        # If the caller already exported PAGER, our defaults must not clobber it.
        with mock.patch.dict(os.environ, {"PAGER": "less"}, clear=False):
            execution = run_bash('printf "%s" "$PAGER"', cwd=ROOT, timeout_seconds=2)
        self.assertEqual(execution.raw_stdout, "less")

    def test_truncation_note_suggests_narrowing_strategies(self) -> None:
        # Force truncation by emitting more than the default budget.
        big = "x" * 5000
        with tempfile.TemporaryDirectory() as tmp:
            execution = run_bash(f"printf '{big}'", cwd=tmp, timeout_seconds=2)
        self.assertTrue(execution.stdout_truncated)
        observation = bash_execution_to_tool_result(execution)
        text = tool_result_text(observation)
        self.assertIn("Re-run with a narrower view", text)
        self.assertIn("sed -n", text)

    def test_non_interactive_env_constant_lists_expected_keys(self) -> None:
        # Sanity guard so the public constant keeps documenting the contract.
        self.assertEqual(
            set(NON_INTERACTIVE_BASH_ENV),
            {"PAGER", "MANPAGER", "LESS", "PIP_PROGRESS_BAR", "TQDM_DISABLE"},
        )

    def test_blocks_long_leading_sleep_without_waiting(self) -> None:
        self.assertEqual(detect_blocked_sleep_pattern("sleep 2"), "standalone sleep 2")
        tool = make_bash_tool(cwd=ROOT)
        result = _execute(tool, {"command": "sleep 2"})

        self.assertTrue(result.is_error)
        self.assertIn("Blocked bash command", tool_result_text(result))

    def test_bash_agent_run_runs_tool_then_finalizes(self) -> None:
        agent = make_bash_agent(provider=FAKE_PROVIDER, cwd=ROOT)
        state, events = agent.run(
            "Use bash to run command: `printf 'demo ok\\n'`",
            max_turns=3,
        )
        for _ in events:
            pass
        tool_result_msg = next(
            message
            for message in reversed(state.messages)
            if message.kind == "tool_result"
        )
        final = next(
            message for message in reversed(state.messages) if message.kind == "final"
        )

        blocks = tool_results_of(tool_result_msg.content)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].tool_name, "bash")
        self.assertIn("demo ok", message_text(tool_result_msg))
        self.assertEqual(final.sender, "bash_agent")
        self.assertIn("demo ok", message_text(final))
        self.assertTrue(
            any(event.kind == "tool_execution_start" for event in state.events)
        )
        self.assertTrue(any(event.kind == "model_request" for event in state.events))

    def test_run_trace_from_state_produces_span_tree(self) -> None:
        agent = make_bash_agent(provider=FAKE_PROVIDER, cwd=ROOT)
        state, events = agent.run(
            "Use bash to run command: `printf 'trace ok\\n'`",
            max_turns=3,
        )
        for _ in events:
            pass

        trace = run_trace_from_state(
            state=state,
            trace_id="test.bash_agent",
            producer="tests",
        )

        spans = trace.spans()
        model_calls = [s for s in spans if s.kind == "model_call"]
        tool_calls = [s for s in spans if s.kind == "tool_call"]
        turns = [s for s in spans if s.kind == "turn"]
        agent_runs = [s for s in spans if s.kind == "agent_run"]

        self.assertGreaterEqual(len(model_calls), 2)
        self.assertGreaterEqual(len(tool_calls), 1)
        self.assertGreaterEqual(len(turns), 1)
        self.assertEqual(len(agent_runs), 1)

        first_call = model_calls[0]
        self.assertEqual(first_call.attributes["agent"], "bash_agent")
        self.assertEqual(first_call.attributes["tools"][0]["name"], "bash")
        self.assertEqual(first_call.attributes["visible_count"], 1)
        self.assertGreaterEqual(first_call.start, 0.0)
        self.assertGreater(first_call.end, first_call.start)

        first_tool = tool_calls[0]
        self.assertEqual(first_tool.attributes["tool_name"], "bash")
        self.assertIsNotNone(first_tool.parent_id)

        event_kinds = [e.kind.value for e in trace.events]
        self.assertIn("model_request", event_kinds)
        self.assertIn("tool_execution_start", event_kinds)
        record = trace_record(trace)
        self.assertEqual(record["events"][0]["kind"], "message")
        self.assertGreater(len(record["spans"]), 0)
        self.assertEqual(event_record(state.events[0])["kind"], "message")


class TraceSpanTest(unittest.TestCase):
    """Tests for span extraction edge cases and serialization."""

    def test_parent_id_returns_none_when_skip_exhausts_stack(self) -> None:
        """_parent_id must return None, not the skipped entry, when all
        stack entries match skip_kinds."""
        from simple_agent_lab.protocols import (
            AgentStartEvent,
            ToolExecutionStartEvent,
            ToolExecutionEndEvent,
            TurnStartEvent,
            TurnEndEvent,
            AgentEndEvent,
        )

        events = [
            AgentStartEvent(index=0, elapsed=0.0),
            TurnStartEvent(index=1, elapsed=0.1, agent="a"),
            ToolExecutionStartEvent(
                index=2, elapsed=0.2, tool_call_id="c1", tool_name="t1"
            ),
            ToolExecutionStartEvent(
                index=3, elapsed=0.3, tool_call_id="c2", tool_name="t2"
            ),
            ToolExecutionEndEvent(
                index=4,
                elapsed=0.4,
                tool_call_id="c1",
                tool_name="t1",
                is_error=False,
                terminate=False,
            ),
            ToolExecutionEndEvent(
                index=5,
                elapsed=0.5,
                tool_call_id="c2",
                tool_name="t2",
                is_error=False,
                terminate=False,
            ),
            TurnEndEvent(index=6, elapsed=0.6, agent="a"),
            AgentEndEvent(index=7, elapsed=0.7, reason="done"),
        ]
        spans = spans_from_events("test", events)
        tool_spans = [s for s in spans if s.kind == "tool_call"]
        self.assertEqual(len(tool_spans), 2)
        for ts in tool_spans:
            turn_spans = [s for s in spans if s.kind == "turn"]
            self.assertEqual(len(turn_spans), 1)
            self.assertEqual(
                ts.parent_id,
                turn_spans[0].id,
                "Parallel tool_call must be child of turn, not sibling tool_call",
            )

    def test_model_turns_extracted_from_trace(self) -> None:
        agent = make_bash_agent(provider=FAKE_PROVIDER, cwd=ROOT)
        state, events = agent.run(
            "Use bash to run command: `printf 'mt ok\\n'`",
            max_turns=3,
        )
        for _ in events:
            pass

        trace = run_trace_from_state(
            state=state,
            trace_id="test.mt",
            producer="tests",
        )
        turns = trace.model_turns()
        self.assertGreaterEqual(len(turns), 1)

        first = turns[0]
        self.assertEqual(first.agent, "bash_agent")
        self.assertIn("model", first.step_id)
        self.assertIsInstance(first.input_messages, list)
        self.assertIsInstance(first.output_message, dict)
        self.assertIsInstance(first.tools, list)
        self.assertGreater(len(first.tools), 0)

    def test_trace_record_includes_model_turns_and_spans(self) -> None:
        agent = make_bash_agent(provider=FAKE_PROVIDER, cwd=ROOT)
        state, events = agent.run(
            "Use bash to run command: `printf 'rec ok\\n'`",
            max_turns=3,
        )
        for _ in events:
            pass

        trace = run_trace_from_state(
            state=state,
            trace_id="test.rec",
            producer="tests",
        )
        record = trace_record(trace)

        self.assertIn("spans", record)
        self.assertIn("model_turns", record)
        self.assertGreater(len(record["spans"]), 0)
        self.assertGreater(len(record["model_turns"]), 0)

        first_mt = record["model_turns"][0]
        self.assertIn("step_id", first_mt)
        self.assertIn("input_messages", first_mt)
        self.assertIn("output_message", first_mt)

    def test_tree_sort_handles_orphans(self) -> None:
        orphan = Span(
            id="orphan", parent_id="nonexistent", kind="x", start=0.0, end=1.0
        )
        root = Span(id="root", parent_id=None, kind="agent_run", start=0.0, end=2.0)
        child = Span(id="child", parent_id="root", kind="turn", start=0.1, end=1.9)
        result = _tree_sort([orphan, root, child])
        ids = [s.id for s in result]
        self.assertEqual(ids[0], "root")
        self.assertEqual(ids[1], "child")
        self.assertIn("orphan", ids)


class MergedSpansTest(unittest.TestCase):
    """Tests for _collect_sub_events and merged_spans end-to-end."""

    @staticmethod
    def _make_echo_generate(name: str):
        def generate(visible: list[Message]) -> Message:
            task_msg = next(m for m in visible if m.kind == "task")
            return assistant_message(
                f"{name}:{message_text(task_msg)}",
                sender=name,
                target="user",
                kind="final",
            )

        return generate

    def test_collect_sub_events_extracts_from_tool_result_message(self) -> None:
        """_collect_sub_events must find sub_events keyed by call_id
        inside message.data['details'][call_id]['sub_events']."""
        from simple_agent_lab.protocols import AgentStartEvent, AgentEndEvent

        fake_sub_events = [
            AgentStartEvent(index=0, elapsed=0.0),
            AgentEndEvent(index=1, elapsed=1.0, reason="done"),
        ]
        from simple_agent_lab.messages import user_message

        msg = user_message(
            "result text",
            kind="tool_result",
            data={"details": {"call_42": {"sub_events": fake_sub_events}}},
        )
        collected = _collect_sub_events([msg])
        self.assertIn("call_42", collected)
        self.assertEqual(len(collected["call_42"]), 2)

    def test_collect_sub_events_skips_non_tool_result_messages(self) -> None:
        from simple_agent_lab.messages import user_message

        msg = user_message("hello", kind="message")
        self.assertEqual(_collect_sub_events([msg]), {})

    def test_collect_sub_events_skips_empty_details(self) -> None:
        from simple_agent_lab.messages import user_message

        msg = user_message("r", kind="tool_result", data={"details": {}})
        self.assertEqual(_collect_sub_events([msg]), {})

    def test_collect_sub_events_skips_missing_sub_events_key(self) -> None:
        from simple_agent_lab.messages import user_message

        msg = user_message(
            "r",
            kind="tool_result",
            data={"details": {"call_1": {"other": "stuff"}}},
        )
        self.assertEqual(_collect_sub_events([msg]), {})

    def test_task_tool_merged_spans_inlines_sub_agent(self) -> None:
        """Full end-to-end: task_tool run → RunTrace → merged_spans()
        must produce sub-agent spans nested under the tool_call span."""
        sub = Agent("sub", self._make_echo_generate("sub"), role="echo")
        parent_generate = _make_delegating_generate("parent", "sub")
        parent = Agent("parent", parent_generate, role="orchestrator")
        parent.tools = [task_tool([sub])]

        state, events = parent.run("do it", max_turns=3)
        for _ in events:
            pass

        trace = run_trace_from_state(
            state=state, trace_id="test.merge", producer="tests"
        )

        parent_only = trace.spans()
        merged = trace.merged_spans()

        self.assertGreater(len(merged), len(parent_only))

        sub_agent_runs = [s for s in merged if s.kind == "agent_run"]
        self.assertGreaterEqual(
            len(sub_agent_runs), 2, "Should have parent + sub agent_run spans"
        )

        task_tool_spans = [
            s
            for s in merged
            if s.kind == "tool_call"
            and s.attributes
            and s.attributes.get("tool_name") == "task"
        ]
        self.assertGreaterEqual(len(task_tool_spans), 1)

        task_span = task_tool_spans[0]
        sub_children = [s for s in merged if s.parent_id == task_span.id]
        self.assertGreater(
            len(sub_children), 0, "Sub-agent spans must be children of task tool_call"
        )

    def test_merged_spans_returns_parent_only_when_no_sub_events(self) -> None:
        agent = make_bash_agent(provider=FAKE_PROVIDER, cwd=ROOT)
        state, events = agent.run("printf 'hi\\n'", max_turns=2)
        for _ in events:
            pass

        trace = run_trace_from_state(
            state=state, trace_id="test.nosub", producer="tests"
        )
        self.assertEqual(trace.spans(), trace.merged_spans())

    def test_task_tool_sub_events_survive_json_round_trip(self) -> None:
        """`task_tool` stashes sub-agent events under a tool_result's
        `data.details[call_id].sub_events`. Those events MUST keep their
        `kind` discriminator after a JSON round-trip, because the trace
        viewer (JS) and any other on-disk consumer dispatch on
        `ev.kind`. Guards against `kind` regressing to a `@property`
        that `dataclasses.asdict` would silently drop."""
        sub = Agent("sub", self._make_echo_generate("sub"), role="echo")
        parent_generate = _make_delegating_generate("parent", "sub")
        parent = Agent("parent", parent_generate, role="orchestrator")
        parent.tools = [task_tool([sub])]

        state, events = parent.run("do it", max_turns=3)
        for _ in events:
            pass

        trace = run_trace_from_state(
            state=state, trace_id="test.round_trip", producer="tests"
        )
        parsed = json.loads(json.dumps(trace_record(trace)))

        sub_event_lists: list[list[dict]] = []
        for msg in parsed["messages"]:
            details = (msg.get("data") or {}).get("details") or {}
            for call_details in details.values():
                sub_events = call_details.get("sub_events")
                if isinstance(sub_events, list) and sub_events:
                    sub_event_lists.append(sub_events)

        self.assertTrue(sub_event_lists, "task_tool run produced no on-disk sub_events")
        kinds: set[str] = set()
        for sub_events in sub_event_lists:
            for ev in sub_events:
                self.assertIn(
                    "kind",
                    ev,
                    f"on-disk sub-event lost its discriminator: {ev!r}",
                )
                self.assertIsInstance(ev["kind"], str)
                kinds.add(ev["kind"])
        # A sub-agent that ran to a `final` should at least cover the
        # agent + turn boundaries: this is what the viewer needs to
        # reconstruct the nested span tree.
        self.assertIn("agent_start", kinds)
        self.assertIn("turn_start", kinds)
        self.assertIn("turn_end", kinds)
        self.assertIn("agent_end", kinds)


def _make_delegating_generate(name: str, sub_name: str):
    """Generate function that delegates to a sub-agent via task tool."""
    call_count = 0

    def generate(visible: list[Message]) -> Message:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            task_msg = next(m for m in visible if m.kind == "task")
            from simple_agent_lab.messages import ToolCallBlock

            return assistant_message(
                (
                    ToolCallBlock(
                        id="task_1",
                        name="task",
                        arguments={
                            "subagent_type": sub_name,
                            "task": message_text(task_msg),
                        },
                    ),
                ),
                sender=name,
                target="user",
                kind="thought",
            )
        return assistant_message("done", sender=name, target="user", kind="final")

    return generate


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
            _resolve_timeout(
                MAX_BASH_TIMEOUT_SECONDS * 10, 5.0, MAX_BASH_TIMEOUT_SECONDS
            ),
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
