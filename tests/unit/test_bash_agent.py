from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from simple_long_horizon_agent import (
    Agent,
    RunTrace,
    State,
    ToolResult,
    assistant_message,
    event_record,
    message_text,
    run_trace_from_state,
    task_tool,
    tool_result_text,
    tool_results_of,
    user_message,
)
from simple_long_horizon_agent.agents.starter import make_bash_agent
from simple_long_horizon_agent.llm import Provider
from simple_long_horizon_agent.messages import Message
from simple_long_horizon_agent.trace import (
    Span,
    event_stream,
    spans_from_events,
)
from simple_long_horizon_agent.trace.spans import _collect_sub_events, _tree_sort
from simple_long_horizon_agent.tools.bash import (
    MAX_BASH_TIMEOUT_SECONDS,
    NON_INTERACTIVE_BASH_ENV,
    _resolve_timeout,
    bash_execution_to_tool_result,
    detect_blocked_sleep_pattern,
    interpret_command_result,
    make_bash_tool,
    run_bash,
)
from tests.unit._support import execute_tool as _execute
from tests.unit._support import make_red_png as _make_red_png


ROOT = Path(__file__).resolve().parents[1]
FAKE_PROVIDER = Provider(id="fake", api="fake", model="fake-model")


def _run_fake_bash(label: str) -> State:
    agent = make_bash_agent(provider=FAKE_PROVIDER, cwd=ROOT)
    state, events = agent.run(
        f"Use bash to run command: `printf '{label}\\n'`",
        max_turns=3,
    )
    list(events)
    return state


def _fake_bash_trace(label: str) -> RunTrace:
    return run_trace_from_state(
        state=_run_fake_bash(label),
        trace_id=f"test.{label}",
        producer="tests",
    )


class BashToolTest(unittest.TestCase):
    def test_runs_command_and_returns_structured_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tool = make_bash_tool(cwd=tmp)
            result = _execute(tool, {"command": "printf 'hello\\n'"})

        self.assertFalse(result.is_error)
        self.assertIn("hello", tool_result_text(result))
        self.assertEqual(result.details["exit_code"], 0)
        self.assertEqual(result.details["raw_stdout"], "hello\n")

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

    def test_bash_tools_choose_execution_mode(self) -> None:
        cases = [
            ("default", make_bash_tool(cwd=ROOT), "parallel"),
            (
                "explicit sequential",
                make_bash_tool(cwd=ROOT, execution_mode="sequential"),
                "sequential",
            ),
            (
                "bash agent default",
                make_bash_agent(provider=FAKE_PROVIDER, cwd=ROOT).tools[0],
                "parallel",
            ),
        ]
        for name, tool, expected in cases:
            with self.subTest(name):
                self.assertEqual(tool.execution_mode, expected)

    def test_grep_no_match_is_observation_not_tool_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            execution = run_bash(
                "grep missing /dev/null",
                cwd=tmp,
                timeout_seconds=3,
            )
        result = bash_execution_to_tool_result(execution)

        self.assertFalse(result.is_error)
        self.assertEqual(
            interpret_command_result("grep missing /dev/null", 1).message,
            "No matches found",
        )
        self.assertIn("No matches found", tool_result_text(result))
        self.assertIn("exit_code: 1", tool_result_text(result))

    def test_non_interactive_environment(self) -> None:
        self.assertEqual(
            set(NON_INTERACTIVE_BASH_ENV),
            {"PAGER", "MANPAGER", "LESS", "PIP_PROGRESS_BAR", "TQDM_DISABLE"},
        )
        scrubbed = {
            key: "" for key in ("PAGER", "MANPAGER", "TQDM_DISABLE", "PIP_PROGRESS_BAR")
        }
        with self.subTest("inject defaults"):
            with mock.patch.dict(os.environ, scrubbed, clear=False):
                for key in scrubbed:
                    os.environ.pop(key, None)
                execution = run_bash(
                    'printf "%s,%s,%s\\n" "$PAGER" "$TQDM_DISABLE" "$PIP_PROGRESS_BAR"',
                    cwd=ROOT,
                    timeout_seconds=3,
                )
            self.assertEqual(execution.exit_code, 0)
            self.assertEqual(execution.raw_stdout, "cat,1,off\n")

        with self.subTest("preserve caller value"):
            with mock.patch.dict(os.environ, {"PAGER": "less"}, clear=False):
                execution = run_bash(
                    'printf "%s" "$PAGER"', cwd=ROOT, timeout_seconds=3
                )
            self.assertEqual(execution.raw_stdout, "less")

    def test_truncation_note_suggests_narrowing_strategies(self) -> None:
        # Force truncation by emitting more than the default budget.
        big = "x" * 5000
        with tempfile.TemporaryDirectory() as tmp:
            execution = run_bash(f"printf '{big}'", cwd=tmp, timeout_seconds=3)
        self.assertTrue(execution.stdout_truncated)
        observation = bash_execution_to_tool_result(execution)
        text = tool_result_text(observation)
        self.assertIn("Re-run with a narrower view", text)
        self.assertIn("sed -n", text)

    def test_blocks_long_leading_sleep_without_waiting(self) -> None:
        self.assertEqual(detect_blocked_sleep_pattern("sleep 2"), "standalone sleep 2")
        tool = make_bash_tool(cwd=ROOT)
        result = _execute(tool, {"command": "sleep 2"})

        self.assertTrue(result.is_error)
        self.assertIn("Blocked bash command", tool_result_text(result))

    def test_bash_agent_run_runs_tool_then_finalizes(self) -> None:
        state = _run_fake_bash("demo ok")
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
        # The fake provider's calls are recorded (and tagged api="fake"
        # elsewhere), not hidden — so model_request events are present.
        self.assertTrue(any(event.kind == "model_request" for event in state.events))

    def test_run_trace_from_state_produces_span_tree(self) -> None:
        state = _run_fake_bash("trace ok")
        trace = run_trace_from_state(
            state=state, trace_id="test.bash_agent", producer="tests"
        )

        spans = trace.spans()
        model_calls = [s for s in spans if s.kind == "model_call"]
        tool_calls = [s for s in spans if s.kind == "tool_call"]
        turns = [s for s in spans if s.kind == "turn"]
        agent_runs = [s for s in spans if s.kind == "agent_run"]

        # The fake provider's model_call spans are kept and tagged api="fake"
        # (so a consumer can filter), not dropped from the tree.
        self.assertGreaterEqual(len(model_calls), 1)
        self.assertTrue(
            all((s.attributes or {}).get("api") == "fake" for s in model_calls)
        )
        self.assertGreaterEqual(len(tool_calls), 1)
        self.assertGreaterEqual(len(turns), 1)
        self.assertEqual(len(agent_runs), 1)

        first_tool = tool_calls[0]
        self.assertEqual(first_tool.attributes["tool_name"], "bash")
        self.assertIsNotNone(first_tool.parent_id)

        event_kinds = [e.kind.value for e in trace.events]
        self.assertIn("model_request", event_kinds)
        self.assertIn("tool_execution_start", event_kinds)
        header, lines, _pool = event_stream(trace)
        self.assertEqual(lines[0]["kind"], "message")
        # Spans are derived by the reader, not embedded in the v5 stream.
        self.assertNotIn("spans", header)
        self.assertGreater(len(trace.spans()), 0)
        self.assertEqual(event_record(state.events[0])["kind"], "message")


class TraceSpanTest(unittest.TestCase):
    """Tests for span extraction edge cases and serialization."""

    def test_parent_id_returns_none_when_skip_exhausts_stack(self) -> None:
        """Parallel tool spans must share their turn parent."""
        from simple_long_horizon_agent.protocols import (
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

    def test_fake_provider_trace_has_tagged_model_turns(self) -> None:
        turns = _fake_bash_trace("mt ok").model_turns()
        # Fake turns are emitted and tagged api="fake" so a training exporter
        # can filter them, rather than the runtime dropping them silently.
        self.assertGreaterEqual(len(turns), 1)
        self.assertTrue(all((t.meta or {}).get("api") == "fake" for t in turns))

    def test_v5_stream_omits_derived_layers_but_keeps_them_derivable(self) -> None:
        trace = _fake_bash_trace("rec ok")
        header, lines, _pool = event_stream(trace)

        # v5: spans / model_turns / cost / messages are NOT embedded — the reader
        # derives them from the event lines (the viewer already does).
        for key in ("spans", "model_turns", "cost", "messages"):
            self.assertNotIn(key, header)
        self.assertTrue(lines and all("kind" in line for line in lines))
        # …but they remain derivable, and fake turns stay tagged for filtering.
        turns = trace.model_turns()
        self.assertGreater(len(turns), 0)
        self.assertTrue(all((t.meta or {}).get("api") == "fake" for t in turns))

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

    def _delegated_trace(self, trace_id: str) -> RunTrace:
        sub = Agent("sub", self._make_echo_generate("sub"), role="echo")
        parent = Agent(
            "parent",
            _make_delegating_generate("parent", "sub"),
            role="orchestrator",
        )
        parent.tools = [task_tool([sub])]
        state, events = parent.run("do it", max_turns=3)
        list(events)
        return run_trace_from_state(state=state, trace_id=trace_id, producer="tests")

    def test_collect_sub_events_extracts_from_tool_result_message(self) -> None:
        """Collect nested events by tool call ID."""
        from simple_long_horizon_agent.protocols import AgentStartEvent, AgentEndEvent

        fake_sub_events = [
            AgentStartEvent(index=0, elapsed=0.0),
            AgentEndEvent(index=1, elapsed=1.0, reason="done"),
        ]
        msg = user_message(
            "result text",
            kind="tool_result",
            sidecar={"details": {"call_42": {"sub_events": fake_sub_events}}},
        )
        collected = _collect_sub_events([msg])
        self.assertIn("call_42", collected)
        self.assertEqual(len(collected["call_42"]), 2)

    def test_collect_sub_events_skips_irrelevant_messages(self) -> None:
        cases = {
            "non-tool result": user_message("hello", kind="message"),
            "empty details": user_message(
                "r", kind="tool_result", sidecar={"details": {}}
            ),
            "missing sub_events": user_message(
                "r",
                kind="tool_result",
                sidecar={"details": {"call_1": {"other": "stuff"}}},
            ),
        }
        for name, message in cases.items():
            with self.subTest(name):
                self.assertEqual(_collect_sub_events([message]), {})

    def test_task_tool_merged_spans_inlines_sub_agent(self) -> None:
        """Inline sub-agent spans beneath the task tool span."""
        trace = self._delegated_trace("test.merge")

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
        trace = _fake_bash_trace("nosub")
        self.assertEqual(trace.spans(), trace.merged_spans())

    def test_task_tool_sub_events_survive_json_round_trip(self) -> None:
        """Keep sub-event kind discriminators in trace-viewer JSON."""
        trace = self._delegated_trace("test.round_trip")
        _header, lines, _pool = event_stream(trace)
        parsed = json.loads(json.dumps(lines))

        sub_event_lists: list[list[dict]] = []
        for ev in parsed:
            if ev.get("kind") != "message":
                continue
            msg = ev.get("message") or {}
            details = (msg.get("sidecar") or {}).get("details") or {}
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
            from simple_long_horizon_agent.messages import ToolCallBlock

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
                kind="step",
            )
        return assistant_message("done", sender=name, target="user", kind="final")

    return generate


class BashToolCrashSafetyTest(unittest.TestCase):
    """Defenses against non-UTF-8 output and NaN/zero/non-numeric timeouts."""

    def test_utf8_output_is_decoded_without_crashing(self) -> None:
        cases = [
            ("invalid stdout", r"printf '\xff\xfe\xfd'", "raw_stdout", "���"),
            ("invalid stderr", r"printf '\xff' >&2", "raw_stderr", "�"),
            ("mixed stdout", r"printf 'ok\xffok'", "raw_stdout", "ok�ok"),
            ("truncated sequence", r"printf '\xc3\x28'", "raw_stdout", "�("),
            ("valid emoji", "printf '🦀\\n'", "raw_stdout", "🦀\n"),
        ]
        for name, command, stream, expected in cases:
            with self.subTest(name):
                execution = run_bash(command, cwd=ROOT, timeout_seconds=3)
                self.assertEqual(execution.exit_code, 0)
                self.assertFalse(execution.is_error)
                self.assertEqual(getattr(execution, stream), expected)

    def test_binary_output_via_tool_returns_structured_result(self) -> None:
        tool = make_bash_tool(cwd=ROOT)
        result = _execute(tool, {"command": r"printf '\xff\xfe'"})
        self.assertFalse(result.is_error)
        self.assertEqual(result.details["exit_code"], 0)
        # Truncation note must NOT fire for tiny binary output.
        self.assertFalse(result.details["stdout_truncated"])
        self.assertEqual(result.details["raw_stdout"], "��")

    def test_resolve_timeout_rejects_invalid_values(self) -> None:
        cases = [
            ("float NaN", float("nan"), "NaN"),
            ("string NaN", "nan", "NaN"),
            ("zero", 0, None),
            ("negative", -1, None),
            ("non-numeric", "not-a-number", None),
        ]
        for name, value, message in cases:
            with self.subTest(name):
                with self.assertRaises(ValueError) as ctx:
                    _resolve_timeout(value, 5.0, 60.0)
                if message:
                    self.assertIn(message, str(ctx.exception))

    def test_resolve_timeout_uses_defaults_and_caps(self) -> None:
        self.assertEqual(MAX_BASH_TIMEOUT_SECONDS, 300.0)
        cases = [
            ("infinity", float("inf"), 5.0, 60.0, 60.0),
            ("missing", None, 5.0, 60.0, 5.0),
            ("empty", "", 5.0, 60.0, 5.0),
            ("overshoot", 9999, 5.0, 60.0, 60.0),
            (
                "public cap",
                MAX_BASH_TIMEOUT_SECONDS * 10,
                5.0,
                MAX_BASH_TIMEOUT_SECONDS,
                MAX_BASH_TIMEOUT_SECONDS,
            ),
        ]
        for name, value, default, maximum, expected in cases:
            with self.subTest(name):
                self.assertEqual(
                    _resolve_timeout(value, default, maximum),
                    expected,
                )

    def test_invalid_timeouts_return_structured_tool_errors(self) -> None:
        cases = [
            ("NaN", float("nan"), "NaN"),
            ("zero", 0, None),
            ("non-numeric", "soon", None),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tool = make_bash_tool(cwd=tmp)
            for name, value, detail in cases:
                with self.subTest(name):
                    result = _execute(
                        tool,
                        {"command": "echo hi", "timeout_seconds": value},
                    )
                    text = tool_result_text(result)
                    self.assertTrue(result.is_error)
                    self.assertIn("Invalid bash timeout", text)
                    if detail:
                        self.assertIn(detail, text)

    def test_partial_binary_output_before_timeout_does_not_crash(self) -> None:
        # Emit invalid UTF-8 then sleep past the timeout. With errors='replace'
        # the partial bytes captured by TimeoutExpired must decode cleanly.
        execution = run_bash(
            r"printf '\xff\xfe' && sleep 5",
            cwd=ROOT,
            timeout_seconds=3,
        )
        self.assertTrue(execution.timed_out)
        self.assertLess(execution.exit_code, 0)
        # Replacement chars survived, plus our timeout banner is appended.
        self.assertIn("�", execution.raw_stdout)
        self.assertIn("Timed out", execution.raw_stderr)


class BashAttachTest(unittest.TestCase):
    """`attach` inlines file paths as image content blocks on the ToolResult."""

    def _attach(
        self,
        paths: list[str],
        *,
        create: tuple[str, ...] = (),
        max_bytes: int | None = None,
    ) -> ToolResult:
        with tempfile.TemporaryDirectory() as tmp:
            for relative in create:
                path = Path(tmp) / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(_make_red_png())
            options = {} if max_bytes is None else {"max_attach_bytes": max_bytes}
            return _execute(
                make_bash_tool(cwd=tmp, **options),
                {"command": "true", "description": "noop", "attach": paths},
            )

    def test_attach_inlines_png_paths_relative_to_cwd(self) -> None:
        for name, relative in [
            ("direct child", "red.png"),
            ("nested child", "out/shot.png"),
        ]:
            with self.subTest(name):
                result = self._attach([relative], create=(relative,))
                image_blocks = [b for b in result.content if b.kind == "image"]
                self.assertEqual(len(image_blocks), 1)
                self.assertEqual(image_blocks[0].mime_type, "image/png")
                self.assertTrue(image_blocks[0].data)

    def test_attach_reports_unusable_files(self) -> None:
        cases = [
            (
                "missing",
                ["missing.png", "also-missing.jpg"],
                (),
                None,
                ("missing.png", "not a file"),
            ),
            ("oversize", ["big.png"], ("big.png",), 10, ("exceeds limit",)),
        ]
        for name, paths, create, max_bytes, notes in cases:
            with self.subTest(name):
                result = self._attach(paths, create=create, max_bytes=max_bytes)
                self.assertFalse(any(b.kind == "image" for b in result.content))
                text = "\n".join(b.text for b in result.content if b.kind == "text")
                for note in notes:
                    self.assertIn(note, text)
