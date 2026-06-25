from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from simple_agent_lab.protocols import AgentEndEvent, AgentStartEvent
from simple_agent_lab.state import State
from simple_agent_lab.trace import (
    TRACE_HEADER_TYPE,
    LiveTraceSession,
    TraceMeta,
    read_jsonl,
    trace_record_from_jsonl,
    write_canonical_trace,
    write_jsonl_atomic,
)


class LiveTraceTest(unittest.TestCase):
    def test_write_jsonl_atomic_leaves_no_part_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            write_jsonl_atomic(path, [{"ok": True}])
            self.assertTrue(path.is_file())
            self.assertFalse(path.with_name(f"{path.name}.part").exists())

    def test_live_trace_session_streams_then_folds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.jsonl"
            state = State(task="demo")
            with LiveTraceSession(
                path,
                state,
                trace_id="test.live",
                producer="test:live",
                flush_interval_s=0.05,
                final_flush_on_exit=False,
            ):
                state.record_event(AgentStartEvent())
                time.sleep(0.15)
                state.record_event(AgentEndEvent(reason="done"))

            self.assertTrue(path.is_file())
            # Mid/end-of-run shape (final_flush_on_exit=False): an append-only
            # event stream -- a header line followed by one event per line.
            live_records = read_jsonl(path)
            self.assertEqual(live_records[0]["type"], TRACE_HEADER_TYPE)
            self.assertEqual(live_records[0]["trace_id"], "test.live")
            event_records = live_records[1:]
            self.assertGreaterEqual(len(event_records), 1)
            self.assertEqual(event_records[0]["kind"], "agent_start")
            self.assertFalse(path.with_name(f"{path.name}.part").exists())

            # The stream folds back into the canonical single record on read.
            folded = trace_record_from_jsonl(path)
            self.assertEqual(folded["trace_id"], "test.live")
            self.assertEqual(folded["type"], "trajectory")
            self.assertGreaterEqual(len(folded["events"]), 1)

            # An explicit canonical write replaces the stream with one record.
            write_canonical_trace(
                path,
                state=state,
                trace_meta=TraceMeta("test.live", "test:live"),
            )
            final_records = read_jsonl(path)
            self.assertEqual(len(final_records), 1)
            self.assertEqual(final_records[0]["trace_id"], "test.live")
            self.assertGreaterEqual(len(final_records[0]["events"]), 2)
            # trace_record_from_jsonl passes a finished record through unchanged.
            self.assertEqual(trace_record_from_jsonl(path), final_records[0])

    def test_event_stream_folds_to_canonical_record(self) -> None:
        """A header+events stream reconstructs the same record as direct export."""
        import json

        from simple_agent_lab.messages import (
            TextBlock,
            ToolCallBlock,
            TokenUsage,
            assistant_message,
            tool_result_message,
        )
        from simple_agent_lab.protocols import (
            ToolExecutionEndEvent,
            ToolExecutionStartEvent,
            TurnStartEvent,
        )
        from simple_agent_lab.trace import (
            event_record,
            run_trace_from_state,
            trace_header_record,
            trace_record,
        )

        state = State(task="demo", run_id="fixed")
        state.send("task", "user", "agent", "do it")
        state.record_event(TurnStartEvent(agent="agent"))
        state.record(
            assistant_message(
                content=(
                    TextBlock("looking"),
                    ToolCallBlock(id="c1", name="bash", arguments={"cmd": "ls"}),
                ),
                sender="agent",
                kind="step",
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            )
        )
        state.record_event(ToolExecutionStartEvent(tool_call_id="c1", tool_name="bash"))
        state.record(
            tool_result_message("out", tool_call_id="c1", tool_name="bash", target="agent")
        )
        state.record_event(
            ToolExecutionEndEvent(
                tool_call_id="c1", tool_name="bash", is_error=False, terminate=False
            )
        )
        state.record(assistant_message("done", sender="agent", kind="final"))

        canonical = trace_record(
            run_trace_from_state(state=state, trace_id="t", producer="p")
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stream.jsonl"
            with path.open("w", encoding="utf-8") as f:
                header = trace_header_record(trace_id="t", producer="p", task="demo")
                f.write(json.dumps(header) + "\n")
                for event in state.events:
                    f.write(json.dumps(event_record(event)) + "\n")
            folded = trace_record_from_jsonl(path)

        for key in ("trace_id", "task", "events", "messages", "spans", "model_turns"):
            self.assertEqual(folded[key], canonical[key], f"mismatch on {key}")

    def test_final_flush_on_exit_writes_without_extra_canonical_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.jsonl"
            state = State(task="demo")
            with LiveTraceSession(
                path,
                state,
                trace_id="test.live.exit",
                producer="test:live",
                flush_interval_s=10.0,
                final_flush_on_exit=True,
            ):
                state.record_event(AgentStartEvent())
                state.record_event(AgentEndEvent(reason="done"))

            records = read_jsonl(path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["trace_id"], "test.live.exit")
            self.assertFalse(path.with_name(f"{path.name}.part").exists())


if __name__ == "__main__":
    unittest.main()
