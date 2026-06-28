from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from simple_agent_lab.messages import AssistantMessage, TextBlock
from simple_agent_lab.protocols import AgentEndEvent, AgentStartEvent, MessageEvent
from simple_agent_lab.state import State
from simple_agent_lab.trace import (
    LiveTraceSession,
    TraceMeta,
    read_jsonl,
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

    def test_live_trace_session_writes_incremental_snapshots(self) -> None:
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
            # v5 stream: a header line then one line per event (appended live).
            live = read_jsonl(path)
            header, events = live[0], live[1:]
            self.assertEqual(header["trace_id"], "test.live")
            self.assertNotIn("events", header)
            self.assertGreaterEqual(len(events), 1)
            self.assertEqual(events[0]["kind"], "agent_start")
            self.assertFalse(path.with_name(f"{path.name}.part").exists())

            write_canonical_trace(
                path,
                state=state,
                trace_meta=TraceMeta("test.live", "test:live"),
            )
            final = read_jsonl(path)
            self.assertEqual(final[0]["trace_id"], "test.live")
            self.assertGreaterEqual(len(final[1:]), 2)

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
            self.assertEqual(records[0]["trace_id"], "test.live.exit")
            self.assertNotIn("events", records[0])
            self.assertFalse(path.with_name(f"{path.name}.part").exists())

    def test_canonical_trace_externalizes_raw_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.jsonl"
            state = State(task="demo")
            state.record_event(
                MessageEvent(
                    message=AssistantMessage(
                        content=(TextBlock(text="hi"),),
                        sender="a",
                        target="a",
                        sidecar={
                            "raw": {"request": {"model": "m"}, "response": {"id": "r"}}
                        },
                    )
                )
            )
            write_canonical_trace(
                path, state=state, trace_meta=TraceMeta("raw", "test:raw")
            )

            # The message event's raw snapshot is externalized to a {raw_ref}
            # pointer; the blob lands in the sibling pool.
            stream = read_jsonl(path)
            msg_event = next(e for e in stream[1:] if e["kind"] == "message")
            self.assertEqual(msg_event["message"]["sidecar"]["raw"], {"raw_ref": 0})
            raw_records = read_jsonl(path.with_name(f"{path.name}.raw.jsonl"))
            self.assertEqual(raw_records[0]["request"]["model"], "m")


if __name__ == "__main__":
    unittest.main()
