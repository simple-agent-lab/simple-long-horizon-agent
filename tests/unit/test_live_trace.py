from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from simple_agent_lab.protocols import AgentEndEvent, AgentStartEvent
from simple_agent_lab.state import State
from simple_agent_lab.trajectory import (
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
            live_records = read_jsonl(path)
            self.assertEqual(len(live_records), 1)
            self.assertGreaterEqual(len(live_records[0]["events"]), 1)
            self.assertEqual(live_records[0]["events"][0]["kind"], "agent_start")
            self.assertFalse(path.with_name(f"{path.name}.part").exists())

            write_canonical_trace(
                path,
                state=state,
                trace_meta=TraceMeta("test.live", "test:live"),
            )
            final_records = read_jsonl(path)
            self.assertEqual(len(final_records), 1)
            self.assertEqual(final_records[0]["trace_id"], "test.live")
            self.assertGreaterEqual(len(final_records[0]["events"]), 2)

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
