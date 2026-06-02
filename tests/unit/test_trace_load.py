from __future__ import annotations

import json
import unittest

from simple_agent_lab import (
    Agent,
    Message,
    ToolCallBlock,
    assistant_message,
)
from simple_agent_lab.tools import AgentTool, text_result
from simple_agent_lab.trajectory import (
    event_from_dict,
    message_from_dict,
    run_trace_from_state,
    state_from_trace_record,
    trace_record,
)


def _tool_then_final_agent() -> Agent:
    """A scripted agent that issues two tool calls (one per turn) then finishes.

    Exercises the full event surface — turns, model request/response, tool
    execution start/end, and message events — so the round-trip covers more
    than just plain text messages.
    """
    turns = {"n": 0}

    def gen(visible: list[Message]) -> Message:
        del visible
        if turns["n"] < 2:
            path = "ab"[turns["n"]]
            turns["n"] += 1
            return assistant_message(
                [ToolCallBlock(id=f"c{path}", name="touch", arguments={"path": path})],
                sender="w",
                target="user",
                kind="step",
            )
        return assistant_message("done", sender="w", target="user", kind="final")

    def touch(call_id, args, abort, on_update):  # noqa: ANN001 - test stub
        del call_id, abort, on_update
        return text_result(f"created {args['path']}")

    tool = AgentTool(
        name="touch", description="create a file", parameters={}, execute=touch
    )
    return Agent("w", gen, tools=(tool,))


class TraceLoadTest(unittest.TestCase):
    def _record(self) -> dict:
        agent = _tool_then_final_agent()
        state, events = agent.run("the task", max_turns=5)
        for _ in events:
            pass
        record = trace_record(
            run_trace_from_state(state=state, trace_id="t", producer="p")
        )
        # Force the JSON boundary a persisted trace would cross.
        return json.loads(json.dumps(record))

    def test_state_round_trips_messages(self) -> None:
        record = self._record()
        restored = state_from_trace_record(record)

        # The message mirror in the record is rebuilt from events identically.
        self.assertEqual(len(restored.messages), len(record["messages"]))
        restored_record = trace_record(
            run_trace_from_state(state=restored, trace_id="t", producer="p")
        )
        self.assertEqual(restored_record["messages"], record["messages"])

    def test_events_round_trip_is_idempotent(self) -> None:
        record = self._record()
        restored = state_from_trace_record(record)
        restored_record = trace_record(
            run_trace_from_state(state=restored, trace_id="t", producer="p")
        )
        # Re-serializing the rebuilt state reproduces the event log exactly,
        # including index/elapsed stamps and the kind discriminators.
        self.assertEqual(restored_record["events"], record["events"])

    def test_event_kinds_all_supported(self) -> None:
        record = self._record()
        kinds = {e["kind"] for e in record["events"]}
        # Every kind in the trace deserializes without raising.
        for event_dict in record["events"]:
            self.assertEqual(event_from_dict(event_dict).kind, event_dict["kind"])
        self.assertIn("tool_execution_end", kinds)
        self.assertIn("model_request", kinds)

    def test_message_from_dict_rejects_unknown_role(self) -> None:
        with self.assertRaises(ValueError):
            message_from_dict({"role": "nope", "content": []})

    def test_event_from_dict_rejects_unknown_kind(self) -> None:
        with self.assertRaises(ValueError):
            event_from_dict({"kind": "nope"})


if __name__ == "__main__":
    unittest.main()
