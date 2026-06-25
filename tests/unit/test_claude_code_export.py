"""Claude Code transcript export: record shape, session-per-compaction, sidechains."""

from __future__ import annotations

import json
import tempfile
import unittest

from simple_agent_lab.messages import (
    TextBlock,
    TokenUsage,
    ToolCallBlock,
    assistant_message,
    runtime_message,
    tool_result_message,
    user_message,
)
from simple_agent_lab.protocols import (
    ContextCompressionEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnStartEvent,
)
from simple_agent_lab.state import State
from simple_agent_lab.trace import (
    claude_code_sessions,
    event_record,
    run_trace_from_state,
    write_claude_code_sessions,
)


def _basic_state() -> State:
    state = State(task="t", run_id="run")
    state.send("task", "user", "agent", "do the thing")
    state.record_event(TurnStartEvent(agent="agent"))
    state.record(
        assistant_message(
            content=(
                TextBlock("looking"),
                ToolCallBlock(id="c1", name="bash", arguments={"cmd": "ls"}),
            ),
            sender="agent",
            kind="step",
            model="m",
            usage=TokenUsage(input_tokens=5, output_tokens=2),
        )
    )
    state.record_event(ToolExecutionStartEvent(tool_call_id="c1", tool_name="bash"))
    state.record(
        tool_result_message(
            "files", tool_call_id="c1", tool_name="bash", target="agent"
        )
    )
    state.record_event(
        ToolExecutionEndEvent(
            tool_call_id="c1", tool_name="bash", is_error=False, terminate=False
        )
    )
    state.record(assistant_message("done", sender="agent", kind="final", model="m"))
    return state


class ClaudeCodeExportTest(unittest.TestCase):
    def test_single_session_record_shape_and_chain(self) -> None:
        trace = run_trace_from_state(
            state=_basic_state(), trace_id="trace1", producer="p"
        )
        sessions = claude_code_sessions(trace)
        self.assertEqual(len(sessions), 1)
        records = sessions[0].records
        self.assertEqual(len(records), 4)  # task, assistant, tool_result, final

        # parentUuid chains within the session; the first record roots at None.
        self.assertIsNone(records[0]["parentUuid"])
        self.assertEqual(records[1]["parentUuid"], records[0]["uuid"])
        self.assertEqual(records[2]["parentUuid"], records[1]["uuid"])
        self.assertEqual(records[3]["parentUuid"], records[2]["uuid"])

        self.assertTrue(all(r["sessionId"] == "trace1" for r in records))
        self.assertTrue(all(r["isSidechain"] is False for r in records))

        assistant = records[1]
        self.assertEqual(assistant["type"], "assistant")
        content = assistant["message"]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "looking"})
        self.assertEqual(
            content[1],
            {"type": "tool_use", "id": "c1", "name": "bash", "input": {"cmd": "ls"}},
        )
        self.assertEqual(assistant["message"]["usage"]["input_tokens"], 5)
        self.assertEqual(assistant["message"]["model"], "m")

        tool_result = records[2]
        self.assertEqual(tool_result["type"], "user")
        block = tool_result["message"]["content"][0]
        self.assertEqual(block["type"], "tool_result")
        self.assertEqual(block["tool_use_id"], "c1")
        self.assertEqual(block["content"], "files")

    def test_compaction_splits_into_two_sessions_with_summary_bridge(self) -> None:
        state = State(task="t", run_id="run")
        state.send("task", "user", "agent", "start")  # 0
        state.record(assistant_message("a1", sender="agent", kind="step"))  # 1
        state.record(user_message("u1", sender="user", target="agent"))  # 2
        state.record(assistant_message("a2", sender="agent", kind="step"))  # 3
        # Fold messages 1-2 into a summary recorded at index 4 (as the runtime does).
        state.record(
            runtime_message(
                "SUMMARY of 1-2", sender="system", target="agent", kind="summary"
            )
        )  # 4
        state.record_event(
            ContextCompressionEvent(
                agent="agent",
                summary_message_index=4,
                compressed_message_indices=[1, 2],
                active_context_indices=[0, 4, 3],
                before_tokens=100,
                after_tokens=20,
                strategy="summarize",
            )
        )
        state.record(
            assistant_message("after compact", sender="agent", kind="final")
        )  # 5

        trace = run_trace_from_state(state=state, trace_id="trace1", producer="p")
        sessions = claude_code_sessions(trace)
        self.assertEqual(len(sessions), 2)

        # Session 0 holds the full pre-compaction history (no summary record).
        s0 = sessions[0]
        self.assertEqual(s0.session_id, "trace1")
        self.assertEqual(len(s0.records), 4)
        self.assertTrue(all(r.get("type") != "summary" for r in s0.records))

        # Session 1 opens with a summary that bridges back to session 0's tail.
        s1 = sessions[1]
        self.assertEqual(s1.session_id, "trace1-compact1")
        summary = s1.records[0]
        self.assertEqual(summary["type"], "summary")
        self.assertEqual(summary["summary"], "SUMMARY of 1-2")
        self.assertEqual(summary["leafUuid"], s0.records[-1]["uuid"])
        # The post-compaction message is a fresh session root.
        post = s1.records[1]
        self.assertEqual(post["sessionId"], "trace1-compact1")
        self.assertIsNone(post["parentUuid"])
        self.assertEqual(post["message"]["content"][0]["text"], "after compact")

    def test_sub_agent_becomes_sidechain_in_same_session(self) -> None:
        sub = State(task="search", run_id="sub")
        sub.send("task", "agent", "searcher", "find X")
        sub.record(assistant_message("found X", sender="searcher", kind="final"))
        sub_event_dicts = [event_record(e) for e in sub.events]

        state = State(task="t", run_id="run")
        state.send("task", "user", "agent", "delegate")
        state.record(
            assistant_message(
                content=(
                    ToolCallBlock(
                        id="task1", name="task", arguments={"prompt": "find X"}
                    ),
                ),
                sender="agent",
                kind="step",
            )
        )
        state.record(
            tool_result_message(
                "sub done",
                tool_call_id="task1",
                tool_name="task",
                target="agent",
                sidecar={"details": {"task1": {"sub_events": sub_event_dicts}}},
            )
        )
        state.record(assistant_message("final", sender="agent", kind="final"))

        trace = run_trace_from_state(state=state, trace_id="trace1", producer="p")
        sessions = claude_code_sessions(trace)
        self.assertEqual(len(sessions), 1)
        records = sessions[0].records

        sidechain = [r for r in records if r["isSidechain"]]
        main = [r for r in records if not r["isSidechain"]]
        self.assertEqual(len(main), 4)  # task, assistant(tool_use), tool_result, final
        self.assertEqual(len(sidechain), 2)  # sub task + sub final

        # Sidechain has its own chain rooted at None, all in the parent session.
        self.assertIsNone(sidechain[0]["parentUuid"])
        self.assertEqual(sidechain[1]["parentUuid"], sidechain[0]["uuid"])
        self.assertTrue(all(r["sessionId"] == "trace1" for r in sidechain))
        self.assertEqual(sidechain[0]["message"]["content"][0]["text"], "find X")
        self.assertEqual(sidechain[1]["message"]["content"][0]["text"], "found X")

        # The main chain skips the sidechain: final links to the tool_result.
        self.assertEqual(main[3]["parentUuid"], main[2]["uuid"])

    def test_write_sessions_emits_one_file_per_session(self) -> None:
        state = State(task="t", run_id="run")
        state.send("task", "user", "agent", "start")
        state.record(
            runtime_message("S", sender="system", target="agent", kind="summary")
        )
        state.record_event(
            ContextCompressionEvent(
                agent="agent",
                summary_message_index=1,
                compressed_message_indices=[0],
                active_context_indices=[1],
                before_tokens=10,
                after_tokens=5,
            )
        )
        state.record(assistant_message("go", sender="agent", kind="final"))
        trace = run_trace_from_state(state=state, trace_id="t1", producer="p")

        with tempfile.TemporaryDirectory() as tmp:
            paths = write_claude_code_sessions(
                trace, tmp, cwd="/repo", git_branch="main"
            )
            self.assertEqual([p.name for p in paths], ["t1.jsonl", "t1-compact1.jsonl"])
            # Each file is compact line-delimited JSON (one record per line).
            lines = paths[1].read_text(encoding="utf-8").strip().splitlines()
            first = json.loads(lines[0])
            self.assertEqual(first["type"], "summary")
            self.assertEqual(json.loads(lines[1])["cwd"], "/repo")
            self.assertEqual(json.loads(lines[1])["gitBranch"], "main")


if __name__ == "__main__":
    unittest.main()
