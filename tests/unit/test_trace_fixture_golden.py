"""Defense against trace-producer / trace-viewer schema drift (the `data` ->
`sidecar` bug class).

The viewer's `sample-trace.jsonl` used to be hand-authored, so it drifted away
from what the runtime actually serializes and silently masked a field rename.
Here the sample is GENERATED from the real `Message`/event dataclasses through
the real `trace_record` path, and a golden test asserts the committed file still
matches. If anyone renames a serialized trace field (e.g. `Message.sidecar`),
this test fails — surfacing the change in a PR diff and forcing the viewer to
follow instead of quietly breaking.

Regenerate after an intentional schema change and review the diff:

    UPDATE_GOLDEN=1 uv run python -m unittest tests.unit.test_trace_fixture_golden
"""

from __future__ import annotations

import dataclasses
import json
import os
import unittest
from pathlib import Path

from simple_agent_lab.messages import (
    AssistantMessage,
    TextBlock,
    ThinkingBlock,
    TokenUsage,
    ToolCallBlock,
    ToolResultBlock,
    UserMessage,
)
from simple_agent_lab.protocols import (
    AgentEndEvent,
    AgentStartEvent,
    ContextCompressionEvent,
    Event,
    MessageEvent,
    ModelRequestEvent,
    ModelResponseEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from simple_agent_lab.state import State
from simple_agent_lab.trace import (
    collect_agents,
    event_record,
    run_trace_from_state,
    trace_header,
)
from simple_agent_lab.trace.jsonl import read_jsonl, write_jsonl

_VIEWER_DIR = Path(__file__).resolve().parents[2] / "studio" / "trace-viewer"
SAMPLE_PATH = _VIEWER_DIR / "sample-trace.jsonl"
# The viewer is a single self-contained file: the sample it shows on first paint
# (and via the "Sample" button) is embedded in this <script> block, not fetched
# from sample-trace.jsonl. Both must stay in lockstep with the real serializer.
INDEX_HTML_PATH = _VIEWER_DIR / "index.html"
_EMBED_OPEN = '<script type="application/json" id="embedded-sample">'
_EMBED_CLOSE = "</script>"


def _read_embedded_sample() -> list:
    html = INDEX_HTML_PATH.read_text(encoding="utf-8")
    start = html.index(_EMBED_OPEN) + len(_EMBED_OPEN)
    end = html.index(_EMBED_CLOSE, start)
    return json.loads(html[start:end].strip())


def _write_embedded_sample(stream: list) -> None:
    html = INDEX_HTML_PATH.read_text(encoding="utf-8")
    start = html.index(_EMBED_OPEN) + len(_EMBED_OPEN)
    end = html.index(_EMBED_CLOSE, start)
    payload = "\n" + json.dumps(stream, ensure_ascii=False) + "\n"
    INDEX_HTML_PATH.write_text(html[:start] + payload + html[end:], encoding="utf-8")


TRACE_ID = "demo.observatory.001"
PRODUCER = "demo:trace-viewer"
MODEL = "demo/observatory-mini"
API = "openai-chat"
PARENT = "obs_agent"
SUB = "search_agent"
TASK = (
    "Investigate the failing wc-line counter regression and ship a fix with a new test."
)
# Static so the record is byte-stable across runs (no clock / no run-counters).
META = {"model": MODEL, "provider": "fake", "run_label": "observatory-demo"}

# A provider raw wire snapshot, the slot the viewer's "Wire debug" panel reads
# off `message.sidecar.raw`. OpenAI-chat shaped with reasoning_effort so the
# sample also exercises the reasoning fields end to end.
RAW_BLOB = {
    "request": {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are obs_agent."},
            {"role": "user", "content": [{"type": "text", "text": TASK}]},
        ],
        "reasoning_effort": "high",
        "temperature": 1.0,
        "max_tokens": 400,
    },
    "response": {
        "id": "chatcmpl-demo-0001",
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "Looking at the repo layout.",
                    "reasoning": "Inspect the layout before touching files.",
                },
            }
        ],
        "usage": {
            "prompt_tokens": 612,
            "completion_tokens": 88,
            "total_tokens": 700,
            "completion_tokens_details": {"reasoning_tokens": 40},
        },
    },
}


def _stamp(events: list[Event]) -> list[Event]:
    """Assign deterministic, monotonically increasing index/elapsed."""
    return [
        dataclasses.replace(e, index=i, elapsed=round(i * 0.25, 3))
        for i, e in enumerate(events)
    ]


def _sub_events() -> list[Event]:
    """The search sub-agent's own trace, nested under the task tool result."""
    return _stamp(
        [
            MessageEvent(
                message=UserMessage(
                    content=(TextBlock("Search the repo for wc_lines callers."),),
                    sender=PARENT,
                    target=SUB,
                    kind="task",
                )
            ),
            AgentStartEvent(agent=SUB, system_prompt=f"You are {SUB}."),
            TurnStartEvent(agent=SUB),
            ModelRequestEvent(
                agent=SUB,
                api=API,
                visible_count=2,
                llm_message_count=2,
                context_view={"input_tokens_estimate": 188},
                tools=[{"name": "bash", "description": "Run a bash command."}],
                llm_payload=[],
            ),
            ModelResponseEvent(
                agent=SUB,
                api=API,
                output_kind="thought",
                target=SUB,
                tool_call_count=1,
                usage=TokenUsage(input_tokens=188, output_tokens=40),
                model=MODEL,
            ),
            MessageEvent(
                message=AssistantMessage(
                    content=(
                        TextBlock("Running ripgrep."),
                        ToolCallBlock(
                            id="sub_call_01",
                            name="bash",
                            arguments={"command": "rg -n wc_lines"},
                        ),
                    ),
                    sender=SUB,
                    target=SUB,
                    kind="thought",
                )
            ),
            ToolExecutionStartEvent(tool_call_id="sub_call_01", tool_name="bash"),
            ToolExecutionEndEvent(
                tool_call_id="sub_call_01",
                tool_name="bash",
                is_error=False,
                terminate=False,
            ),
            MessageEvent(
                message=UserMessage(
                    content=(
                        ToolResultBlock(
                            tool_call_id="sub_call_01",
                            tool_name="bash",
                            content=(
                                TextBlock(
                                    "src/simple_agent_lab/tools/wc.py:1:def wc_lines"
                                ),
                            ),
                            is_error=False,
                        ),
                    ),
                    sender="tool",
                    target=SUB,
                    kind="tool_result",
                )
            ),
            TurnEndEvent(agent=SUB),
            ModelResponseEvent(
                agent=SUB,
                api=API,
                output_kind="final",
                target=PARENT,
                tool_call_count=0,
                usage=TokenUsage(input_tokens=264, output_tokens=52),
                model=MODEL,
            ),
            MessageEvent(
                message=AssistantMessage(
                    content=(
                        TextBlock("wc_lines: 1 caller, 1 test. Low blast radius."),
                    ),
                    sender=SUB,
                    target=PARENT,
                    kind="final",
                )
            ),
            AgentEndEvent(reason="final"),
        ]
    )


def _build_events() -> list[Event]:
    """Hand-built but via REAL dataclasses, so the serialized shape is coupled to
    the runtime schema; covers every slot the viewer reads."""
    return _stamp(
        [
            MessageEvent(
                message=UserMessage(
                    content=(TextBlock(TASK),),
                    sender="user",
                    target=PARENT,
                    kind="task",
                )
            ),
            AgentStartEvent(agent=PARENT, system_prompt=f"You are {PARENT}."),
            # Turn 1 — thinking + bash (ok); carries the wire-debug raw blob.
            TurnStartEvent(agent=PARENT),
            ModelRequestEvent(
                agent=PARENT,
                api=API,
                visible_count=1,
                llm_message_count=2,
                context_view={"input_tokens_estimate": 612, "messages": 2},
                tools=[
                    {"name": "bash", "description": "Run a bash command."},
                    {"name": "task", "description": "Delegate a focused subtask."},
                ],
                llm_payload=[
                    {"role": "system", "content": "You are obs_agent."},
                    {"role": "user", "content": TASK},
                ],
            ),
            ModelResponseEvent(
                agent=PARENT,
                api=API,
                output_kind="thought",
                target=PARENT,
                tool_call_count=1,
                usage=TokenUsage(
                    input_tokens=612, output_tokens=88, cache_write_tokens=612
                ),
                model=MODEL,
            ),
            MessageEvent(
                message=AssistantMessage(
                    content=(
                        ThinkingBlock(text="Inspect the layout before editing."),
                        TextBlock("Looking at the repo layout."),
                        ToolCallBlock(
                            id="call_01",
                            name="bash",
                            arguments={"command": "ls src/simple_agent_lab"},
                        ),
                    ),
                    sender=PARENT,
                    target=PARENT,
                    kind="thought",
                    usage=TokenUsage(
                        input_tokens=612, output_tokens=88, cache_write_tokens=612
                    ),
                    model=MODEL,
                    sidecar={"raw": RAW_BLOB},
                )
            ),
            ToolExecutionStartEvent(tool_call_id="call_01", tool_name="bash"),
            ToolExecutionEndEvent(
                tool_call_id="call_01",
                tool_name="bash",
                is_error=False,
                terminate=False,
            ),
            MessageEvent(
                message=UserMessage(
                    content=(
                        ToolResultBlock(
                            tool_call_id="call_01",
                            tool_name="bash",
                            content=(
                                TextBlock("core.py\nmessages.py\nstate.py\ntools/"),
                            ),
                            is_error=False,
                        ),
                    ),
                    sender="tool",
                    target=PARENT,
                    kind="tool_result",
                )
            ),
            TurnEndEvent(agent=PARENT),
            # Turn 2 — delegate to a sub-agent; result carries nested sub_events.
            TurnStartEvent(agent=PARENT),
            ModelRequestEvent(
                agent=PARENT,
                api=API,
                visible_count=3,
                llm_message_count=4,
                context_view={"input_tokens_estimate": 894, "messages": 4},
                tools=[],
                llm_payload=[],
            ),
            ModelResponseEvent(
                agent=PARENT,
                api=API,
                output_kind="thought",
                target=PARENT,
                tool_call_count=1,
                usage=TokenUsage(
                    input_tokens=894,
                    output_tokens=120,
                    cache_read_tokens=612,
                    cache_write_tokens=282,
                ),
                model=MODEL,
            ),
            MessageEvent(
                message=AssistantMessage(
                    content=(
                        TextBlock("Dispatching a search sub-agent."),
                        ToolCallBlock(
                            id="call_02_task",
                            name="task",
                            arguments={
                                "agent": SUB,
                                "task": "Search for wc_lines callers.",
                            },
                        ),
                    ),
                    sender=PARENT,
                    target=PARENT,
                    kind="thought",
                    usage=TokenUsage(
                        input_tokens=894,
                        output_tokens=120,
                        cache_read_tokens=612,
                        cache_write_tokens=282,
                    ),
                    model=MODEL,
                )
            ),
            ToolExecutionStartEvent(tool_call_id="call_02_task", tool_name="task"),
            ToolExecutionEndEvent(
                tool_call_id="call_02_task",
                tool_name="task",
                is_error=False,
                terminate=False,
            ),
            MessageEvent(
                message=UserMessage(
                    content=(
                        ToolResultBlock(
                            tool_call_id="call_02_task",
                            tool_name="task",
                            content=(
                                TextBlock("Sub-agent: wc_lines has 1 caller, 1 test."),
                            ),
                            is_error=False,
                        ),
                    ),
                    sender="tool",
                    target=PARENT,
                    kind="tool_result",
                    sidecar={
                        "details": {"call_02_task": {"sub_events": _sub_events()}}
                    },
                )
            ),
            TurnEndEvent(agent=PARENT),
            # Turn 3 — a failing tool call, then a compression, then the final.
            TurnStartEvent(agent=PARENT),
            ModelRequestEvent(
                agent=PARENT,
                api=API,
                visible_count=5,
                llm_message_count=6,
                context_view={"input_tokens_estimate": 1842, "messages": 6},
                tools=[],
                llm_payload=[],
            ),
            ModelResponseEvent(
                agent=PARENT,
                api=API,
                output_kind="thought",
                target=PARENT,
                tool_call_count=1,
                usage=TokenUsage(
                    input_tokens=1842,
                    output_tokens=96,
                    cache_read_tokens=894,
                    cache_write_tokens=948,
                ),
                model=MODEL,
            ),
            MessageEvent(
                message=AssistantMessage(
                    content=(
                        TextBlock("Applying the fix."),
                        ToolCallBlock(
                            id="call_03",
                            name="bash",
                            arguments={"command": "patch -p0 < /tmp/missing.patch"},
                        ),
                    ),
                    sender=PARENT,
                    target=PARENT,
                    kind="thought",
                    usage=TokenUsage(
                        input_tokens=1842,
                        output_tokens=96,
                        cache_read_tokens=894,
                        cache_write_tokens=948,
                    ),
                    model=MODEL,
                )
            ),
            ToolExecutionStartEvent(tool_call_id="call_03", tool_name="bash"),
            ToolExecutionEndEvent(
                tool_call_id="call_03", tool_name="bash", is_error=True, terminate=False
            ),
            MessageEvent(
                message=UserMessage(
                    content=(
                        ToolResultBlock(
                            tool_call_id="call_03",
                            tool_name="bash",
                            content=(
                                TextBlock("patch: /tmp/missing.patch: No such file"),
                            ),
                            is_error=True,
                        ),
                    ),
                    sender="tool",
                    target=PARENT,
                    kind="tool_result",
                )
            ),
            TurnEndEvent(agent=PARENT),
            ContextCompressionEvent(
                agent=PARENT,
                summary_message_index=5,
                compressed_message_indices=[1, 2, 3, 4],
                active_context_indices=[0, 5, 6],
                before_tokens=4180,
                after_tokens=1432,
                strategy="tool-compact",
            ),
            TurnStartEvent(agent=PARENT),
            ModelRequestEvent(
                agent=PARENT,
                api=API,
                visible_count=5,
                llm_message_count=5,
                context_view={"input_tokens_estimate": 1432, "messages": 5},
                tools=[],
                llm_payload=[],
            ),
            ModelResponseEvent(
                agent=PARENT,
                api=API,
                output_kind="final",
                target="user",
                tool_call_count=0,
                usage=TokenUsage(
                    input_tokens=1812,
                    output_tokens=76,
                    cache_write_tokens=1812,
                ),
                model=MODEL,
            ),
            MessageEvent(
                message=AssistantMessage(
                    content=(
                        TextBlock(
                            "Fixed the off-by-one in wc.py; the focused test passes."
                        ),
                    ),
                    sender=PARENT,
                    target="user",
                    kind="final",
                    usage=TokenUsage(
                        input_tokens=1812,
                        output_tokens=76,
                        cache_write_tokens=1812,
                    ),
                    model=MODEL,
                )
            ),
            TurnEndEvent(agent=PARENT, terminated=True),
            AgentEndEvent(reason="done"),
        ]
    )


def _build_state() -> State:
    # Hand-built events bypass `record_event`, so stamp deterministic uuids here
    # (a real run gets random UUID4s) — keeps the golden stable while still
    # exercising the per-event `uuid` field.
    events = [
        dataclasses.replace(e, uuid=f"evt-{i}") for i, e in enumerate(_build_events())
    ]
    return State(task=TASK, events=events)


def build_stream() -> list[dict]:
    """The canonical v5 trajectory stream — a header line then one line per event
    — produced through the real serializers.

    Provider raw snapshots are kept INLINE (not externalized to a pool) so the
    embedded fixture is self-contained and the viewer's Wire panel works on it
    offline.
    """
    trace = run_trace_from_state(
        state=_build_state(), trace_id=TRACE_ID, producer=PRODUCER, meta=META
    )
    return [trace_header(trace), *[event_record(e) for e in trace.events]]


def _assembled(stream: list[dict]) -> dict:
    """Fold the stream (header + event lines) into the in-memory trace the viewer
    builds, matching the JS `assembleTrace`."""

    header, *events = stream
    return {**header, "events": events}


class TraceFixtureGoldenTest(unittest.TestCase):
    def test_sample_trace_matches_real_serializer(self) -> None:
        stream = build_stream()
        if os.environ.get("UPDATE_GOLDEN"):
            write_jsonl(SAMPLE_PATH, stream)
            self.skipTest(f"regenerated {SAMPLE_PATH.name}")
        self.assertEqual(
            read_jsonl(SAMPLE_PATH),
            stream,
            "sample-trace.jsonl drifted from the real trace serializer. If you "
            "renamed/restructured a serialized trace field, the viewer reads it too "
            "— update the viewer, then regenerate with UPDATE_GOLDEN=1 and review "
            "the diff.",
        )

    def test_embedded_sample_matches_real_serializer(self) -> None:
        # The viewer SHOWS the embedded copy (the file is an offline artifact),
        # so it is the one that actually goes stale. Guard it the same way.
        stream = build_stream()
        if os.environ.get("UPDATE_GOLDEN"):
            _write_embedded_sample(stream)
            self.skipTest("regenerated embedded-sample in index.html")
        self.assertEqual(
            _read_embedded_sample(),
            stream,
            "the embedded sample in index.html drifted from the real trace "
            "serializer — update the viewer for the schema change, then regenerate "
            "with UPDATE_GOLDEN=1 and review the diff.",
        )

    def test_file_and_embedded_samples_are_identical(self) -> None:
        # One source of truth: the offline file and the shown embedded copy must
        # be the same stream, so neither can drift independently.
        self.assertEqual(read_jsonl(SAMPLE_PATH), _read_embedded_sample())

    def test_fixture_exercises_every_viewer_read_path(self) -> None:
        # A generated fixture is only a guard if it contains the slots the viewer
        # reads. Pin those so a future trim can't quietly drop one and let the
        # golden pass while the viewer goes blank.
        record = _assembled(build_stream())
        msgs = [e["message"] for e in record["events"] if e["kind"] == "message"]
        sidecars = [m.get("sidecar") or {} for m in msgs]

        self.assertTrue(
            any(s.get("raw") for s in sidecars),
            "no message.sidecar.raw — the Wire debug panel would have nothing",
        )
        all_details = [s["details"] for s in sidecars if s.get("details")]
        self.assertTrue(all_details, "no message.sidecar.details — no sub-agent rows")
        self.assertTrue(
            any("sub_events" in v for d in all_details for v in d.values()),
            "no sub_events under details — sub-agent drill-down would be empty",
        )

        kinds = {e["kind"] for e in record["events"]}
        for required in (
            "agent_start",
            "model_request",
            "model_response",
            "tool_execution_start",
            "tool_execution_end",
            "context_compression",
            "agent_end",
        ):
            self.assertIn(required, kinds, f"fixture is missing a {required!r} event")
        self.assertTrue(
            any(
                e["kind"] == "tool_execution_end" and e.get("is_error")
                for e in record["events"]
            ),
            "no failing tool call — error rendering path unexercised",
        )

        # v5: the agents registry is DERIVED from the stream (agent_start /
        # compressor model_request carry system_prompt); the reader — and the
        # viewer's JS twin — must find a real prompt, and no event keeps the
        # reconstructable llm_payload.
        agents = collect_agents(record["events"])
        self.assertTrue(
            agents and all(isinstance(p, str) and p for p in agents.values()),
            "collect_agents found no real system prompt in the event stream",
        )
        self.assertFalse(
            any("llm_payload" in e for e in record["events"]),
            "an event still carries llm_payload — v5 drops it",
        )

        # Every event carries a stable, unique uuid (stamped on record_event).
        uuids = [e["uuid"] for e in record["events"]]
        self.assertTrue(all(uuids), "an event is missing its uuid")
        self.assertEqual(len(uuids), len(set(uuids)), "event uuids are not unique")

    def test_sample_is_loadable_v5_stream(self) -> None:
        # What the viewer's loader checks: a header line (schema, no events) then
        # one line per event, with an agent system prompt derivable from it.
        stream = read_jsonl(SAMPLE_PATH)
        header, events = stream[0], stream[1:]
        self.assertEqual(header["schema"], "simple-agent-lab.trajectory.v5")
        self.assertEqual(header["type"], "trajectory")
        self.assertNotIn("events", header)
        self.assertTrue(events and all("kind" in e for e in events))
        self.assertTrue(
            collect_agents(events),
            "no agent system prompt derivable from the stream — request "
            "reconstruction would have nothing real to show",
        )


if __name__ == "__main__":
    unittest.main()
