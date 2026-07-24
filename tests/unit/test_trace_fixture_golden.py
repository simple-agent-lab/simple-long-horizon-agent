"""Guard against trace-producer / trace-viewer schema drift.

The viewer sample is generated from real message and event dataclasses through
the production serializer. Serialized field changes therefore fail this golden
test until the viewer and fixture are intentionally updated together.

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
# The viewer renders this embedded copy instead of fetching the JSONL fixture,
# so both copies must stay in lockstep with the serializer.
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

# OpenAI-shaped wire data exercises `message.sidecar.raw` and reasoning fields.
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


def _usage(
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> TokenUsage:
    return TokenUsage(
        input_tokens, output_tokens, cache_read_tokens, cache_write_tokens
    )


def _user(
    *content: TextBlock | ToolResultBlock,
    sender: str,
    target: str,
    kind: str,
    sidecar: dict | None = None,
) -> MessageEvent:
    kwargs = dict(content=content, sender=sender, target=target, kind=kind)
    if sidecar is not None:
        kwargs["sidecar"] = sidecar
    return MessageEvent(message=UserMessage(**kwargs))


def _model_output(
    *,
    agent: str,
    kind: str,
    target: str,
    tool_calls: int,
    usage: TokenUsage,
    content: tuple[TextBlock | ThinkingBlock | ToolCallBlock, ...],
    include_message_usage: bool = False,
    sidecar: dict | None = None,
) -> list[Event]:
    message_kwargs = {
        "content": content,
        "sender": agent,
        "target": target,
        "kind": kind,
        **({"usage": usage, "model": MODEL} if include_message_usage else {}),
        **({"sidecar": sidecar} if sidecar is not None else {}),
    }
    return [
        ModelResponseEvent(
            agent=agent,
            api=API,
            output_kind=kind,
            target=target,
            tool_call_count=tool_calls,
            usage=usage,
            model=MODEL,
        ),
        MessageEvent(message=AssistantMessage(**message_kwargs)),
    ]


def _request(
    agent: str,
    visible: int,
    messages: int,
    estimate: int,
    *,
    tools: list[dict] | None = None,
    payload: list[dict] | None = None,
    include_message_count: bool = True,
) -> ModelRequestEvent:
    context = {
        "input_tokens_estimate": estimate,
        **({"messages": messages} if include_message_count else {}),
    }
    return ModelRequestEvent(
        agent=agent,
        api=API,
        visible_count=visible,
        llm_message_count=messages,
        context_view=context,
        tools=tools or [],
        llm_payload=payload or [],
    )


def _tool_exchange(
    call_id: str,
    tool_name: str,
    text: str,
    *,
    target: str,
    is_error: bool = False,
    sidecar: dict | None = None,
) -> list[Event]:
    result = ToolResultBlock(
        tool_call_id=call_id,
        tool_name=tool_name,
        content=(TextBlock(text),),
        is_error=is_error,
    )
    return [
        ToolExecutionStartEvent(tool_call_id=call_id, tool_name=tool_name),
        ToolExecutionEndEvent(
            tool_call_id=call_id,
            tool_name=tool_name,
            is_error=is_error,
            terminate=False,
        ),
        _user(
            result,
            sender="tool",
            target=target,
            kind="tool_result",
            sidecar=sidecar,
        ),
    ]


def _sub_events() -> list[Event]:
    """The search sub-agent's own trace, nested under the task tool result."""
    thought_usage = _usage(188, 40)
    return _stamp(
        [
            _user(
                TextBlock("Search the repo for wc_lines callers."),
                sender=PARENT,
                target=SUB,
                kind="task",
            ),
            AgentStartEvent(agent=SUB, system_prompt=f"You are {SUB}."),
            TurnStartEvent(agent=SUB),
            _request(
                SUB,
                2,
                2,
                188,
                tools=[{"name": "bash", "description": "Run a bash command."}],
                include_message_count=False,
            ),
            *_model_output(
                agent=SUB,
                kind="thought",
                target=SUB,
                tool_calls=1,
                usage=thought_usage,
                content=(
                    TextBlock("Running ripgrep."),
                    ToolCallBlock(
                        id="sub_call_01",
                        name="bash",
                        arguments={"command": "rg -n wc_lines"},
                    ),
                ),
            ),
            *_tool_exchange(
                "sub_call_01",
                "bash",
                "src/simple_agent_lab/tools/wc.py:1:def wc_lines",
                target=SUB,
            ),
            TurnEndEvent(agent=SUB),
            *_model_output(
                agent=SUB,
                kind="final",
                target=PARENT,
                tool_calls=0,
                usage=_usage(264, 52),
                content=(TextBlock("wc_lines: 1 caller, 1 test. Low blast radius."),),
            ),
            AgentEndEvent(reason="final"),
        ]
    )


def _build_events() -> list[Event]:
    """Hand-built but via REAL dataclasses, so the serialized shape is coupled to
    the runtime schema; covers every slot the viewer reads."""
    turn_1_usage = _usage(612, 88, cache_write_tokens=612)
    turn_2_usage = _usage(894, 120, 612, 282)
    turn_3_usage = _usage(1842, 96, 894, 948)
    final_usage = _usage(1812, 76, cache_write_tokens=1812)
    return _stamp(
        [
            _user(TextBlock(TASK), sender="user", target=PARENT, kind="task"),
            AgentStartEvent(agent=PARENT, system_prompt=f"You are {PARENT}."),
            # Turn 1 — thinking + bash (ok); carries the wire-debug raw blob.
            TurnStartEvent(agent=PARENT),
            _request(
                PARENT,
                1,
                2,
                612,
                tools=[
                    {"name": "bash", "description": "Run a bash command."},
                    {"name": "task", "description": "Delegate a focused subtask."},
                ],
                payload=[
                    {"role": "system", "content": "You are obs_agent."},
                    {"role": "user", "content": TASK},
                ],
            ),
            *_model_output(
                agent=PARENT,
                kind="thought",
                target=PARENT,
                tool_calls=1,
                usage=turn_1_usage,
                content=(
                    ThinkingBlock(text="Inspect the layout before editing."),
                    TextBlock("Looking at the repo layout."),
                    ToolCallBlock(
                        id="call_01",
                        name="bash",
                        arguments={"command": "ls src/simple_agent_lab"},
                    ),
                ),
                include_message_usage=True,
                sidecar={"raw": RAW_BLOB},
            ),
            *_tool_exchange(
                "call_01",
                "bash",
                "core.py\nmessages.py\nstate.py\ntools/",
                target=PARENT,
            ),
            TurnEndEvent(agent=PARENT),
            # Turn 2 — delegate to a sub-agent; result carries nested sub_events.
            TurnStartEvent(agent=PARENT),
            _request(PARENT, 3, 4, 894),
            *_model_output(
                agent=PARENT,
                kind="thought",
                target=PARENT,
                tool_calls=1,
                usage=turn_2_usage,
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
                include_message_usage=True,
            ),
            *_tool_exchange(
                "call_02_task",
                "task",
                "Sub-agent: wc_lines has 1 caller, 1 test.",
                target=PARENT,
                sidecar={"details": {"call_02_task": {"sub_events": _sub_events()}}},
            ),
            TurnEndEvent(agent=PARENT),
            # Turn 3 — a failing tool call, then a compression, then the final.
            TurnStartEvent(agent=PARENT),
            _request(PARENT, 5, 6, 1842),
            *_model_output(
                agent=PARENT,
                kind="thought",
                target=PARENT,
                tool_calls=1,
                usage=turn_3_usage,
                content=(
                    TextBlock("Applying the fix."),
                    ToolCallBlock(
                        id="call_03",
                        name="bash",
                        arguments={"command": "patch -p0 < /tmp/missing.patch"},
                    ),
                ),
                include_message_usage=True,
            ),
            *_tool_exchange(
                "call_03",
                "bash",
                "patch: /tmp/missing.patch: No such file",
                target=PARENT,
                is_error=True,
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
            _request(PARENT, 5, 5, 1432),
            *_model_output(
                agent=PARENT,
                kind="final",
                target="user",
                tool_calls=0,
                usage=final_usage,
                content=(
                    TextBlock(
                        "Fixed the off-by-one in wc.py; the focused test passes."
                    ),
                ),
                include_message_usage=True,
            ),
            TurnEndEvent(agent=PARENT, terminated=True),
            AgentEndEvent(reason="done"),
        ]
    )


def _build_state() -> State:
    # Hand-built events need deterministic UUIDs in place of record_event's UUID4s.
    events = [
        dataclasses.replace(e, uuid=f"evt-{i}") for i, e in enumerate(_build_events())
    ]
    return State(task=TASK, events=events)


def build_stream() -> list[dict]:
    """Build the canonical v5 header/event stream through real serializers."""
    trace = run_trace_from_state(
        state=_build_state(), trace_id=TRACE_ID, producer=PRODUCER, meta=META
    )
    return [trace_header(trace), *[event_record(e) for e in trace.events]]


class TraceFixtureGoldenTest(unittest.TestCase):
    def test_samples_match_real_serializer(self) -> None:
        stream = build_stream()
        samples = {
            SAMPLE_PATH.name: (
                lambda: read_jsonl(SAMPLE_PATH),
                lambda: write_jsonl(SAMPLE_PATH, stream),
            ),
            "embedded-sample": (
                _read_embedded_sample,
                lambda: _write_embedded_sample(stream),
            ),
        }
        update = os.environ.get("UPDATE_GOLDEN")
        for name, (read, write) in samples.items():
            with self.subTest(sample=name):
                if update:
                    write()
                else:
                    self.assertEqual(
                        read(),
                        stream,
                        f"{name} drifted from the real trace serializer; update the "
                        "viewer for intentional schema changes, regenerate with "
                        "UPDATE_GOLDEN=1, and review the diff.",
                    )
        if update:
            self.skipTest(f"regenerated {', '.join(samples)}")

    def test_file_and_embedded_samples_are_identical(self) -> None:
        self.assertEqual(read_jsonl(SAMPLE_PATH), _read_embedded_sample())

    def test_fixture_exercises_every_viewer_read_path(self) -> None:
        header, *events = build_stream()
        record = {**header, "events": events}
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

        uuids = [e["uuid"] for e in record["events"]]
        self.assertTrue(all(uuids), "an event is missing its uuid")
        self.assertEqual(len(uuids), len(set(uuids)), "event uuids are not unique")

    def test_sample_is_loadable_v5_stream(self) -> None:
        stream = read_jsonl(SAMPLE_PATH)
        header, events = stream[0], stream[1:]
        for field, expected in {
            "schema": "simple-agent-lab.trajectory.v5",
            "type": "trajectory",
        }.items():
            with self.subTest(field=field):
                self.assertEqual(header[field], expected)
        self.assertNotIn("events", header)
        self.assertTrue(events and all("kind" in e for e in events))
        self.assertTrue(
            collect_agents(events),
            "no agent system prompt derivable from the stream — request "
            "reconstruction would have nothing real to show",
        )
