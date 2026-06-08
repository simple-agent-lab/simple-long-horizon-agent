from __future__ import annotations

import unittest
from typing import Any

from simple_agent_lab import (
    Agent,
    EventKind,
    HookContext,
    HookDecision,
    HookFiredEvent,
    HookPoint,
    Message,
    MessageEvent,
    State,
    TextBlock,
    ToolCallBlock,
    assistant_message,
    message_text,
    runtime_message,
)
from simple_agent_lab.hooks import HookMap
from simple_agent_lab.tools import (
    AbortFlag,
    AgentTool,
    ToolResult,
    ToolUpdateFn,
    text_result,
)


def _tool_caller(name: str, *, tool_name: str = "echo", arguments: dict[str, Any]):
    """A `generate` fn that calls one tool, then finalizes once it sees a result."""

    def generate(visible: list[Message]) -> Message:
        if any(message.kind == "tool_result" for message in visible):
            return assistant_message(
                "final ok", sender=name, target="user", kind="final"
            )
        return assistant_message(
            [TextBlock("calling"), ToolCallBlock("call_1", tool_name, arguments)],
            sender=name,
            target=name,
            kind="step",
        )

    return generate


class _RecordingTool:
    """A fake tool that records the args each call received."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def execute(
        self,
        call_id: str,
        args: dict[str, Any],
        abort: AbortFlag,
        on_update: ToolUpdateFn | None,
    ) -> ToolResult:
        del call_id, abort, on_update
        self.calls.append(dict(args))
        return text_result(f"ran:{args.get('text')}")


def _echo_agent(
    name: str,
    recorder: _RecordingTool,
    *,
    arguments: dict[str, Any],
    hooks: HookMap | None = None,
) -> Agent:
    tool = AgentTool(
        name="echo",
        description="Echo text.",
        parameters={"type": "object"},
        execute=recorder.execute,
        execution_mode="sequential",
    )
    return Agent(
        name,
        _tool_caller(name, arguments=arguments),
        tools=(tool,),
        hooks=hooks or {},
    )


def _hook_events(state: State, point: HookPoint) -> list[HookFiredEvent]:
    return [
        event
        for event in state.events
        if isinstance(event, HookFiredEvent) and event.point == str(point)
    ]


def _texts(messages: list[Message]) -> list[str]:
    return [message_text(message) for message in messages]


class HookTest(unittest.TestCase):
    def test_observe_only_hook_runs_and_does_not_change_result(self) -> None:
        seen: list[str] = []

        def observer(ctx: HookContext) -> HookDecision | None:
            seen.append(ctx.tool_call.name if ctx.tool_call else "")
            return None

        recorder = _RecordingTool()
        agent = _echo_agent(
            "caller",
            recorder,
            arguments={"text": "hi"},
            hooks={HookPoint.PRE_TOOL_USE: [observer]},
        )

        state, events = agent.run("go", max_turns=3)
        for _ in events:
            pass

        self.assertEqual(seen, ["echo"])
        self.assertEqual(recorder.calls, [{"text": "hi"}])
        fired = _hook_events(state, HookPoint.PRE_TOOL_USE)
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0].block_reason, "")
        self.assertEqual(fired[0].emitted, 0)
        self.assertEqual(fired[0].target, "echo")

    def test_pre_tool_use_block_skips_execution(self) -> None:
        def deny(ctx: HookContext) -> HookDecision | None:
            del ctx
            return HookDecision(block_reason="nope")

        recorder = _RecordingTool()
        agent = _echo_agent(
            "caller",
            recorder,
            arguments={"text": "hi"},
            hooks={HookPoint.PRE_TOOL_USE: [deny]},
        )

        state, events = agent.run("go", max_turns=3)
        for _ in events:
            pass

        # Tool never ran.
        self.assertEqual(recorder.calls, [])
        # The denied call comes back to the model as an error tool result.
        bundle = next(
            message
            for message in reversed(state.messages)
            if message.kind == "tool_result"
        )
        self.assertIn("nope", message_text(bundle))
        self.assertTrue(bundle.content[0].is_error)
        # The block is recorded, and the run still completes.
        fired = _hook_events(state, HookPoint.PRE_TOOL_USE)
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0].block_reason, "nope")
        self.assertEqual(state.events[-1].kind, EventKind.AGENT_END)

    def test_session_start_hook_emits_a_message(self) -> None:
        def seed(ctx: HookContext) -> HookDecision | None:
            del ctx
            return HookDecision(
                emit_messages=(
                    runtime_message(
                        "seeded context", sender="hook", target="w", kind="context"
                    ),
                )
            )

        def writer(visible: list[Message]) -> Message:
            del visible
            return assistant_message("done", sender="w", target="user", kind="final")

        agent = Agent("w", writer, hooks={HookPoint.SESSION_START: [seed]})
        state, events = agent.run("go")
        yielded = list(events)

        # The emitted message is appended to the transcript...
        self.assertIn("seeded context", _texts(state.messages))
        # ...and it rode the yield stream as a real MessageEvent (no drift
        # between yielded events and state.events).
        self.assertTrue(
            any(
                isinstance(event, MessageEvent)
                and message_text(event.message) == "seeded context"
                for event in yielded
            )
        )
        fired = _hook_events(state, HookPoint.SESSION_START)
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0].emitted, 1)

    def test_pre_tool_use_emit_is_ignored(self) -> None:
        # Emitting mid-dispatch would orphan the tool_call/result pair, so the
        # gate ignores emit_messages at PRE_TOOL_USE.
        def emitter(ctx: HookContext) -> HookDecision | None:
            del ctx
            return HookDecision(
                emit_messages=(
                    runtime_message(
                        "should not appear",
                        sender="hook",
                        target="caller",
                        kind="context",
                    ),
                )
            )

        recorder = _RecordingTool()
        agent = _echo_agent(
            "caller",
            recorder,
            arguments={"text": "hi"},
            hooks={HookPoint.PRE_TOOL_USE: [emitter]},
        )
        state, events = agent.run("go", max_turns=3)
        for _ in events:
            pass

        self.assertNotIn("should not appear", _texts(state.messages))
        # Emit-only (no block) leaves the call to run normally.
        self.assertEqual(recorder.calls, [{"text": "hi"}])
        self.assertEqual(_hook_events(state, HookPoint.PRE_TOOL_USE)[0].emitted, 0)

    def test_session_hooks_bracket_the_run(self) -> None:
        def observer(ctx: HookContext) -> HookDecision | None:
            del ctx
            return None

        def writer(visible: list[Message]) -> Message:
            del visible
            return assistant_message("done", sender="w", target="user", kind="final")

        hooks: HookMap = {
            HookPoint.SESSION_START: [observer],
            HookPoint.SESSION_END: [observer],
        }
        state, events = Agent("w", writer, hooks=hooks).run("go")
        for _ in events:
            pass

        self.assertEqual(len(_hook_events(state, HookPoint.SESSION_START)), 1)
        self.assertEqual(len(_hook_events(state, HookPoint.SESSION_END)), 1)
        kinds = [event.kind for event in state.events]
        # SESSION_START fires immediately after agent_start, before any turn.
        start_at = kinds.index(EventKind.AGENT_START)
        self.assertEqual(kinds[start_at + 1], EventKind.HOOK_FIRED)
        self.assertNotIn(EventKind.TURN_START, kinds[: start_at + 1])
        # SESSION_END fires immediately before agent_end.
        self.assertEqual(kinds[-1], EventKind.AGENT_END)
        self.assertEqual(kinds[-2], EventKind.HOOK_FIRED)

    def test_first_block_short_circuits_later_hooks(self) -> None:
        order: list[str] = []

        def first(ctx: HookContext) -> HookDecision | None:
            del ctx
            order.append("first")
            return HookDecision(block_reason="stop")

        def second(ctx: HookContext) -> HookDecision | None:
            del ctx
            order.append("second")
            return None

        recorder = _RecordingTool()
        agent = _echo_agent(
            "caller",
            recorder,
            arguments={"text": "hi"},
            hooks={HookPoint.PRE_TOOL_USE: [first, second]},
        )
        _, events = agent.run("go", max_turns=3)
        for _ in events:
            pass

        # The veto is terminal: the second hook never runs.
        self.assertEqual(order, ["first"])
        self.assertEqual(recorder.calls, [])

    def test_run_without_hooks_records_no_hook_events(self) -> None:
        recorder = _RecordingTool()
        agent = _echo_agent("caller", recorder, arguments={"text": "hi"})

        state, events = agent.run("go", max_turns=3)
        for _ in events:
            pass

        self.assertEqual(recorder.calls, [{"text": "hi"}])
        self.assertFalse(
            any(isinstance(event, HookFiredEvent) for event in state.events)
        )


if __name__ == "__main__":
    unittest.main()
