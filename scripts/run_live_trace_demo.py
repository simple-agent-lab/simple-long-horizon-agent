"""Stream a synthetic agent trajectory to disk so the trace viewer can tail it.

This script does NOT call any LLM provider. It just hand-builds events on a
``State`` and hands the state to :class:`IncrementalTraceWriter`. The
writer's background thread atomically rewrites ``trajectory.jsonl`` every
~1.5s, which lets you exercise the viewer's live-refresh + LIVE indicator
path without standing up a real bench eval.

Usage::

    uv run python scripts/run_live_trace_demo.py
    uv run python scripts/run_live_trace_demo.py \\
        --out evals/out/_live_demo/trajectory.jsonl \\
        --turns 8 --turn-delay 2.5

Then point the viewer at the file::

    open "http://127.0.0.1:8765/?load=$(pwd)/evals/out/_live_demo/trajectory.jsonl"

A real run-end is simulated by emitting an ``agent_end`` event and stopping
the writer; the file's mtime stops moving after that and the viewer's LIVE
indicator fades off ~10s later.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_agent_lab.messages import (  # noqa: E402
    AssistantMessage,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    make_message,
)
from simple_agent_lab.protocols import (  # noqa: E402
    AgentEndEvent,
    AgentStartEvent,
    ModelRequestEvent,
    ModelResponseEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from simple_agent_lab.state import State  # noqa: E402
from simple_agent_lab.trace import (  # noqa: E402
    LiveTraceSession,
    TraceMeta,
    default_stderr_flush_error,
    write_canonical_trace,
)

DEFAULT_OUT = ROOT / "evals" / "out" / "_live_demo" / "trajectory.jsonl"

THINK_SAMPLES = [
    "Inspecting the repo layout before touching files.",
    "Sub-agent will grep for callers while I read the suspect module.",
    "The off-by-one explains the failing test — patching now.",
    "Re-running the focused test to confirm the fix sticks.",
    "Tightening the diff and writing a concise final summary.",
]
TEXT_SAMPLES = [
    "Looking at the repo layout.",
    "Reading the suspect file and dispatching a search sub-agent in parallel.",
    "Applying fix.",
    "Writing the fix inline and rerunning the test.",
    "Verifying the fix passes the failing test.",
]
TOOL_RESULT_SAMPLES = [
    "agents/\ncore.py\ncontext_view.py\nllm/\nmessages.py\nprotocols.py\nstate.py\ntools/\ntrace/",
    "def wc_lines(path):\n    with open(path) as f:\n        return len(f.read().split('\\n')) - 1",
    "src/simple_agent_lab/tools/wc.py:1:def wc_lines(path):\ntests/unit/test_wc.py:8:    assert wc_lines('fixtures/3lines.txt') == 3",
    "============= 1 passed in 0.34s ==============",
    "Fix applied and verified.",
]


def _emit_turn(state: State, turn_index: int, agent: str, *, turn_delay: float) -> None:
    """Append one full turn's worth of events to ``state`` with realistic pacing."""

    state.record_event(TurnStartEvent(agent=agent))
    time.sleep(max(0.05, turn_delay * 0.05))

    state.record_event(
        ModelRequestEvent(
            agent=agent,
            visible_count=turn_index * 2 + 1,
            llm_message_count=turn_index * 2 + 1,
            context_view={
                "input_tokens_estimate": 600 + 220 * turn_index,
                "messages": turn_index * 2 + 1,
            },
            tools=[{"name": "bash", "description": "Run a bash command."}],
            llm_payload=[],
        )
    )
    # Simulate LLM latency.
    time.sleep(max(0.1, turn_delay * 0.35))

    call_id = f"call_{turn_index:02d}"
    state.record_event(
        ModelResponseEvent(
            agent=agent,
            output_kind="step",
            target=agent,
            tool_call_count=1,
        )
    )
    assistant_msg = AssistantMessage(
        sender=agent,
        target=agent,
        kind="step",
        content=(
            TextBlock(text=random.choice(TEXT_SAMPLES)),
            ToolCallBlock(
                id=call_id,
                name="bash",
                arguments={"command": f"echo 'turn {turn_index}'"},
            ),
        ),
    )
    state.record(assistant_msg)

    state.record_event(ToolExecutionStartEvent(tool_call_id=call_id, tool_name="bash"))
    # This is the long stretch — the agent loop is now blocked inside the
    # tool call. The background writer should keep flushing during this gap.
    time.sleep(max(0.1, turn_delay * 0.45))
    state.record_event(
        ToolExecutionEndEvent(
            tool_call_id=call_id,
            tool_name="bash",
            is_error=False,
            terminate=False,
        )
    )

    tool_result = make_message(
        role="user",
        content=(
            ToolResultBlock(
                tool_call_id=call_id,
                tool_name="bash",
                content=(TextBlock(text=random.choice(TOOL_RESULT_SAMPLES)),),
                is_error=False,
            ),
        ),
        sender="tool",
        target=agent,
        kind="tool_result",
    )
    state.record(tool_result)
    state.record_event(TurnEndEvent(agent=agent, terminated=False))
    time.sleep(max(0.05, turn_delay * 0.15))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument(
        "--turn-delay",
        type=float,
        default=2.0,
        help="Approximate seconds spent per simulated agent turn (default: 2.0)",
    )
    parser.add_argument(
        "--flush-interval",
        type=float,
        default=1.5,
        help="Seconds between background trace flushes (default: 1.5)",
    )
    parser.add_argument(
        "--linger-s",
        type=float,
        default=0.0,
        help="Wait this long with no events at the end of the run (default: 0)",
    )
    parser.add_argument("--agent", default="bash_agent")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Wipe any leftover file from a previous run so the viewer's mtime
    # comparison sees a fresh start.
    if out_path.exists():
        out_path.unlink()

    state = State(task="Fix the failing wc-line regression and ship a tiny test.")
    # Initialize the conversation with the original task as a user message so
    # the viewer's "task" preview lines up with what the agent actually sees.
    state.send("task", "user", args.agent, state.task)
    state.record_event(AgentStartEvent())

    trace_meta = TraceMeta(
        trace_id="demo.live.001",
        producer="demo:live-trace",
        meta_fn=lambda: {
            "demo": "live-trace",
            "events_so_far": len(state.events),
            "in_progress": True,
        },
    )
    print(f"[live-demo] writing → {out_path}", flush=True)
    print(
        f"[live-demo] flushing every {args.flush_interval:g}s for "
        f"{args.turns} turn(s) at ~{args.turn_delay:g}s each",
        flush=True,
    )

    with LiveTraceSession(
        out_path,
        state,
        trace_id=trace_meta.trace_id,
        producer=trace_meta.producer,
        meta_fn=trace_meta.meta_fn,
        flush_interval_s=args.flush_interval,
        on_error=default_stderr_flush_error,
    ):
        for turn in range(1, args.turns + 1):
            print(f"[live-demo] turn {turn}/{args.turns} starting", flush=True)
            _emit_turn(state, turn, args.agent, turn_delay=args.turn_delay)
        # A short pause with no new events lets you see the LIVE pill
        # stay on for a moment before fading.
        if args.linger_s > 0:
            print(
                f"[live-demo] lingering {args.linger_s:g}s with no events", flush=True
            )
            time.sleep(args.linger_s)
        # Mark final message + agent end so the viewer's stat strip shows a
        # clean "done" exit reason once the run finishes.
        state.record(
            AssistantMessage(
                sender=args.agent,
                target="user",
                kind="final",
                content=(
                    TextBlock(
                        text=(
                            "Fixed the off-by-one in src/simple_agent_lab/tools/wc.py "
                            "and confirmed the focused test passes. Suggest adding a "
                            "fixture without a trailing newline before shipping."
                        ),
                    ),
                ),
            )
        )
        state.record_event(AgentEndEvent(reason="done"))

    meta_fn = trace_meta.meta_fn

    def final_meta() -> dict[str, Any]:
        base_meta = meta_fn() if meta_fn is not None else {}
        return {**dict(base_meta or {}), "in_progress": False}

    write_canonical_trace(
        out_path,
        state=state,
        trace_meta=TraceMeta(
            trace_id=trace_meta.trace_id,
            producer=trace_meta.producer,
            meta_fn=final_meta,
        ),
    )

    print(
        f"[live-demo] done after {len(state.events)} events — "
        "the LIVE indicator should fade off in ~10s",
        flush=True,
    )


if __name__ == "__main__":
    main()
