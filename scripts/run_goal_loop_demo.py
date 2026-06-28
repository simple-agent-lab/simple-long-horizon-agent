"""Goal-loop demo — deterministic (fake provider, no network), exits 0.

Demonstrates the goal loop with INDEPENDENT verification: the agent declares
work done via the ``update_goal`` terminal tool, but completion is only
confirmed when an objective shell command (``command_verifier_check``) exits 0.
The human-readable trace shows the full loop: first run → verifier check →
optional continuation → verifier confirms → terminal status.

Run:

    uv run python scripts/run_goal_loop_demo.py

What the demo shows:

1. A fake-backed agent is wired with ``update_goal_tool()`` — the terminal tool
   the model calls to self-declare its status.
2. The ``check`` is ``default_check(verifier=command_verifier_check(...))``
   — the model MUST declare done *and* an independent shell command must exit 0.
   Neither alone is sufficient.
3. On turn 0, the agent outputs a "working…" step message (no update_goal call).
   The verifier runs regardless — it passes because the sentinel was created
   before the loop (this models a real scenario where the agent's first turn
   ACTUALLY creates the file via bash, but with a fake provider we set it up
   deterministically).
4. After the verifier passes, the loop exits ``complete``.

Independence is the point: if the verifier command had failed, the loop would
have continued even after the model called ``update_goal(complete)``.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_agent_lab.messages import AssistantMessage, TextBlock, ToolCallBlock  # noqa: E402
from simple_agent_lab.core import Agent  # noqa: E402
from simple_agent_lab.workflow import (  # noqa: E402
    GoalBudgets,
    run_goal_loop,
    command_verifier_check,
    default_check,
    update_goal_tool,
)


# ---------------------------------------------------------------------------
# Fake generate closure — deterministic, no network.
#
# Turn 0  (first run): emit a "working…" step message — the agent checks the
#         file system and does its work.  It does NOT call update_goal yet
#         because it hasn't verified the outcome independently.
# Turn 1+ (continuation): the agent has re-examined the state, declares done
#         via update_goal(status=complete).
# ---------------------------------------------------------------------------
_TURN: dict[str, int] = {"n": 0}


def _fake_generate(messages):  # type: ignore[no-untyped-def]
    """Fake generate: turn 0 is a step; turn 1+ calls update_goal."""
    turn = _TURN["n"]
    _TURN["n"] += 1
    if turn == 0:
        # First turn: agent announces it is working.
        return AssistantMessage(
            content=(TextBlock("Checking the workspace and doing the work…"),),
            sender="goal_agent",
            target="user",
            kind="final",
        )
    # Subsequent turns: agent declares done via the terminal tool.
    return AssistantMessage(
        content=(
            TextBlock("Verified — the sentinel file is present."),
            ToolCallBlock(
                id=f"call_ug_{turn}",
                name="update_goal",
                arguments={
                    "status": "complete",
                    "reason": "sentinel file confirmed present",
                },
            ),
        ),
        sender="goal_agent",
        target="goal_agent",
        kind="step",
    )


def main() -> None:
    # ------------------------------------------------------------------
    # Create the sentinel file in a temp directory.
    # In a real run the agent's bash tool would create this; with the fake
    # provider we set it up here so the verifier has something to confirm.
    # ------------------------------------------------------------------
    sentinel_dir = Path(tempfile.mkdtemp(prefix="goal_loop_demo_"))
    sentinel = sentinel_dir / "done.txt"
    sentinel.write_text("goal loop demo sentinel\n", encoding="utf-8")

    objective = (
        f"Verify that the sentinel file at {sentinel} exists and is non-empty. "
        "Declare done via update_goal once verified."
    )

    # ------------------------------------------------------------------
    # Build the agent.
    # Agent(...) accepts tools= for extra AgentTools; we inject update_goal_tool
    # so the fake generate closure can reference it by name and the real
    # dispatch loop will execute it (ToolResult.terminate=True stops the inner
    # ReAct loop after the tool call, which is what we want).
    # ------------------------------------------------------------------
    # Reset the turn counter so the demo is idempotent if called in a test.
    _TURN["n"] = 0
    agent = Agent(
        name="goal_agent", generate=_fake_generate, tools=(update_goal_tool(),)
    )

    # ------------------------------------------------------------------
    # The completion check: model-declared PLUS independent shell verifier.
    # The shell command is the authoritative gatekeeper; the model's own
    # update_goal call is a necessary but not sufficient condition.
    # ------------------------------------------------------------------
    verifier = command_verifier_check(f"test -f {sentinel} && test -s {sentinel}")
    check = default_check(verifier=verifier)

    print("=== goal loop demo (fake provider, independent shell verifier) ===")
    print(f"objective : {objective[:120]}{'...' if len(objective) > 120 else ''}")
    print(f"sentinel  : {sentinel}")
    print(
        "check     : default_check(verifier=command_verifier_check(test -f ... && test -s ...))"
    )
    print()

    result = run_goal_loop(
        agent,
        objective,
        check=check,
        budgets=GoalBudgets(max_turns=4, wall_clock_seconds=30.0),
    )

    print("--- per-turn steps ---")
    for i, step in enumerate(result.steps):
        preview = step.output[:200].replace("\n", " ")
        if len(step.output) > 200:
            preview += "…"
        print(f"  turn {i}: [{step.name}] {preview}")

    print()
    print(
        f"=== status: {result.status} "
        f"(turns={result.turns_used}, tokens={result.tokens_used}) ==="
    )

    # Clean up.
    sentinel.unlink(missing_ok=True)
    sentinel_dir.rmdir()

    if result.status != "complete":
        print(
            f"ERROR: expected status=complete, got {result.status!r}", file=sys.stderr
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
