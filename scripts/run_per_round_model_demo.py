"""Per-round model switching demo.

Shows how one agent can use a different model on each round: give the agent
a map of named models plus a ``choose_model`` function that names which model
serves each round. Here ``choose`` runs the first round (the tool call) on a
cheap model and escalates to a strong one afterwards — and it would also
escalate early if a round came back with a tool error (``ctx.last_failed``).
The core loop and the ``generate(visible) -> Message`` contract are untouched;
only which model gets called changes per round.

Example::

    uv run python scripts/run_per_round_model_demo.py

The demo is deterministic: it uses the fake adapter, which echoes
``provider.model`` onto the response, so the served model is visible on
each round.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_agent_lab import AssistantMessage, RoundContext, print_trace  # noqa: E402
from simple_agent_lab.agents.bash import make_bash_agent  # noqa: E402
from simple_agent_lab.llm import Provider  # noqa: E402

# Two fake "models" so the switch is visible without any network call.
MODELS = {
    "fast": Provider(id="fast", api="fake", model="fast-model"),
    "strong": Provider(id="strong", api="fake", model="strong-model"),
}

# Pin a command so the fake takes two rounds (tool call, then final answer).
DEMO_TASK = "Use bash to run command: `echo per-round`"


def choose(ctx: RoundContext) -> str:
    """Cheap to explore, strong to finish — or strong early if a round failed."""
    if ctx.last_failed:
        return "strong"
    return "fast" if ctx.round == 0 else "strong"


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-round model switching demo")
    parser.add_argument("--no-trace", action="store_true")
    args = parser.parse_args()

    print("=== per-round model switching: model map + choose_model ===")
    agent = make_bash_agent(MODELS, cwd=ROOT, choose_model=choose)
    state, events = agent.run(DEMO_TASK, max_turns=3)
    for _ in events:
        pass

    print("\n=== model used per round ===")
    answers = [
        message
        for message in state.messages
        if isinstance(message, AssistantMessage) and message.sender == "bash_agent"
    ]
    for round_index, message in enumerate(answers):
        print(f"  round {round_index}: {message.model} ({message.kind})")

    if not args.no_trace:
        print("\n=== full trace ===")
        print_trace(state)


if __name__ == "__main__":
    main()
