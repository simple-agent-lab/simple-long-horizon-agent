"""Per-round model switching demo.

Shows how one agent can use a different model on each round: pass
``make_llm_agent`` (or a preset like ``make_bash_agent``) a list of
providers instead of one, and round N uses the Nth model, with the last
model sticking once the list runs out. Here ``[FAST, STRONG]`` runs the
first round (the tool call) on a cheap model and the final answer on a
strong one. The core loop and the ``generate(visible) -> Message`` contract
are untouched; only which provider gets called changes per round.

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

from simple_agent_lab import AssistantMessage, print_trace  # noqa: E402
from simple_agent_lab.agents.bash import make_bash_agent  # noqa: E402
from simple_agent_lab.llm import Provider  # noqa: E402

# Two fake "models" so the switch is visible without any network call.
FAST = Provider(id="fast", api="fake", model="fast-model")
STRONG = Provider(id="strong", api="fake", model="strong-model")

# Pin a command so the fake takes two rounds (tool call, then final answer).
DEMO_TASK = "Use bash to run command: `echo per-round`"


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-round model switching demo")
    parser.add_argument("--no-trace", action="store_true")
    args = parser.parse_args()

    print("=== per-round model switching: provider=[FAST, STRONG] ===")
    agent = make_bash_agent([FAST, STRONG], cwd=ROOT)
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
