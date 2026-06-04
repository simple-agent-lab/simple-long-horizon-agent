"""Per-round (per-turn) model switching demo.

Shows how one agent can use a different model on each turn. The model is
chosen by a ``ProviderSelector`` — a ``(turn) -> Provider`` callable that
``make_llm_agent`` resolves before every model step (see ADR 0022). The
core loop and the ``generate(visible) -> Message`` contract are untouched;
only which provider gets called changes per round.

Examples::

    uv run python scripts/run_per_round_model_demo.py
    uv run python scripts/run_per_round_model_demo.py --policy cycle

By default the demo is deterministic: it uses the fake adapter, which echoes
``provider.model`` onto the response, so the served model is visible on each
turn. The default ``escalate`` policy runs turn 0 on a cheap model and every
later turn on a strong model; ``cycle`` alternates models each round.
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

# Pin a command so the fake takes two turns (tool call, then final answer).
DEMO_TASK = "Use bash to run command: `echo per-round`"


def build_selector(policy: str):
    """Return a `(turn) -> Provider` selector for the chosen policy.

    A plain list of models is just a one-line selector — that is the whole
    point of keeping the rotation policy in caller code rather than in the
    framework (ADR 0022).
    """
    models = [FAST, STRONG]
    if policy == "cycle":
        return lambda turn: models[turn % len(models)]
    # "escalate": cheap first turn, strong model from then on.
    return lambda turn: FAST if turn == 0 else STRONG


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-round model switching demo")
    parser.add_argument(
        "--policy",
        choices=["escalate", "cycle"],
        default="escalate",
        help="Per-turn model policy: 'escalate' (default) or 'cycle'.",
    )
    parser.add_argument("--no-trace", action="store_true")
    args = parser.parse_args()

    print(f"=== per-round model switching (policy={args.policy}) ===")
    agent = make_bash_agent(build_selector(args.policy), cwd=ROOT)
    state, events = agent.run(DEMO_TASK, max_turns=3)
    for _ in events:
        pass

    print("\n=== model used per turn ===")
    turn = 0
    for message in state.messages:
        if isinstance(message, AssistantMessage) and message.sender == "bash_agent":
            print(f"  turn {turn}: {message.model} ({message.kind})")
            turn += 1

    if not args.no_trace:
        print("\n=== full trace ===")
        print_trace(state)


if __name__ == "__main__":
    main()
