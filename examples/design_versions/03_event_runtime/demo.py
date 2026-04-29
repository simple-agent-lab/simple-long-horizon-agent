"""Event-sourced agent-loop demo.

The default path uses FakeModel and does not call a network service.

Run:

    python3 examples/design_versions/03_event_runtime/demo.py
"""

from __future__ import annotations

import argparse

from core import Agent, AgentLoop, RunConfig, RuntimeState, print_trace
from models import ModelConfig, load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["fake", "openai"], default="fake")
    parser.add_argument("--model", default="fake-model")
    args = parser.parse_args()

    agent = Agent("assistant", "You are a tiny teaching agent.")
    model = load_model(ModelConfig(provider=args.provider, model=args.model))
    state = RuntimeState("Design an event-sourced agent loop.")
    state.user(agent.name, state.task)

    loop = AgentLoop(RunConfig(max_steps=4, meta="header"))
    result = loop.run(agent, state, model)
    print_trace(result.state)
    print(f"\nstop_reason={result.stop_reason}, steps={result.steps}")


if __name__ == "__main__":
    main()
