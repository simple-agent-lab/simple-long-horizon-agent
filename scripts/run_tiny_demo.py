"""Run tiny multi-agent recipes on the same message runtime.

Examples:

    python3 scripts/run_tiny_demo.py --recipe debate
    python3 scripts/run_tiny_demo.py --recipe pipeline
    python3 scripts/run_tiny_demo.py --recipe parallel
    python3 scripts/run_tiny_demo.py --recipe all --last-messages 2

The agents are deterministic toy functions. Replace an agent's step function
with an LLM call to make the recipe live.
"""

from __future__ import annotations

import argparse
import sys

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_agent_lab import (
    Agent,
    Message,
    State,
    assistant_message,
    last_message,
    message_text,
    print_trace,
    run_to_completion,
    sequence,
)


DEFAULT_TASK = "Design a simple multi-agent lab for student research."


def run_recipe(
    agents: list[Agent],
    state: State,
    schedule: list[str],
    last_messages: int | None,
) -> State:
    return run_to_completion(
        {agent.name: agent for agent in agents},
        state,
        sequence(*schedule),
        last=last_messages,
    )


# -----------------------------------------------------------------------------
# Debate recipe


def proposer(agent: Agent, visible: list[Message], state: State) -> Message:
    task = last_message(visible, kind="task").content if visible else state.task
    return assistant_message(
        (
            f"For '{task}', start with one tiny message runtime. Agents read "
            "a context_view, emit one message per step, and leave typed events. "
            "Debate, pipeline, and workspace are recipes, not separate frameworks."
        ),
        sender=agent.name,
        target="all",
        kind="message",
    )


def critic(agent: Agent, visible: list[Message], state: State) -> Message:
    proposal = last_message(visible, sender="proposer")
    return assistant_message(
        (
            "The proposal is good. Keep context management as context_view(), "
            f"not a framework. Visible messages for critic={len(visible)}. "
            f"Proposal length={len(message_text(proposal))}."
        ),
        sender=agent.name,
        target="all",
        kind="critique",
    )


def judge(agent: Agent, visible: list[Message], state: State) -> Message:
    proposal = maybe_message_text(visible, sender="proposer")
    critique = maybe_message_text(visible, sender="critic")
    return assistant_message(
        (
            "Use Agent + Message + State + context_view() + run(). "
            "It is small enough to teach and still supports research into "
            "communication, staged handoff, and visibility policies. "
            f"Visible evidence: {proposal} | {critique}"
        ),
        sender=agent.name,
        target="user",
        kind="final",
    )


def run_debate(task: str, last_messages: int | None) -> State:
    agents = [
        Agent("proposer", "proposes a design", proposer),
        Agent("critic", "finds missing pieces", critic),
        Agent("judge", "chooses final answer", judge),
    ]
    state = State(task)
    state.send("task", "user", "proposer", task)
    return run_recipe(agents, state, ["proposer", "critic", "judge"], last_messages)


# -----------------------------------------------------------------------------
# Pipeline recipe


def researcher_a(agent: Agent, visible: list[Message], state: State) -> Message:
    return assistant_message(
        (
            "Message runtime makes communication explicit. "
            f"I saw {len(visible)} message(s)."
        ),
        sender=agent.name,
        target="synthesizer",
        kind="note",
    )


def researcher_b(agent: Agent, visible: list[Message], state: State) -> Message:
    return assistant_message(
        (
            "Recipes model pipeline, debate, voting, and workspace collaboration. "
            f"I saw {len(visible)} message(s)."
        ),
        sender=agent.name,
        target="synthesizer",
        kind="note",
    )


def synthesizer(agent: Agent, visible: list[Message], state: State) -> Message:
    notes = " ".join(message.content for message in visible if message.kind == "note")
    return assistant_message(
        f"Pipeline result: keep the runtime tiny; compose recipes on top. Notes: {notes}",
        sender=agent.name,
        target="user",
        kind="final",
    )


def run_pipeline(task: str, last_messages: int | None) -> State:
    agents = [
        Agent("researcher_a", "studies the core runtime", researcher_a),
        Agent("researcher_b", "studies recipes", researcher_b),
        Agent("synthesizer", "merges notes", synthesizer),
    ]
    state = State(task)
    state.send("task", "user", "researcher_a", task)
    state.send("task", "user", "researcher_b", task)
    return run_recipe(
        agents,
        state,
        ["researcher_a", "researcher_b", "synthesizer"],
        last_messages,
    )


# -----------------------------------------------------------------------------
# Parallel synthesis recipe, inspired by SWALM AgentTool


def angle_core(agent: Agent, visible: list[Message], state: State) -> Message:
    return assistant_message(
        "Core: Agent + Message + State + context_view() + run().",
        sender=agent.name,
        target="synthesizer",
        kind="finding",
    )


def angle_trace(agent: Agent, visible: list[Message], state: State) -> Message:
    return assistant_message(
        "Trace: state.events is the trace; no extra Trace class needed.",
        sender=agent.name,
        target="synthesizer",
        kind="finding",
    )


def angle_teaching(agent: Agent, visible: list[Message], state: State) -> Message:
    return assistant_message(
        "Teaching: recipes are easier to modify than frameworks.",
        sender=agent.name,
        target="synthesizer",
        kind="finding",
    )


def parallel_synthesizer(agent: Agent, visible: list[Message], state: State) -> Message:
    findings = " ".join(
        message.content for message in visible if message.kind == "finding"
    )
    return assistant_message(
        f"Parallel synthesis: {findings}",
        sender=agent.name,
        target="user",
        kind="final",
    )


def run_parallel(task: str, last_messages: int | None) -> State:
    agents = [
        Agent("angle_core", "explores core abstraction", angle_core),
        Agent("angle_trace", "explores observability", angle_trace),
        Agent("angle_teaching", "explores pedagogy", angle_teaching),
        Agent("synthesizer", "synthesizes findings", parallel_synthesizer),
    ]
    state = State(task)
    for target in ["angle_core", "angle_trace", "angle_teaching"]:
        state.send("task", "user", target, task)
    return run_recipe(
        agents,
        state,
        ["angle_core", "angle_trace", "angle_teaching", "synthesizer"],
        last_messages,
    )


RECIPES = {
    "debate": run_debate,
    "pipeline": run_pipeline,
    "parallel": run_parallel,
}


def maybe_message_text(messages: list[Message], *, sender: str) -> str:
    try:
        return message_text(last_message(messages, sender=sender))
    except LookupError:
        return f"<{sender} not visible>"


def print_result(name: str, state: State, show_trace: bool) -> None:
    final = last_message(state, kind="final")
    state.data["metrics"] = {
        "events": len(state.events),
        "final_step": len(state.events) - 1,
    }

    print(f"\n{name}")
    print("=" * len(name))
    print(final.content)
    print(f"\nmetrics: {state.data['metrics']}")
    if show_trace:
        print_trace(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny message-runtime multi-agent demo")
    parser.add_argument("--recipe", choices=["all", *RECIPES.keys()], default="all")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument(
        "--last-messages",
        type=int,
        default=None,
        help="Limit each agent context_view to the last N visible messages",
    )
    parser.add_argument("--no-trace", action="store_true")
    args = parser.parse_args()

    names = list(RECIPES) if args.recipe == "all" else [args.recipe]
    for name in names:
        state = RECIPES[name](args.task, args.last_messages)
        print_result(name, state, show_trace=not args.no_trace)


if __name__ == "__main__":
    main()
