"""Prompt evolution: the candidate payload is an agent's system prompt.

The evolution loop's evaluator builds an `Agent` from each candidate's
`system_prompt`, runs it over a small arithmetic task list, and scores
format-exact answers (1.0 exact, 0.5 right-number-but-verbose, 0.0 wrong).
The proposer is an LLM rewriting the prompt using the evaluator's feedback.

Offline default: a deterministic stand-in model that answers correctly but
verbosely UNLESS the system prompt tells it to reply with only the number,
and a canned proposer that gradually discovers that instruction. The point
is to show the seams — swap `--live` in to use the real provider env for
both the agent under test and the proposer.

Run:
    uv run python scripts/run_prompt_evolution_demo.py
    uv run python scripts/run_prompt_evolution_demo.py --live
"""

from __future__ import annotations

import argparse
import random
import re
from typing import Sequence

from simple_agent_lab.core import Agent
from simple_agent_lab.evolve import (
    AskFn,
    Candidate,
    EvolutionBudgets,
    EvolutionRecord,
    accept_correct,
    agent_task_evaluator,
    ask_from_provider,
    llm_propose,
    parse_fields,
    render_fields,
    run_evolution,
    select_weighted,
)
from simple_agent_lab.messages import AssistantMessage, TextBlock

# (question, expected exact reply)
TASKS = [
    ("What is 17 + 25?", "42"),
    ("What is 9 * 8?", "72"),
    ("What is 100 - 37?", "63"),
    ("What is 144 / 12?", "12"),
]
ANSWERS = dict(TASKS)
SEED_PROMPT = "You are a helpful assistant."


def score(task: str, output: str) -> float:
    expected = ANSWERS[task]
    if output.strip() == expected:
        return 1.0
    if re.search(rf"\b{re.escape(expected)}\b", output):
        return 0.5
    return 0.0


def build_fake_agent(candidate: Candidate) -> Agent:
    """Deterministic stand-in for a model that follows its system prompt.

    Always computes the right number; replies with ONLY the number when the
    prompt asks for that, otherwise wraps it in a sentence (scoring 0.5).
    """

    prompt = str(candidate.payload["system_prompt"]).lower()
    terse = "only the number" in prompt or "nothing else" in prompt

    def generate(visible) -> AssistantMessage:
        question = next(m for m in reversed(visible) if m.role == "user")
        task = next(t for t in ANSWERS if t in str(question.content))
        answer = ANSWERS[task]
        text = answer if terse else f"The answer to your question is {answer}."
        return AssistantMessage(
            content=(TextBlock(text=text),),
            sender="candidate_agent",
            target="user",
            kind="final",
        )

    return Agent(name="candidate_agent", generate=generate)


def fake_ask(rng: random.Random) -> AskFn:
    """Canned proposer replies: plausible prompt rewrites, one per call."""

    rewrites = [
        "Answer arithmetic questions accurately and briefly.",
        "You are a calculator. Reply with only the number, nothing else.",
        "Solve the arithmetic problem. Reply with only the number.",
    ]

    def ask(prompt: str) -> str:
        current = parse_fields(prompt, ["system_prompt"])["system_prompt"]
        candidates = [r for r in rewrites if r != current] or rewrites
        rewrite = rng.choice(candidates)
        return (
            "Making the reply format explicit should raise the exact-match "
            "score.\n\n" + render_fields({"system_prompt": rewrite}, ["system_prompt"])
        )

    return ask


def build_live_agent_factory():
    from simple_agent_lab.llm import provider_from_env
    from simple_agent_lab.llm_agent import make_llm_agent

    provider = provider_from_env()

    def build_agent(candidate: Candidate) -> Agent:
        return make_llm_agent(
            name="candidate_agent",
            provider=provider,
            system_prompt=str(candidate.payload["system_prompt"]),
            target="user",
        )

    return build_agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="use the provider env")
    parser.add_argument("--budget", type=int, default=8, help="max archive size")
    parser.add_argument("--seed", type=int, default=0, help="rng seed")
    args = parser.parse_args()

    if args.live:
        from simple_agent_lab.llm import provider_from_env

        build_agent = build_live_agent_factory()
        ask = ask_from_provider(provider_from_env())
    else:
        build_agent = build_fake_agent
        ask = fake_ask(random.Random(args.seed))

    propose = llm_propose(
        ask,
        task=(
            "Rewrite this agent's system prompt so its answers to arithmetic "
            "questions exactly match the expected reply format."
        ),
        fields=["system_prompt"],
    )
    evaluate = agent_task_evaluator(
        build_agent, [t for t, _ in TASKS], score, max_turns=3
    )

    def show(record: EvolutionRecord) -> None:
        marker = "+" if record.accepted else "-"
        print(
            f"{marker} {record.candidate.id}  fitness {record.evaluation.fitness:.2f}  "
            f"prompt: {str(record.candidate.payload.get('system_prompt', ''))[:70]!r}"
        )

    result = run_evolution(
        seeds=[{"system_prompt": SEED_PROMPT}],
        propose=propose,
        evaluate=evaluate,
        select=select_weighted(power=2.0, inspirations=1),
        accept=accept_correct,
        budgets=EvolutionBudgets(max_candidates=args.budget, target_fitness=1.0),
        rng_seed=args.seed,
        on_record=show,
    )

    print(f"\nstatus: {result.status}")
    if result.best is not None:
        print(f"best prompt: {result.best.candidate.payload['system_prompt']!r}")
        print(f"fitness: {result.best.evaluation.fitness:.2f}")


if __name__ == "__main__":
    main()
