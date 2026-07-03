"""Agent-definition evolution using the typed component layer.

The candidate is a standard *agent genome* (`agent_genome`): a system
prompt plus an `instructions` block, each a declared `ComponentSpec` with
proposer-facing docs. `genome_propose` targets one component per mutation
and validates what comes back; `mix_operators` splits proposals between
mutation and LLM crossover; the evaluator builds a live `Agent` from each
candidate (`build_genome_agent` in live mode) and scores format-exact
arithmetic answers (1.0 exact, 0.5 right-number-but-verbose, 0.0 wrong).

Offline default: a deterministic stand-in model that answers correctly but
verbosely UNLESS the evolved prompt/instructions demand only the number,
and a canned proposer that gradually discovers that demand. The point is
the seams — `--live` swaps in the real provider env for both the agent
under test and the proposer.

Run:
    uv run python scripts/run_prompt_evolution_demo.py
    uv run python scripts/run_prompt_evolution_demo.py --live
"""

from __future__ import annotations

import argparse
import random
import re

from simple_agent_lab.core import Agent
from simple_agent_lab.evolve import (
    AskFn,
    Candidate,
    EvolutionBudgets,
    EvolutionRecord,
    accept_correct,
    agent_genome,
    agent_task_evaluator,
    ask_from_provider,
    build_genome_agent,
    crossover_propose,
    genome_propose,
    mix_operators,
    render_fields,
    run_evolution,
    seed_agent_payload,
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
GENOME = agent_genome(
    task=(
        "Make the agent's answers to arithmetic questions exactly match the "
        "expected reply format."
    )
)


def score(task: str, output: str) -> float:
    expected = ANSWERS[task]
    if output.strip() == expected:
        return 1.0
    if re.search(rf"\b{re.escape(expected)}\b", output):
        return 0.5
    return 0.0


def build_fake_agent(candidate: Candidate) -> Agent:
    """Deterministic stand-in for a model that follows its evolved genome.

    Always computes the right number; replies with ONLY the number when the
    prompt or instructions ask for that, otherwise wraps it in a sentence
    (scoring 0.5).
    """

    genome_text = " ".join(str(v) for v in candidate.payload.values()).lower()
    terse = "only the number" in genome_text or "nothing else" in genome_text

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
    """Canned proposer: rewrites whichever component the prompt targeted."""

    prompt_rewrites = [
        "Answer arithmetic questions accurately and briefly.",
        "You are a calculator. Reply with only the number, nothing else.",
    ]
    instruction_rewrites = [
        "Do not show your reasoning.",
        "Output format: only the number, no punctuation, nothing else.",
    ]

    def ask(prompt: str) -> str:
        if "### system_prompt" in prompt:
            field, rewrite = "system_prompt", rng.choice(prompt_rewrites)
        else:
            field, rewrite = "instructions", rng.choice(instruction_rewrites)
        return (
            "Making the reply format explicit should raise the exact-match "
            "score.\n\n" + render_fields({field: rewrite}, [field])
        )

    return ask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="use the provider env")
    parser.add_argument("--budget", type=int, default=10, help="max archive size")
    parser.add_argument("--seed", type=int, default=0, help="rng seed")
    args = parser.parse_args()

    if args.live:
        from simple_agent_lab.llm import provider_from_env

        provider = provider_from_env()
        ask = ask_from_provider(provider)

        def build_agent(candidate: Candidate) -> Agent:
            return build_genome_agent(candidate, provider=provider)
    else:
        build_agent = build_fake_agent
        ask = fake_ask(random.Random(args.seed))

    # 80% single-component mutation, 20% crossover of parent + inspiration.
    propose = mix_operators(
        [
            (genome_propose(ask, GENOME), 0.8),
            (
                crossover_propose(
                    ask, task=GENOME.task, fields=list(GENOME.mutable_keys())
                ),
                0.2,
            ),
        ]
    )
    evaluate = agent_task_evaluator(
        build_agent, [t for t, _ in TASKS], score, max_turns=3
    )

    def show(record: EvolutionRecord) -> None:
        marker = "+" if record.accepted else "-"
        payload = record.candidate.payload
        summary = " | ".join(
            f"{key}={str(payload.get(key, ''))[:45]!r}"
            for key in GENOME.mutable_keys()
            if str(payload.get(key, "")).strip()
        )
        print(
            f"{marker} {record.candidate.id}  {record.candidate.operator:>13}  "
            f"fitness {record.evaluation.fitness:.2f}  {summary}"
        )

    result = run_evolution(
        seeds=[seed_agent_payload("You are a helpful assistant.")],
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
        best = result.best.candidate.payload
        print(f"best system_prompt: {best['system_prompt']!r}")
        print(f"best instructions:  {best['instructions']!r}")
        print(f"fitness: {result.best.evaluation.fitness:.2f}")


if __name__ == "__main__":
    main()
