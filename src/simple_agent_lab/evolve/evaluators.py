"""Bridge from a candidate to the existing agent runtime, for fitness.

`agent_task_evaluator` is the piece that makes this *agent* evolution rather
than program evolution: a candidate's payload configures an `Agent` (its
system prompt, tool descriptions, whatever `build_agent` reads), the agent
runs over a fixed task list through the ordinary `workflow.base.run_agent`
loop, and per-task scores fold into one `Evaluation`.

Evaluation cost dominates evolution cost, so keep `tasks` small while
iterating on a method and grow it only when selection starts overfitting to
the task list.
"""

from __future__ import annotations

from typing import Callable, Sequence

from simple_agent_lab.core import Agent
from simple_agent_lab.workflow.base import run_agent

from .types import Candidate, EvaluateFn, Evaluation

# (task, final_output) -> score. Convention: higher is better, 1.0 is solved.
ScoreFn = Callable[[str, str], float]

_FEEDBACK_TASKS = 5  # worst tasks quoted back to the proposer
_FEEDBACK_OUTPUT_CHARS = 300


def agent_task_evaluator(
    build_agent: Callable[[Candidate], Agent],
    tasks: Sequence[str],
    score: ScoreFn,
    *,
    max_turns: int = 10,
    solved_threshold: float = 1.0,
) -> EvaluateFn:
    """An `EvaluateFn` that scores a candidate by running an agent it defines.

    `fitness` is the mean task score. `feedback` quotes the worst-scoring
    tasks with the agent's (truncated) output, so an LLM proposer sees what
    actually went wrong instead of just a number. A `build_agent` or run
    crash surfaces through the loop's failure handling (the candidate is
    recorded as incorrect), so one broken candidate never stops a run.
    """

    if not tasks:
        raise ValueError("agent_task_evaluator needs at least one task")

    def evaluate(candidate: Candidate) -> Evaluation:
        scores: list[float] = []
        outputs: list[str] = []
        for task in tasks:
            agent = build_agent(candidate)
            step = run_agent(agent, task, max_turns=max_turns)
            scores.append(score(task, step.output))
            outputs.append(step.output)

        fitness = sum(scores) / len(scores)
        worst = sorted(range(len(tasks)), key=lambda i: scores[i])
        feedback_lines = [
            f"task: {tasks[i]}\n  score: {scores[i]:.2f}\n"
            f"  output: {outputs[i][:_FEEDBACK_OUTPUT_CHARS]}"
            for i in worst[:_FEEDBACK_TASKS]
            if scores[i] < solved_threshold
        ]
        return Evaluation(
            fitness=fitness,
            correct=True,
            metrics={
                "task_scores": list(scores),
                "solved": sum(1 for s in scores if s >= solved_threshold),
                "total": len(tasks),
            },
            feedback="\n".join(feedback_lines),
        )

    return evaluate
