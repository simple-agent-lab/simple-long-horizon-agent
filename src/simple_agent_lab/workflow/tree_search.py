"""Monte Carlo / Language Agent Tree Search (LATS-lite) over a conversation.

The frontier counterpart of `run_goal_loop`: where the goal loop drives a single
*linear* conversation until a check passes, tree search explores a *branching*
tree of continuations, scores each with an independent value agent, and keeps
the best — adding backtracking the linear loop lacks (LATS, Language Agent Tree
Search; ReST-MCTS*). It is built entirely on `Agent.run` / `Agent.resume` and
the event-sourced `fork_state` seam, so `core` is untouched.

A node is a conversation `State`. The root is virtual; its children are fresh
`worker` runs on the task. A deeper child is produced by forking its parent's
`State` and `resume`-ing it one bounded segment further. Each new node is scored
in [0, 1] by the `value` agent; the score is backpropagated up the visited path;
selection descends by UCT (exploit mean value, explore rarely-visited nodes).
The result is the highest-valued terminal node's answer.

Granularity honesty: the inner `core.run` loop is a black box that runs to
`final` or `max_turns`, so a node is a *bounded trajectory segment*
(`segment_turns`), not a single token/tool call. On single-turn, tool-free tasks
every child is terminal at depth 1, so the search degrades gracefully to
best-of-`branch` by value — its depth shows on multi-turn agentic tasks.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from simple_agent_lab.core import Agent
from simple_agent_lab.llm import Provider as LLMProvider
from simple_agent_lab.llm_agent import make_llm_agent
from simple_agent_lab.tools import AbortFlag

from .base import (
    StepResult,
    WorkflowResult,
    as_text,
    emitted_final,
    fork_state,
    never_abort,
    resume_agent,
    run_agent,
)

VALUE_ROLE = "Score how close a trajectory is to solving the task."
VALUE_SYSTEM_PROMPT = (
    "You are a value function for tree search. Given the task and one agent's "
    "work so far, estimate how close it is to a correct, complete solution. "
    "Reply with ONLY a single number from 0.0 (no progress / wrong) to 1.0 "
    "(verifiably solved)."
)
REFLECT_ROLE = "Diagnose a weak trajectory so siblings can avoid its mistakes."
REFLECT_SYSTEM_PROMPT = (
    "You are a reflection agent for tree search. Given the task and a weak or "
    "failed attempt, write one or two concrete sentences on what went wrong and "
    "what a better attempt should do differently. Be specific and actionable."
)

CONTINUE_PROMPT = (
    "Continue working toward the task. Take the next concrete step(s) and, when "
    "you can verify the task is fully solved, give the final answer."
)


@dataclass
class _Node:
    """One search node: a conversation `State` and its MCTS bookkeeping."""

    depth: int
    terminal: bool
    state: Any = None  # State; None only for the virtual root
    output: str = ""
    step: StepResult | None = None
    parent: "_Node | None" = None
    children: list["_Node"] = field(default_factory=list)
    untried: int = 0
    visits: int = 0
    total_value: float = 0.0
    value: float = 0.0
    reflection: str = ""

    @property
    def mean_value(self) -> float:
        return self.total_value / self.visits if self.visits else 0.0


def _parse_value(text: str) -> float:
    """Extract a [0, 1] score from the value agent's free text (default 0.0)."""
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return 0.0
    return max(0.0, min(1.0, float(match.group())))


def _value_prompt(task: str, output: str) -> str:
    return (
        "Estimate how close this attempt is to a correct, complete solution.\n\n"
        f"Task:\n{task}\n\nAttempt so far:\n{output}\n\n"
        "Reply with ONLY a number from 0.0 to 1.0."
    )


def _reflect_prompt(task: str, output: str) -> str:
    return (
        "This attempt is weak or failed. In one or two sentences, say what went "
        "wrong and what a better attempt should do differently.\n\n"
        f"Task:\n{task}\n\nAttempt:\n{output}"
    )


def _can_expand(node: _Node, max_depth: int) -> bool:
    return node.untried > 0 and not node.terminal and node.depth < max_depth


def _uct(child: _Node, parent_visits: int, c: float) -> float:
    if child.visits == 0:
        return math.inf
    return child.mean_value + c * math.sqrt(math.log(parent_visits) / child.visits)


def _select(root: _Node, max_depth: int, c: float) -> _Node | None:
    """Descend from `root` by UCT to a node that can still be expanded.

    Returns the first expandable node on the best-UCT path, or `None` if that
    path dead-ends (the caller then scans for any expandable node elsewhere).
    """
    node = root
    while True:
        if _can_expand(node, max_depth):
            return node
        if not node.children:
            return None
        node = max(node.children, key=lambda ch: _uct(ch, node.visits, c))


def _any_expandable(root: _Node, max_depth: int) -> _Node | None:
    stack = [root]
    while stack:
        node = stack.pop()
        if _can_expand(node, max_depth):
            return node
        stack.extend(node.children)
    return None


def _backprop(node: _Node, value: float) -> None:
    cursor: _Node | None = node
    while cursor is not None:
        cursor.visits += 1
        cursor.total_value += value
        cursor = cursor.parent


def run_mcts(
    worker: Agent,
    value: Agent,
    task: str,
    *,
    budget: int = 16,
    branch: int = 3,
    max_depth: int = 4,
    segment_turns: int = 2,
    reflect: Agent | None = None,
    reflect_threshold: float = 0.3,
    c_uct: float = 1.4,
    abort: AbortFlag = never_abort,
) -> WorkflowResult:
    """Search a tree of continuations and return the best terminal answer.

    Runs up to `budget` expansions. Each expansion selects a node by UCT, grows
    one child (a fresh `worker` run at the root, else a forked `resume` of the
    parent's `State` bounded to `segment_turns`), scores it with `value` in
    [0, 1], and backpropagates. `branch` bounds children per node; `max_depth`
    bounds segment depth. When `reflect` is given, a child scoring below
    `reflect_threshold` yields a note attached to its parent that later siblings
    see in their continuation prompt (LATS-style reflection).

    The output is the highest-scoring terminal node (fallback: highest-scoring
    node seen). `steps` holds every worker, value, and reflection run.
    """
    if budget < 1:
        raise ValueError("run_mcts requires budget >= 1")
    if branch < 1:
        raise ValueError("run_mcts requires branch >= 1")

    task_text = as_text(task)
    root = _Node(depth=0, terminal=False, untried=branch)
    steps: list[StepResult] = []
    all_nodes: list[_Node] = []

    expansions = 0
    while expansions < budget and not abort():
        node = _select(root, max_depth, c_uct) or _any_expandable(root, max_depth)
        if node is None:
            break

        child = _grow(node, worker, task_text, segment_turns, abort)
        node.untried -= 1
        node.children.append(child)
        all_nodes.append(child)
        if child.step is not None:
            steps.append(child.step)
        expansions += 1

        score, value_step = _score(value, task_text, child, abort)
        child.value = score
        steps.append(value_step)
        _backprop(child, score)

        if reflect is not None and score < reflect_threshold and not abort():
            note = run_agent(
                reflect,
                _reflect_prompt(task_text, child.output),
                abort=abort,
                role="reflect",
            )
            steps.append(note)
            node.reflection = note.output

    best = _best_node(all_nodes)
    return WorkflowResult(output=best.output if best else "", steps=steps)


def _grow(
    node: _Node, worker: Agent, task: str, segment_turns: int, abort: AbortFlag
) -> _Node:
    """Produce one child of `node`: a fresh run at the root, else a forked resume."""
    if node.parent is None and node.state is None:
        prompt = task if not node.reflection else f"{task}\n\nNote:\n{node.reflection}"
        step = run_agent(worker, prompt, max_turns=segment_turns, abort=abort, role="rollout")
    else:
        followup = CONTINUE_PROMPT
        if node.reflection:
            followup = f"{CONTINUE_PROMPT}\n\nNote:\n{node.reflection}"
        step = resume_agent(
            worker,
            fork_state(node.state),
            followup,
            max_turns=segment_turns,
            abort=abort,
            role="rollout",
        )
    return _Node(
        depth=node.depth + 1,
        terminal=emitted_final(step.state, worker.name),
        state=step.state,
        output=step.output,
        step=step,
        parent=node,
        # A child inherits its parent's branching factor. `node.untried` has not
        # been decremented yet, so untried + already-created children == branch.
        untried=node.untried + len(node.children),
    )


def _score(
    value: Agent, task: str, child: _Node, abort: AbortFlag
) -> tuple[float, StepResult]:
    step = run_agent(value, _value_prompt(task, child.output), abort=abort, role="value")
    return _parse_value(step.output), step


def _best_node(nodes: list[_Node]) -> _Node | None:
    terminal = [n for n in nodes if n.terminal]
    pool = terminal or nodes
    return max(pool, key=lambda n: n.value) if pool else None


def make_value_agent(
    provider: LLMProvider,
    *,
    name: str = "value",
    role: str = VALUE_ROLE,
    system_prompt: str = VALUE_SYSTEM_PROMPT,
    request_extra: Mapping[str, Any] | None = None,
    timeout_seconds: float | None = None,
) -> Agent:
    """Build the value `Agent` (scores a node's trajectory) for `run_mcts`."""
    return make_llm_agent(
        name=name,
        provider=provider,
        role=role,
        system_prompt=system_prompt,
        target="user",
        request_extra=request_extra,
        timeout_seconds=timeout_seconds,
    )


def make_reflect_agent(
    provider: LLMProvider,
    *,
    name: str = "reflect",
    role: str = REFLECT_ROLE,
    system_prompt: str = REFLECT_SYSTEM_PROMPT,
    request_extra: Mapping[str, Any] | None = None,
    timeout_seconds: float | None = None,
) -> Agent:
    """Build the optional reflection `Agent` for `run_mcts`."""
    return make_llm_agent(
        name=name,
        provider=provider,
        role=role,
        system_prompt=system_prompt,
        target="user",
        request_extra=request_extra,
        timeout_seconds=timeout_seconds,
    )
