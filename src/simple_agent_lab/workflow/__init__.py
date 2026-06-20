"""Common multi-agent workflows for Simple Agent Lab.

A *workflow* orchestrates several agents; it is **not** a new agent loop.
Every agent in every workflow here is driven by the project's single ReAct
loop — `simple_agent_lab.core.run`, reached via `Agent.run(task)` and wrapped
by `workflow.base.run_agent`. The workflow code only decides which agent runs
when and what task each receives; the per-agent turn/tool handling is always
the same `core.run` code path the bash agent uses.

Patterns included:

- `run_chain` — sequential prompt chaining (output of each stage feeds the
  next).
- `run_planner_executor` — a planner produces a plan, an executor carries it
  out.
- `run_reflection` — a generator drafts, a critic reviews, the generator
  revises until approved or out of rounds.
- `run_routing` — a router classifies the task and dispatches to one
  specialist.
- `run_parallel` — workers run concurrently, then an optional aggregator
  folds their answers into one.

Each orchestration function takes pre-built `Agent`s, so it is independent of
provider/model choice. The `make_*` builders are convenience presets that
construct sensible default agents for a workflow given a provider.

All workflows return a `WorkflowResult` (a final `output` plus per-step
`StepResult`s, each carrying its full `State` for tracing).
"""

from __future__ import annotations

from .base import (
    StepResult,
    WorkflowResult,
    as_text,
    emitted_final,
    final_output,
    fork_state,
    never_abort,
    pick_index,
    resume_agent,
    run_agent,
)
from .parallel import (
    AGGREGATOR_ROLE,
    AGGREGATOR_SYSTEM_PROMPT,
    make_aggregator_agent,
    run_parallel,
)
from .planner_executor import (
    EXECUTOR_ROLE,
    EXECUTOR_SYSTEM_PROMPT,
    PLANNER_ROLE,
    PLANNER_SYSTEM_PROMPT,
    make_executor_agent,
    make_planner_agent,
    run_planner_executor,
)
from .reflection import (
    CRITIC_ROLE,
    CRITIC_SYSTEM_PROMPT,
    DEFAULT_APPROVAL_MARKER,
    GENERATOR_ROLE,
    GENERATOR_SYSTEM_PROMPT,
    is_approved,
    make_critic_agent,
    make_generator_agent,
    run_reflection,
)
from .routing import (
    ROUTER_ROLE,
    Route,
    make_router_agent,
    run_routing,
    select_route,
)
from .sequential import JoinFn, default_join, run_chain
from .pdr import (
    DISTILLER_ROLE,
    DISTILLER_SYSTEM_PROMPT,
    make_distiller_agent,
    run_pdr,
)
from .tournament import (
    SELECTOR_ROLE,
    SELECTOR_SYSTEM_PROMPT,
    SUMMARIZER_ROLE,
    SUMMARIZER_SYSTEM_PROMPT,
    make_selector_agent,
    make_summarizer_agent,
    run_rtv,
)
from .tree_search import (
    REFLECT_ROLE,
    REFLECT_SYSTEM_PROMPT,
    VALUE_ROLE,
    VALUE_SYSTEM_PROMPT,
    make_reflect_agent,
    make_value_agent,
    run_mcts,
)
from .goal_loop import (
    CompletionCheck,
    CompletionResult,
    GoalBudgets,
    GoalResult,
    GoalStatus,
    run_goal_loop,
)
from .goal_checks import (
    UPDATE_GOAL_TOOL_NAME,
    command_verifier_check,
    default_check,
    judge_agent_check,
    model_declared_check,
    update_goal_tool,
)

__all__ = [
    # base
    "StepResult",
    "WorkflowResult",
    "run_agent",
    "resume_agent",
    "fork_state",
    "emitted_final",
    "final_output",
    "as_text",
    "never_abort",
    "pick_index",
    # sequential
    "run_chain",
    "default_join",
    "JoinFn",
    # planner / executor
    "run_planner_executor",
    "make_planner_agent",
    "make_executor_agent",
    "PLANNER_ROLE",
    "PLANNER_SYSTEM_PROMPT",
    "EXECUTOR_ROLE",
    "EXECUTOR_SYSTEM_PROMPT",
    # reflection
    "run_reflection",
    "is_approved",
    "make_generator_agent",
    "make_critic_agent",
    "DEFAULT_APPROVAL_MARKER",
    "GENERATOR_ROLE",
    "GENERATOR_SYSTEM_PROMPT",
    "CRITIC_ROLE",
    "CRITIC_SYSTEM_PROMPT",
    # routing
    "Route",
    "run_routing",
    "select_route",
    "make_router_agent",
    "ROUTER_ROLE",
    # parallel
    "run_parallel",
    "make_aggregator_agent",
    "AGGREGATOR_ROLE",
    "AGGREGATOR_SYSTEM_PROMPT",
    # tournament (RTV)
    "run_rtv",
    "make_selector_agent",
    "make_summarizer_agent",
    "SELECTOR_ROLE",
    "SELECTOR_SYSTEM_PROMPT",
    "SUMMARIZER_ROLE",
    "SUMMARIZER_SYSTEM_PROMPT",
    # parallel-distill-refine (PDR)
    "run_pdr",
    "make_distiller_agent",
    "DISTILLER_ROLE",
    "DISTILLER_SYSTEM_PROMPT",
    # tree search (MCTS / LATS-lite)
    "run_mcts",
    "make_value_agent",
    "make_reflect_agent",
    "VALUE_ROLE",
    "VALUE_SYSTEM_PROMPT",
    "REFLECT_ROLE",
    "REFLECT_SYSTEM_PROMPT",
    # goal loop
    "run_goal_loop",
    "GoalBudgets",
    "CompletionResult",
    "CompletionCheck",
    "GoalResult",
    "GoalStatus",
    # goal checks
    "update_goal_tool",
    "UPDATE_GOAL_TOOL_NAME",
    "model_declared_check",
    "command_verifier_check",
    "judge_agent_check",
    "default_check",
]
