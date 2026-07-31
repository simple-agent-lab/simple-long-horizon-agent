"""Common multi-agent workflows for Simple Long Horizon Agent.

A *workflow* orchestrates several agents; it is **not** a new agent loop.
Every agent in every workflow here is driven by the project's single ReAct
loop — `simple_long_horizon_agent.core.run`, reached via `Agent.run(task)` and wrapped
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
    final_output,
    never_abort,
    run_agent,
)
from .parallel import (
    make_aggregator_agent,
    run_parallel,
)
from .planner_executor import (
    make_executor_agent,
    make_planner_agent,
    run_planner_executor,
)
from .reflection import (
    DEFAULT_APPROVAL_MARKER,
    is_approved,
    make_critic_agent,
    make_generator_agent,
    run_reflection,
)
from .routing import (
    Route,
    make_router_agent,
    run_routing,
    select_route,
)
from .sequential import JoinFn, default_join, run_chain
from .pdr import (
    make_distiller_agent,
    run_pdr,
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
    VERIFY_BEFORE_DONE_ADDENDUM,
    VERIFY_CONTINUATION,
    command_verifier_check,
    default_check,
    executed_completion_check,
    judge_agent_check,
    make_completion_judge,
    model_declared_check,
    update_goal_tool,
    verified_completion_check,
)
from .trace import (
    compose_workflow_trace_state,
    workflow_steps_breakdown,
    write_workflow_subagent_traces,
)

__all__ = [
    # base
    "StepResult",
    "WorkflowResult",
    "run_agent",
    "final_output",
    "as_text",
    "never_abort",
    # sequential
    "run_chain",
    "default_join",
    "JoinFn",
    # planner / executor
    "run_planner_executor",
    "make_planner_agent",
    "make_executor_agent",
    # reflection
    "run_reflection",
    "is_approved",
    "make_generator_agent",
    "make_critic_agent",
    "DEFAULT_APPROVAL_MARKER",
    # routing
    "Route",
    "run_routing",
    "select_route",
    "make_router_agent",
    # parallel
    "run_parallel",
    "make_aggregator_agent",
    # parallel-distill-refine (PDR)
    "run_pdr",
    "make_distiller_agent",
    # goal loop
    "run_goal_loop",
    "GoalBudgets",
    "CompletionResult",
    "CompletionCheck",
    "GoalResult",
    "GoalStatus",
    # goal checks
    "update_goal_tool",
    "model_declared_check",
    "command_verifier_check",
    "executed_completion_check",
    "judge_agent_check",
    "default_check",
    # verified completion (the reusable judge-gate optimization)
    "make_completion_judge",
    "verified_completion_check",
    "VERIFY_BEFORE_DONE_ADDENDUM",
    "VERIFY_CONTINUATION",
    # trace helpers
    "compose_workflow_trace_state",
    "workflow_steps_breakdown",
    "write_workflow_subagent_traces",
]
