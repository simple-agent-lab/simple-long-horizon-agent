# Workflow

This package is a small library of **agent workflows**: orchestrations that coordinate several specialized agents to solve a task that a single agent loop handles poorly.


## Base ReAct Agent Loop

Every agent in every workflow is driven by `simple_long_horizon_agent.core.run`, reached through `Agent.run(task)`, which is implemented in `base.run_agent`:

```python
# workflow/base.py
def run_agent(agent, task, *, max_turns=10, abort=never_abort, ...):
    state, events = agent.run(task, max_turns=max_turns, abort=abort)  # core.run
    for _ in events:               # drain the generator to advance the loop
        if abort(): break
    return StepResult(name=agent.name, output=final_output(state, agent.name),
                      state=state, ...)
```


## Final Result Types

Every workflow returns a `WorkflowResult`, every individual agent run is a `StepResult`.

| Type | Fields | Purpose |
|------|--------|---------|
| `StepResult` | `name`, `role`, `task`, `output`, `state` | One agent run. `output` is its final answer text; `state` is the full run (messages + trace events) so the step is replayable. |
| `WorkflowResult` | `output`, `steps: list[StepResult]` | The whole run. `output` is the final answer; `steps` is the per-agent audit trail. `.states` returns each step's `State`. |

Helper functions you can reuse when composing your own workflows:

- `run_agent(agent, task, *, max_turns=10, abort=never_abort, context=(), role="")` — run one agent to completion, capture a `StepResult`.
- `final_output(state, agent_name)` — extract an agent's answer (prefers the terminal `final` message; falls back to its last assistant turn if the run hit `max_turns`).
- `as_text(task)` — best-effort plain text for a task (passes strings through).
- `never_abort()` — the default `AbortFlag`.


## Supporting Agent Workflows

### 1. Sequential chain — `sequential.py`

Run agents one after another, threading each stage's output into the next.

```python
from simple_long_horizon_agent.workflow import run_chain

result = run_chain([outliner, drafter, polisher], "Write a tutorial on async IO")
print(result.output)            # the polisher's final answer
print(len(result.steps))        # 3
```

- First agent gets the raw `task`. Each later agent gets `join(task, previous_output)`.
- Default `join` (`default_join`) carries the **original task forward** alongside the previous answer.
- Override with `run_chain(..., join=my_join)` where `join(task, prev) -> str`.


### 2. Planner / executor — `planner_executor.py`

Split "decide what to do" from "do it". The plan becomes an inspectable artifact (the first step's `output`).

```python
from simple_long_horizon_agent.workflow import (
    run_planner_executor, make_planner_agent, make_executor_agent,
)
from simple_long_horizon_agent.tools.bash import make_bash_tool

planner  = make_planner_agent(provider)                       # no tools: it just plans
executor = make_executor_agent(provider, tools=[make_bash_tool(cwd=workdir)])

result = run_planner_executor(planner, executor, "Refactor the auth module")
plan   = result.steps[0].output        # the plan
answer = result.output                 # the executor's result
```


### 3. Reflection — `reflection.py`

A generator drafts, a critic reviews, the generator revises — repeat until the critic approves or a round budget is hit.

```python
from simple_long_horizon_agent.workflow import (
    run_reflection, make_generator_agent, make_critic_agent,
)

gen    = make_generator_agent(provider)
critic = make_critic_agent(provider)    # prompted to emit "APPROVED" when satisfied

result = run_reflection(gen, critic, "Prove the function terminates", max_rounds=3)
print(result.output)                    # the latest (accepted) draft
```

- **Stop condition:** the critic is asked to output an approval marker (default `"APPROVED"`). `run_reflection` watches for it via `is_approved`. Use a custom marker with `approval_marker=...` (keep the critic's prompt in sync).
- `steps` records every run in order: `draft, critique, revise, critique, …`.


### 4. Routing — `routing.py`

A router picks the single best specialist, then that specialist handles the task.

```python
from simple_long_horizon_agent.workflow import Route, run_routing, make_router_agent

routes = [
    Route("coder",      coder_agent,      "write / debug code"),
    Route("researcher", researcher_agent, "look things up and summarize"),
]
router = make_router_agent(provider, routes)     # system prompt auto-lists the routes

result = run_routing(router, routes, "find the off-by-one bug in utils.py",
                     default="coder")
print(result.steps[1].name)              # which specialist ran
```

- `Route(name, agent, description="")` is one destination.
- If nothing resolves, the result holds **only** the router step (so you can see what it said) — set `default=` to always fall through to one route.


### 5. Parallelization — `parallel.py`

Fan several agents out concurrently, then optionally fold their answers into one.

```python
from simple_long_horizon_agent.workflow import run_parallel, make_aggregator_agent

# Ensemble: same task to every worker, then synthesize.
agg = make_aggregator_agent(provider)
result = run_parallel([gpt_worker, claude_worker, local_worker],
                      "Summarize this incident report",
                      aggregator=agg)

# Map-reduce: a different sub-task per worker (pass `tasks=`).
result = run_parallel([w1, w2, w3], task="(unused)",
                      tasks=["section 1", "section 2", "section 3"],
                      aggregator=agg)
```

- Workers run in a `ThreadPoolExecutor` (each has its own `State`, so there's no shared loop state to race on). `steps` preserve **worker order**, not completion order; the aggregator step is appended last.
- Without an `aggregator`, the output is a labelled concatenation of the worker answers.


## Building the Agents You Pass in

The orchestration functions take pre-built `Agent`s. Three ways to make them:

```python
# a) A workflow's own preset builders (sensible role prompt, no tools by default)
from simple_long_horizon_agent.workflow import make_critic_agent
critic = make_critic_agent(provider)

# b) The generic factory, for full control over prompt + tools
from simple_long_horizon_agent import make_llm_agent
from simple_long_horizon_agent.tools.bash import make_bash_tool
coder = make_llm_agent(name="coder", provider=provider,
                       system_prompt="You only write Python.",
                       tools=[make_bash_tool(cwd=workdir)])

# c) Reuse an existing preset agent as a specialist/worker
from simple_long_horizon_agent.agents import make_bash_agent
shell = make_bash_agent(provider, name="shell")
```

A `provider` is a plain data value:

```python
from simple_long_horizon_agent.llm import Provider
provider = Provider(id="claude", api="anthropic-messages",
                    model="claude-sonnet-4-5", api_key_env="ANTHROPIC_API_KEY")
```

## Tracing a run

Because every step keeps its `State`, a whole workflow is inspectable after the fact (events, token usage, every message each subagent saw):

```python
from simple_long_horizon_agent import print_trace

result = run_reflection(gen, critic, task)
for step in result.steps:
    print(f"=== {step.role} ({step.name}) ===")
    print_trace(step.state)
```
