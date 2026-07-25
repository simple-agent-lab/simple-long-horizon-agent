# SWE-bench Pro Repo-Chain Experiment

Operator contract for long SWE-bench Pro chain runs. Executable behavior lives
in `evals/swebench/pro_chain_runner.py` (shared CLI, auth lanes, predictions),
the two host runners under `runs/swebench/run_swebench_pro_*_chains.py`,
`src/simple_agent_lab/evals/chain.py` (continuation, handoff, recovery), and
`src/simple_agent_lab/evals/suites/swebench/container.py`.

## Shape

The runner evaluates ordered issue chains from an explicit `--chains-json`
manifest. After each instance the host passes `out/chain_state.json` to the next
as `input/chain_state.json`, preserving the model-visible continuation across
fresh containers. The repo-chain runner uses no filesystem memory; the memory
runner is its sibling arm.

Defaults: `ScaleAI/SWE-bench_Pro` `test`, agent `bash`, compression `none`,
model-authored handoff at 217600 estimated tokens, 250 solver turns per instance
shared across handoff windows, `openai-responses`, trajectories on.

Independent variables: `--agent-flavor {bash,loop,pdr}`, `--task-tool`,
`--compression-strategy {none,summarize}`, `--handoff`/`--no-handoff`,
`--context-window-tokens N`. **`summarize` and handoff are mutually exclusive**;
with compression off, handoff is on by default and `--no-handoff` deliberately
runs with no context-window protection.

## Planning

Each JSONL node needs only `chain_id`, `step_index`, `instance_id`, `repo`, and
`commit_time`. The planner orders nodes within a chain by `step_index` (commit
time then instance id as tiebreaks), adds uncovered dataset rows as singletons,
orders units longest-first with deterministic tie breaks, and assigns each
running unit one provider-auth lane for its lifetime. `--parallel` uses the
expanded lane count from `--provider-auth-envs`; the host maps a selected token
such as `OPENAI_AUTH_TOKEN2` onto the container's canonical `OPENAI_AUTH_TOKEN`.

Run ids identify one invocation. The runner refuses a non-empty existing run
directory, and **interrupted-chain resume is not supported** — use a new run id.

## Context management

With `--compression-strategy summarize`, `SummarizeStrategy` uses the solver's
provider settings, threshold 217600, `keep_recent=4`, preserving `task`,
`system`, and `context`. Prior instance prompts are demoted from `task` to
ordinary messages before a new instance starts, so only the current problem
stays pinned.

With handoff, context size is checked at clean turn boundaries:

- **Mid-instance** — ask the model for durable repository notes, keep only the
  current task plus those notes, continue the same instance.
- **At an instance boundary** — if already over threshold, the next instance
  starts with only the handoff notes.

Handoff generation uses a scratch state; failure or empty output leaves context
intact. Successful resets emit a `ContextCompressionEvent` with strategy
`context_window_handoff`, and handoff turns do not consume the solver budget.

## Invalid-prompt recovery

For provider `invalid_prompt` errors (including code `-4321`): a rejected task
is dropped from continuation and the instance skipped; a rejected tool output
has its tool-call/result exchange replaced with a short retry note. Tool-output
retries share the instance turn budget, capped at 20; exhaustion clears the
active context before the next instance. This assumes continuation files came
from the normal runtime path — it does not repair hand-edited chain state.

## Outputs

Per instance: `input/instance.json`, `input/chain_config.json`, optional
`input/chain_state.json`, `out/result.json`, `out/chain_state.json`, and
`out/trajectory.jsonl`. `result.json` records the patch, status, chain id,
agent and context mode, compression metrics, event span, and handoff counters.
One planned chain or singleton is one run unit — there is no chain-part layer.

Each instance writes `result.json` immediately; the run-level predictions file
is written once, atomically, after every auth lane stops. Final collection
retains every planned instance, giving missing results an empty patch so
failures do not shrink the denominator.

## Recommended command

```bash
uv run --extra swebench python -m runs.swebench.run_swebench_pro_repo_chains \
  --all \
  --chains-json evals/swebench/data/swe_bench_pro_chain_experiment_nodes_deep.jsonl \
  --provider-auth-envs OPENAI_AUTH_TOKEN:12,OPENAI_AUTH_TOKEN2:11 \
  --api-kind openai-responses \
  --max-turns 250 \
  --run-official-eval
```

For code changes, run both runners' `--help`, a small `--plan-only` fixture, and
`runs/dev/run_ci.sh`. Live provider and official-evaluator runs are operator
checks, not required CI.
