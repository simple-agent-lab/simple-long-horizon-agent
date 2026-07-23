# SWE-bench Pro Repo-Chain Experiment

This is the operator contract for long SWE-bench Pro repo-chain runs. Executable
behavior lives in:

- `evals/swebench/pro_chain_runner.py`: shared CLI, auth-lane scheduling,
  provider environment, run-level files, and prediction collection.
- `runs/swebench/run_swebench_pro_repo_chains.py`: repo-state planning and its
  ordered per-instance container loop.
- `runs/swebench/run_swebench_pro_memory_chains.py`: filesystem-memory planning,
  namespace mounts, and its ordered per-instance container loop.
- `src/simple_agent_lab/evals/chain.py`: in-container continuation, handoff,
  compression, invalid-prompt recovery, and trace output.
- `src/simple_agent_lab/evals/suites/swebench/container.py`: SWE-bench prompts,
  agent construction, workspace preparation, and patch extraction.
- `evals/swebench/pro_memory_chain.py`: shared issue-chain manifest loading and
  longest-first planning.

## Experiment Shape

The runner evaluates ordered issue chains from an explicit `--chains-json`
manifest. It does not use filesystem memory. After each instance, the host
passes `out/chain_state.json` to the next instance as
`input/chain_state.json`, preserving the model-visible continuation across
fresh SWE-bench containers.

Defaults:

- Dataset: `ScaleAI/SWE-bench_Pro`, split `test`.
- Agent: `bash`; alternatives are `loop` and `pdr`.
- Compression: `none`.
- Context protection: model-authored handoff at `217600` estimated tokens.
- Max solver turns: `250` per instance, shared across handoff windows.
- API kind: `openai-responses`.
- Full per-instance trajectory output: enabled.

The independent experiment variables are:

- `--agent-flavor {bash,loop,pdr}`.
- `--task-tool` to add task delegation.
- `--compression-strategy {none,summarize}`.
- `--handoff` / `--no-handoff` for the no-compression arm.
- `--context-window-tokens N` for the handoff threshold.

`summarize` and handoff are mutually exclusive. With compression disabled,
handoff is on by default; `--no-handoff` deliberately runs without context
window protection.

## Planning And Scheduling

`--chains-json` is required. The normal input is
`evals/swebench/data/swe_bench_pro_chain_experiment_nodes_deep.jsonl`. Each
JSONL node needs only `chain_id`, `step_index`, `instance_id`, `repo`, and
`commit_time`.

The planner:

1. Orders nodes within each chain by `step_index`, then by commit time and
   instance id as a fallback.
2. Adds every selected dataset row not covered by the manifest as a singleton
   unit.
3. Orders units longest-first, with deterministic repository and chain-id tie
   breaks.
4. Assigns each running unit one provider-auth lane for its lifetime.

`--parallel slots` uses the expanded lane count from
`--provider-auth-envs`. The host maps a
selected token such as `OPENAI_AUTH_TOKEN2` into the container's canonical
`OPENAI_AUTH_TOKEN`.

Run ids identify one invocation. The runner canonicalizes the id and refuses a
non-empty existing run directory. Exact interrupted-chain resume is not
supported; use a new run id.

## Container And Continuation Flow

For each planned unit, the host runs its instances in order:

1. Stage `input/instance.json` and `input/chain_config.json`.
2. Stage the prior `out/chain_state.json` as `input/chain_state.json`, except
   for the unit's first instance.
3. Launch `run_suite_instance` with
   `simple_agent_lab.evals.chain` as the container runner.
4. Collect `out/result.json`, `out/chain_state.json`, and optional trace files.
5. Refresh the run-level predictions file.

The agent and bash tool run inside the current task container. The host does
not keep a live agent and does not proxy commands through a separate
`docker exec` tool.

Continuation state contains the task seed, active model-visible messages, and
small runtime state. Inactive transcript entries are not copied to the next
container; they remain available only in the originating instance trajectory.
Continuation files are internal artifacts produced by this runner, not a
hand-edited interchange format.

## Context Management

With `--compression-strategy summarize`, `SummarizeStrategy` uses the same
provider settings as the solver, a threshold of `217600`, `keep_recent=4`, and
preserves `task`, `system`, and `context` messages. Prior instance prompts are
demoted from `task` to ordinary messages before a new instance starts, so only
the current problem remains pinned.

With handoff enabled, the runner checks context size at clean turn boundaries:

- Mid-instance: it asks the model for durable repository notes, then keeps only
  the current task plus those notes in the active context and continues the
  same instance.
- At an instance boundary: if the context is already over the threshold, the
  next instance starts with only the handoff notes.

Handoff generation uses a scratch state. Failure or empty output leaves the
existing context intact. Successful resets emit a `ContextCompressionEvent`
with strategy `context_window_handoff`; handoff-generation turns do not consume
the solver's turn budget.

## Invalid-Prompt Recovery

For provider `invalid_prompt` errors, including code `-4321`:

- A rejected current task is removed from continuation and the instance is
  skipped.
- A rejected tool output causes its tool-call/result exchange to be replaced
  with a short retry note.
- Tool-output retries share the instance turn budget and are capped at 20.
- Exhaustion clears the active context before the next instance.

This recovery assumes tool-call pairs and continuation files came from the
normal runtime path; it does not repair arbitrary hand-edited or corrupted
chain-state payloads.

## Outputs

Each instance directory contains:

- `input/instance.json`
- `input/chain_config.json`
- optional `input/chain_state.json`
- `out/result.json`
- `out/chain_state.json`
- `out/trajectory.jsonl` and optional raw trace data unless trajectories are
  disabled

`result.json` records the patch, status/error, chain id, selected agent and
context mode, compression metrics, event span, and handoff counters. There is
no chain-part layer: one planned issue chain or singleton is one run unit.

Batch outputs are `experiment.json`, `instances.jsonl`, the run predictions
JSONL, optional skipped/failure records, and one
`_repo_chains/<chain>/summary.json` per unit. Prediction refreshes retain every
planned instance; missing results receive an empty patch so the denominator
does not shrink during partial runs.

## Recommended Command

Set `OPENAI_MODEL`, provider tokens, and optional `REASONING_EFFORT` in `.env`.
Prepare the wheelhouse on the first run or after dependency changes.

```bash
uv run --extra swebench python runs/swebench/run_swebench_pro_repo_chains.py \
  --all \
  --chains-json evals/swebench/data/swe_bench_pro_chain_experiment_nodes_deep.jsonl \
  --provider-auth-envs OPENAI_AUTH_TOKEN:12,OPENAI_AUTH_TOKEN2:11 \
  --api-kind openai-responses \
  --max-turns 250 \
  --run-official-eval
```

Add `--compression-strategy summarize`, `--task-tool`, or an alternate
`--agent-flavor` for another experiment arm. Add `--no-write-trajectories`
when storage cost outweighs replay/debugging value.

## Validation

For code changes, run both runner `--help` commands, a small `--plan-only`
fixture, and `runs/dev/run_ci.sh`. Live provider and official-evaluator runs are
operator checks, not required CI.
