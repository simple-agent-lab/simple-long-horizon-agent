# SWE-bench Pro Repo-Chain Experiment

This note records the intended SWE-bench Pro experiment configuration and the
current runner contract. It is an operator handoff for repo-chain runs; the
source of truth for executable behavior is
`runs/swebench/run_swebench_pro_repo_chains.py`,
`simple_agent_lab.evals.chain`, and
`evals/swebench/pro_repo_chain.py`, with issue-chain planning helpers from
`evals/swebench/pro_memory_chain.py`.

## Design Target

`runs/swebench/run_swebench_pro_repo_chains.py` is the single SWE-bench Pro
repo-chain runner. Its planning now borrows the issue-chain units and
longest-first order from `run_swebench_pro_memory_chains.py`: `--chains-json`
must be passed explicitly and usually points at the vendored deep chain-nodes
JSONL under `evals/swebench/data/`. Every covered issue keeps that manifest's
chain order, and uncovered dataset instances become singleton units. Each
planned unit still runs with repo-chain context, not filesystem memory: it
carries `out/chain_state.json` forward until that unit is exhausted. Baselines
that change agent topology, task delegation, or context handling are variables
of this runner, not separate host scripts.

The supported comparison modes are:

- `--agent-flavor {bash,loop,goal,pdr}`: selects the chain agent. The default
  `bash` flavor is the plain bash-tool agent. The `goal` flavor maps to the
  current `thread_goal_loop`; in a repo chain it reuses that same loop but is
  seeded with the shared chain state, so it inherits earlier instances' context
  (see "Goal Flavor In A Chain").
- `--compression-strategy none`: the default; no context summarization. Long
  chains stay under the window through handoff instead (see "Handoff").
- `--compression-strategy summarize`: turns on chain compression
  (`SummarizeStrategy`) and disables handoff.
- `--handoff` / `--no-handoff`: handoff is on by default; `--no-handoff` runs a
  no-compression chain "naked" with no context-window protection.
- `--context-window-tokens N`: handoff trigger threshold (default `217600`,
  i.e. `272000 * 0.8`, matching the summarize threshold for a fair comparison).
- `--task-tool`: adds the task tool to the selected chain agent or workflow
  solver. Without it, repo-chain flavors use bash only plus any workflow-local
  control tools.

When `--run-id` is omitted, the runner derives a timestamped prefix from the
selected variables, such as `pro-repo-chain-none-*` (default bash + none),
`pro-repo-chain-summarize-*`, or `pro-repo-chain-goal-task-none-*`.
Run IDs identify one invocation: both Pro chain runners refuse a non-empty
existing run directory. Choose a new `--run-id` after an interrupted run; exact
chain resume is not supported, and refusing reuse prevents stale per-instance
results or memory from entering a later experiment. Explicit IDs are
canonicalized once before any batch, instance, summary, memory, or prediction
path is built, so path separators cannot split one invocation across multiple
directories or escape the run root.

## Objective

Evaluate Simple Agent Lab long repo chains on SWE-bench Pro by running each
planned issue-chain unit as a long-lived agent chain. Instances within a unit
follow the memory-chain manifest order, and run units are submitted
longest-first so long chains occupy provider lanes early while shorter chains
backfill. The default variable set studies
`agent_flavor=bash`, `compression_strategy=none`, and `task_tool=false`;
the runner also supports goal-loop flavors, `summarize` compression, and
task-tool delegation. Common baselines are bash, goal (`--agent-flavor goal`),
goal + compression (`--agent-flavor goal --compression-strategy summarize`),
and goal + task (`--agent-flavor goal --task-tool`). The non-compression
baselines keep long chains under the context window with handoff (default on);
compression arms use `summarize` (which turns handoff off).

## Container-Inside Migration Checklist

- Add a generic eval hook for custom in-container runner modules so the Pro
  repo-chain path can use `run_suite_instance` without forking Docker backend
  logic.
- Move repo-chain execution into a wheel-shipped container runner that restores
  chain state, builds the local in-container agent, runs the current instance,
  writes `out/result.json`, and emits the next `out/chain_state.json`.
- Serialize continuation as an artifact contract (`input/chain_state.json` to
  `out/chain_state.json`) instead of keeping a host-side live `State`.
- Keep host scripts focused on planning, provider-slot assignment, staging
  config/state artifacts, launching `run_suite_instance`, collecting results,
  and refreshing predictions.
- Fold the task-tool baseline into the same host runner with `mode=repo_chain`,
  `task_tool=true`, optional compression, selected `agent_flavor`, and
  context-window handoff metadata.
- Preserve auth-slot scheduling by copying the selected host token into the
  container's canonical `OPENAI_AUTH_TOKEN` while recording the original slot in
  result metadata.
- Cover the new contract with unit tests for runner-module selection,
  chain-state round trip, in-container runner smoke, staged config payloads,
  and prediction refresh.

## Dataset

- Dataset: `ScaleAI/SWE-bench_Pro`
- Split: `test`
- Issue-chain manifest: explicit `--chains-json`; the usual path is
  `evals/swebench/data/swe_bench_pro_chain_experiment_nodes_deep.jsonl`, the
  same vendored deep chain-nodes JSONL used by
  `run_swebench_pro_memory_chains.py`
- Local run root: `evals/out/swebench_pro`
- Container runner:
  `simple_agent_lab.evals.chain`

## Agent And Model

- Agent selection: `--agent-flavor` controls the chain agent. The default
  `bash` flavor is the plain bash-tool agent; `goal` maps to
  `thread_goal_loop`; `loop` and `pdr` are also supported no-read repo-chain
  flavors.
- Tools: repo-chain flavors intentionally exclude the dedicated `read` tool.
  The baseline tool surface is bash. `--task-tool` adds the task tool to the
  selected chain agent or workflow solver.
- Tooling: the agent now runs inside each SWE-bench Pro instance container,
  through the generic eval backend plus a repo-chain-specific in-container
  runner. The `bash` tool is the normal local in-container bash tool rooted at
  the instance workdir. The host no longer keeps a live agent and no longer
  sends tool commands with `docker exec`.
- Prompt contract: the main agent name, role, system prompt, and per-instance
  task prompt are imported from the original SWE-bench container module. The
  repo-chain state does not inject an extra model-visible repository prompt
  before the first instance.
- Model: read from `.env` / environment variable `OPENAI_MODEL`; `--model`
  only overrides it when explicitly passed.
- Reasoning effort: read from `REASONING_EFFORT` (or legacy
  `OPENAI_REASONING_EFFORT`); `--reasoning-effort` only overrides it when
  explicitly passed.
- API kind: `openai-responses`
- Responses request details: include `reasoning.encrypted_content` in
  Responses requests so encrypted reasoning items are available for replay. Use
  the existing Responses request-extra passthrough for stateful fields such as
  `store` and `previous_response_id`; the runner still does not automatically
  maintain a server-side `previous_response_id` chain.
- Max turns per instance: `250`
- Provider auth slots: `--provider-auth-envs` assigns chains to named
  host env vars such as `OPENAI_AUTH_TOKEN2`. Before launching a container, the
  host copies the selected token value into the container's canonical
  `OPENAI_AUTH_TOKEN`, because the provider is now constructed inside the
  container.

### Goal Flavor In A Chain

The `goal` flavor uses the same `run_thread_goal_loop` inside and outside a
chain; the repo chain only changes where the loop starts and how each steering
message is framed:

- Context inheritance: the loop is seeded with the shared chain `State`
  (the `state=` argument of `run_thread_goal_loop`), so the goal solver resumes
  on top of every earlier instance's accumulated context from its first segment
  instead of starting fresh per instance. Its bash turns are recorded on the
  same chain state, so they carry forward to the next instance.
- Steering preface: a trusted host preface (`CHAIN_GOAL_PREFACE` in
  `simple_agent_lab.evals.chain`) is prepended to every steering message. It
  tells the model this is one long chain of ordered sub-problems, to reuse the
  accumulated context instead of starting over, and to solve only the current
  sub-problem.
- Unchanged mechanics: the goal store, `get_goal`/`update_goal` tools, steering
  body, and completion rules are identical to the standalone `goal` arm. With
  `state` and the preface left at their defaults, the chain goal path reproduces
  the standalone goal behavior exactly.
- Budgets: the goal loop runs a segmented budget (`SAL_WORKFLOW_LOOP_MAX_TURNS`
  outer segments, each up to `SAL_WORKFLOW_WORKER_MAX_TURNS` inner turns). For the
  `goal` flavor the runner maps `--max-turns` onto `SAL_WORKFLOW_WORKER_MAX_TURNS`
  (overriding any `.env` value) and defaults `SAL_WORKFLOW_LOOP_MAX_TURNS` to `1`,
  so one segment runs up to `--max-turns` inner turns; set
  `SAL_WORKFLOW_LOOP_MAX_TURNS` in `.env` to restore multi-segment goal. For the
  non-goal `while`-loop flavors, `--max-turns` is the per-instance turn budget
  directly.
- Shared construction: both the standalone goal arm and the chain goal path
  build the solver and drive the loop through the single `run_goal_flavor`
  helper in `simple_agent_lab.agents.flavors`.

## Compression

- Strategy: configured by `--compression-strategy` (default `none`).
- `none`: the default; no context summarization.
- `summarize`: existing `SummarizeStrategy`.
- Compressor model: same provider/model settings as the main agent
- Compressor agent name: `swebench_compressor`
- Context window assumption: `272000` tokens
- Threshold: `217600` tokens (`272000 * 0.8`)
- Keep recent: `4`
- Preserve kinds: `task`, `system`, `context`
- Only the current instance prompt remains a pinned `task`; when the next
  instance starts, prior instance prompts are demoted to ordinary messages so
  they can be summarized with their solution transcript.
- A summarize trigger can emit up to two folds: one contiguous span before the
  current instance prompt, and one contiguous current-instance early-work span
  after that prompt but before the recent tail. If the current early-work span
  is empty, it is skipped. Each fold stays on one side of the current task.
- Full `trajectory.jsonl` output: enabled by default; disable only with
  `--no-write-trajectories`

## Handoff

Handoff is the default context-window mechanism for no-compression chains. The
runner reacts to the context window *as soon as it is reached* rather than
waiting for the current instance to finish. When the active context reaches
`--context-window-tokens`, the model writes a handoff document immediately; the
chain then either moves on to the next instance (if the current one just
finished) or keeps working on the SAME instance in a fresh window seeded with
that handoff.

- Mid-instance trigger: while an instance solves, the runner checks the active
  context at each turn boundary with
  `estimate_context_tokens(state.active_context_messages())` (see
  `_context_window_abort`). It always lets at least one turn run in the current
  window first, so every window makes progress. Once the estimate reaches
  `--context-window-tokens` (default `217600` = `272000 * 0.8`, the same trigger
  as the summarize threshold so the two arms are compared fairly, and below the
  real `272000` model window so a window's own work fits) and the instance is not
  yet done, the solver stops at that turn boundary and a handoff fires.
- In-place window reset: the handoff document is appended and the *active*
  context is repointed to just the current task plus that document (see
  `_apply_context_window_handoff`). The full transcript stays in
  `state.messages`/`state.events` for the trajectory; only what the model sees
  going forward is reset. The SAME instance then keeps solving in the fresh
  window. `chain_window_index` increments at each reset, and the instance's
  overall turn budget is shared across its windows (a handoff never grants extra
  turns; handoff-generation turns are not charged to the solver budget).
- Boundary trigger: if an instance instead finishes with `status == "ok"` while
  its context is still at/over the window (it completed just as the window
  filled) and it is not the last instance in the chain part, a boundary handoff
  fires: the next instance starts in a fresh chain state whose only
  active-context message is the handoff document. This is the "just completed at
  the boundary -> next problem" case.
- Handoff document: the model is asked one more time (a tool-less turn that
  inherits the current visible context) to write a durable handoff document —
  repository architecture, key files, build/test/run commands and conventions,
  decisions made, and current state (done / in progress / known issues). See
  `CHAIN_HANDOFF_PROMPT` and `CHAIN_HANDOFF_CONTEXT_PREFACE` in
  `simple_agent_lab.evals.chain`.
- Defaults and exclusivity: handoff is on by default. It is disabled when
  `--compression-strategy summarize` (compression owns the window instead), when
  `--no-handoff` is passed, or when `--context-window-tokens` is `0`. With
  `--no-handoff` and `none` compression, the chain runs "naked" with no
  context-window protection.
- Failure safety: if the extra handoff turn fails or returns an empty document,
  its scratch-state prompt and partial events are discarded. The solver keeps
  working on the same instance with the full context and remaining turn budget;
  a failed boundary handoff likewise carries the full context to the next
  instance. `MAX_CONTEXT_WINDOW_HANDOFFS` caps successful resets per instance as
  a backstop against a pathologically small window.
- Metrics: each fired handoff records a `ContextCompressionEvent` with strategy
  `context_window_handoff`, and `result.json` reports `handoff`,
  `handoff_written`, `boundary_handoff_written`, `handoff_context_tokens`, and
  `context_window_handoffs` (the count of mid-instance resets). Per-chain
  `summary.json` and the `[DONE]` log report occurrence counts: total handoffs are
  mid-instance resets plus boundary handoffs.

## Task-Tool Mode

The task-tool variable is selected through the unified repo-chain runner:

```bash
uv run --extra swebench python runs/swebench/run_swebench_pro_repo_chains.py \
  --all \
  --chains-json evals/swebench/data/swe_bench_pro_chain_experiment_nodes_deep.jsonl \
  --task-tool \
  --compression-strategy none \
  --provider-auth-envs OPENAI_AUTH_TOKEN:12,OPENAI_AUTH_TOKEN2:11 \
  --api-kind openai-responses \
  --max-turns 250 \
  --run-official-eval
```

This stages `mode=repo_chain`, the selected `agent_flavor`, and
`task_tool=true` in `input/chain_config.json`. The container builds the selected
agent inside the instance container and gives it bash plus task. Context-window
management defaults to handoff (see "Handoff"): the window is reset as soon as it
is reached, mid-instance if needed, and the instance keeps working in the fresh
window. Task-tool runs always pair with handoff or compression.

## Chain Planning

Rows are matched to the analyzed chain manifest used by the memory-chain
runner. The default vendored flat JSONL orders nodes by `step_index` (falling
back to `commit_time` and `instance_id`); the older nested JSON shape orders by
`commit_time`. Every dataset instance not covered by a raw chain becomes a
length-1 singleton unit, so a full split still runs every selected instance
exactly once. Planned units are ordered longest-first, with same-length ties
sorted deterministically by repo and chain id.

The repo-chain runner does not enable filesystem memory for these units. The
only cross-instance state is still the active repo-chain transcript and
`out/chain_state.json`.

The default parallelism is `--parallel slots`, which means one worker per
provider-auth lane. With the formal `OPENAI_AUTH_TOKEN:12,OPENAI_AUTH_TOKEN2:11`
spec, the runner opens 23 concurrent lanes and queues the longest-first planned
units across them. `--parallel parts` remains accepted as a compatibility alias
that caps workers at the smaller of planned units and declared slots.

## Output Contract

Each instance writes:

- `input/instance.json`
- `input/chain_config.json`
- `input/chain_state.json` when there is prior repo-chain context for this
  planned unit
- `out/result.json`
- `out/chain_state.json`
- `out/trajectory.jsonl` unless `--no-write-trajectories` is set

Trajectories are per instance. The per-chain directory contains summaries and
skip records, not a synthetic per-chain trajectory.

Each `result.json` includes `model_patch`, status, error fields, chain part
metadata, compression metrics, the chain event span for that instance, and the
handoff fields `handoff` (whether handoff was active), `handoff_written`
(whether this instance reset the window at all), `boundary_handoff_written`,
`context_window_handoffs` (how many times it reset the window mid-instance), and
`handoff_context_tokens` (the estimated active context at the last reset). This
means partial runs can still be converted into predictions for completed or
skipped instances by collecting existing `result.json` files.

Repo-chain continuity is an artifact contract, not a host-side live Python
object. For each planned unit, the host passes the previous
`out/chain_state.json` to the next instance as `input/chain_state.json`.
The payload carries the active context messages, task seed, and run metadata
needed to continue the repository chain while keeping the agent loop inside
the current instance container. Compressed-away inactive transcript entries are
not carried forward in this continuation artifact; durable information should
survive through the active summary context. After a mid-instance handoff, the
outgoing `out/chain_state.json` already carries only the current task plus the
latest handoff document as the active context (the dropped transcript stays in
the trajectory but not in the continuation). When a boundary handoff fires
instead, the outgoing `out/chain_state.json` is reset to carry only the model's
handoff document (plus chain metadata), so the next window deliberately starts
without the prior transcript.

The repo-chain runner also refreshes `<run_id>_predictions.jsonl` after each
completed instance. Every refresh is bound to the current plan: unexpected
instance directories are rejected, and planned instances without a current
result receive an empty patch. The file therefore keeps the full experimental
denominator while the run is active and after all workers finish.

Batch-level outputs include:

- `experiment.json`
- `instances.jsonl`
- `<run_id>_predictions.jsonl`
- `skipped_instances.jsonl` when any instances are skipped
- `_repo_chains/<chain>/summary.json`

## Invalid Prompt Handling

If the provider raises `invalid_prompt` or code `-4321`, the in-container
repo-chain runner applies the same recovery policy to every agent flavor,
including `goal`, and classifies the latest relevant active user-visible
message. Goal-loop steering messages are skipped during classification so the
runner reaches the current instance prompt or tool output that caused the
provider rejection:

- If it is the current instance prompt, the instance is skipped and that prompt
  is dropped from active context so the next instance can proceed.
- If it is a tool output, the latest connected tool-call/tool-result exchange
  is removed from active context and replaced with a short compressible message note:
  `Removed invalid_prompt-triggering tool call/output. Use another command.`
  The model request is then retried.
- Tool-output invalid-prompt rewrites are capped at 20 per instance.
- Invalid-prompt retries share the same 250-turn per-instance budget; failed
  invalid-prompt attempts do not reset the turn budget.
- If retries exhaust the retry cap or turn budget, the instance is skipped and
  the active context is cleared so the next instance starts from an empty
  repo-chain context.
- Before each model request, the runner repairs active context by dropping any
  orphan tool call or tool result left by prior context surgery. This protects
  OpenAI Responses requests from `No tool output found for function call ...`
  errors caused by incomplete tool pairs.

Skipped instances are recorded in their `result.json` and in
`skipped_instances.jsonl`.

## Recommended Formal Command

Set `OPENAI_MODEL`, `OPENAI_AUTH_TOKEN`, `OPENAI_AUTH_TOKEN2`, and
`REASONING_EFFORT` in `.env` or the process environment before launching. Do
not pass `--model` or `--reasoning-effort` unless intentionally overriding
those values for one run. The in-container path installs the current
`simple-agent-lab` wheel into every instance container; use
`--prepare-wheelhouse` the first time or after dependency changes to refresh the
offline wheelhouse.

```bash
uv run --extra swebench python runs/swebench/run_swebench_pro_repo_chains.py \
  --all \
  --chains-json evals/swebench/data/swe_bench_pro_chain_experiment_nodes_deep.jsonl \
  --provider-auth-envs OPENAI_AUTH_TOKEN:12,OPENAI_AUTH_TOKEN2:11 \
  --api-kind openai-responses \
  --max-turns 250 \
  --run-official-eval
```

This default command runs the `bash` flavor with no chain compression
(`--compression-strategy none`). Add `--agent-flavor goal` for the goal-loop
baseline, `--agent-flavor goal --compression-strategy summarize` for the
goal compression baseline, or `--agent-flavor goal --task-tool` for the goal
task-delegation baseline. For the `goal` flavor, `--max-turns` sets the goal
worker budget (`SAL_WORKFLOW_WORKER_MAX_TURNS`) and the outer loop defaults to a
single segment (`SAL_WORKFLOW_LOOP_MAX_TURNS=1`); see "Goal Flavor In A Chain".

Full trajectories are written by default. Pass `--no-write-trajectories` only
when storage cost is unacceptable; long repo chains can grow to hundreds of GB
because every model request payload can include a large repo-chain context.

## Verification Status

Completed:

- A lightweight single-instance smoke wrote `result.json` and converted it to a
  predictions JSONL without writing full trajectories
  (`smoke-responses-20260628-quick`).
- Unit tests cover planning, long-tail splitting, per-instance trajectory
  default/disable, transactional handoff failure, handoff occurrence counts,
  invalid-prompt classification, invalid-prompt context edits, wall-time
  forwarding, and turn-budget accounting for invalid-prompt retries.
- A deterministic low-threshold Responses smoke forced one `summarize`
  compression and verified the adapter could continue with the compressed
  active context.
- A single-instance end-to-end smoke ran the official SWE-bench Pro evaluator
  path and wrote one eval result (`e2e-responses-20260628-quick`). The instance
  was unresolved as expected for `max-turns=1`; the path, not score, was the
  smoke signal.
- A previous full run was launched as
  `pro-repo-chain-summarize-responses-20260628-020217` with 23 chain parts,
  `--parallel parts`, `openai-responses`, and full trajectories disabled; it was
  later paused, so do not treat it as an active run.

## Paused Run Review Notes

The paused run
`evals/out/swebench_pro/pro-repo-chain-summarize-responses-20260628-020217` completed
338 of 731 planned instances before it was stopped. It wrote 317 ok results,
13 error results, and 8 skipped results; 23 additional instances had staged
`input/instance.json` files but no `result.json`, matching the 23 interrupted
workers.

Issues found from the paused run and code review:

- Instance prompts were appended as generic `message` entries, so the
  compression preserve list did not pin the current problem statement. They are
  now appended as `task` messages.
- The paused run used `preserve_kinds=["task", "system", "context"]`, which
  allowed prior summaries to be summarized again. The current runner now keeps
  that cascading-summary behavior intentionally.
- The old invalid-prompt handler rewrote only the latest tool result. In
  qutebrowser part 1 it produced 164 rewrite events and 8 skipped instances
  after retrying the same class of prompt 20 times.
- Two qutebrowser instances failed with `No tool output found for function
  call ...`, consistent with active context containing an incomplete tool pair.
- The log contained 20,411 `429` retry lines. Treat provider resource shortage
  as run-level backpressure when selecting concurrency/auth-token slots rather
  than as a task-quality signal.
- Full trajectories were disabled, so the exact prompt payload cannot be
  reconstructed after interruption. Per-instance results and partial
  predictions are durable; exact repo-chain resume is not.
- Some generated or dependency paths appeared in model patches, including
  `.pb.go` and `vendor/` files. The current ignore approach filters only what
  Git ignore rules can filter; tracked or force-staged generated files need a
  separate patch-filtering decision.
- The Responses adapter supports `store` and `previous_response_id` passthrough,
  but the repo-chain runner does not automatically maintain a server-side
  response chain. Current runs are stateless replay runs unless explicit
  request extras are supplied.

Still useful while a formal run is active:

- Monitor `evals/out/swebench_pro/pro-repo-chain-summarize-responses-20260628-020217.log`
  for `[DONE]`, `[FAIL]`, compression events, skipped instances, and final
  predictions/eval output.
