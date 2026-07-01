# SWE-bench Pro Repo-Session Part-Level Goal Experiment

This note records the intended SWE-bench Pro experiment configuration and the
current runner contract. It is an operator handoff for the repo-session
part-level goal runs; the source of truth for executable behavior is
`runs/swebench/run_swebench_pro_repo_sessions.py`,
`simple_agent_lab.evals.suites.swebench.repo_session_runner`,
`simple_agent_lab.evals.suites.swebench.repo_session_state`, and
`evals/swebench/pro_repo_session.py`.

## Design Target

`runs/swebench/run_swebench_pro_repo_sessions.py` is the single SWE-bench Pro
part-level goal runner. One planned session part is one goal: it keeps running
that part's ordered instances, carrying `out/session_state.json` forward, until
the part is exhausted. Baselines that change agent topology or context handling
are configuration modes of this runner, not separate host scripts.

The supported comparison modes are:

- `--agent-flavor bash --compression-strategy summarize`: the default
  repo-session compression run.
- `--agent-flavor bash_task --compression-strategy none`: the no-compression
  chain-task baseline. This stages `mode=chain_task` for the in-container
  runner and enables the context-window restart metadata.

When `--run-id` is omitted, the runner derives a timestamped prefix from the
selected mode: `pro-repo-summarize-*` for the default compression run and
`pro-repo-chain-task-*` for the chain-task baseline.

## Objective

Evaluate Simple Agent Lab long repo sessions on SWE-bench Pro by running each
planned repository part as a long-lived agent goal. Instances within a
repository are ordered by `base_commit` commit timestamp so earlier repository
context can carry into later tasks. The default mode studies summarization
compression; the `bash_task`/`none` mode keeps the same part-level goal
structure without summarization.

## Container-Inside Migration Checklist

- Add a generic eval hook for custom in-container runner modules so the Pro
  repo-session path can use `run_suite_instance` without forking Docker backend
  logic.
- Move repo-session execution into a wheel-shipped container runner that restores
  session state, builds the local in-container agent, runs the current instance,
  writes `out/result.json`, and emits the next `out/session_state.json`.
- Serialize continuation as an artifact contract (`input/session_state.json` to
  `out/session_state.json`) instead of keeping a host-side live `State`.
- Keep host scripts focused on planning, provider-slot assignment, staging
  config/state artifacts, launching `run_suite_instance`, collecting results,
  and refreshing predictions.
- Fold the chain-task baseline into the same host runner with
  `mode=chain_task`, `bash_task`, no compression, and context-window restart
  metadata.
- Preserve auth-slot scheduling by copying the selected host token into the
  container's canonical `OPENAI_AUTH_TOKEN` while recording the original slot in
  result metadata.
- Cover the new contract with unit tests for runner-module selection,
  session-state round trip, in-container runner smoke, staged config payloads,
  and prediction refresh.

## Dataset

- Dataset: `ScaleAI/SWE-bench_Pro`
- Split: `test`
- Local run root: `evals/out/swebench_pro`
- Commit ordering cache: `evals/out/swebench_pro/repo-cache`
- Container runner:
  `simple_agent_lab.evals.suites.swebench.repo_session_runner`

## Agent And Model

- Main agent flavor: configured by `--agent-flavor`; current comparison modes
  are `bash` and `bash_task`.
- Tooling: the agent now runs inside each SWE-bench Pro instance container,
  through the generic eval backend plus a repo-session-specific in-container
  runner. The `bash` tool is the normal local in-container bash tool rooted at
  the instance workdir. The host no longer keeps a live agent and no longer
  sends tool commands with `docker exec`.
- Prompt contract: the main agent name, role, system prompt, and per-instance
  task prompt are imported from the original SWE-bench container module. The
  repo-session state does not inject an extra model-visible repository prompt
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
- Provider auth slots: `--provider-auth-envs` still assigns sessions to named
  host env vars such as `OPENAI_AUTH_TOKEN2`. Before launching a container, the
  host copies the selected token value into the container's canonical
  `OPENAI_AUTH_TOKEN`, because the provider is now constructed inside the
  container.

## Compression

- Strategy: configured by `--compression-strategy`.
- `summarize`: existing `SummarizeStrategy`.
- `none`: no context summarization. With `bash_task`, this is the chain-task
  baseline mode.
- Compressor model: same provider/model settings as the main agent
- Compressor agent name: `swebench_compressor`
- Context window assumption: `272000` tokens
- Threshold: `217600` tokens (`272000 * 0.8`)
- Keep recent: `12`
- Preserve kinds: `task`, `system`, `context`, `summary`
- Full `trajectory.jsonl` output: disabled by default; enable only with
  `--write-trajectories`

## Chain-Task Mode

The chain-task baseline is selected through the unified repo-session runner:

```bash
uv run --extra swebench python runs/swebench/run_swebench_pro_repo_sessions.py \
  --all \
  --agent-flavor bash_task \
  --compression-strategy none \
  --max-context-restarts-per-instance 1 \
  --parallel parts \
  --provider-auth-envs OPENAI_AUTH_TOKEN:12,OPENAI_AUTH_TOKEN2:11 \
  --api-kind openai-responses \
  --max-turns 250 \
  --run-official-eval
```

This stages `mode=chain_task` in `input/repo_session_config.json`. The
container builds the `bash_task` flavor inside the instance container, disables
compression, and keeps the existing context-window fallback: if the parent or
task worker hits a context-window error, the runner opens a fresh chain window
and retries the current instance up to `--max-context-restarts-per-instance`.

## Session Planning

Rows are grouped by `repo`, then sorted inside each repo by commit timestamp.
After sorting, long repos are split into contiguous session parts:

- More than 80 instances: 3 parts
- More than 50 instances: 2 parts
- 50 or fewer instances: 1 part

The default parallelism is `--parallel parts`, which means one worker per
planned session part. `--parallel repos` is kept as a compatibility alias for
the same behavior.

For the current locally cached Pro dataset manifest, the full run has:

- Repositories: 11
- Instances: 731
- Planned session parts / default parallel workers: 23

Per-repo counts used for the 23-part plan:

- `NodeBB/NodeBB`: 44 instances, 1 part
- `ansible/ansible`: 96 instances, 3 parts
- `element-hq/element-web`: 56 instances, 2 parts
- `flipt-io/flipt`: 85 instances, 3 parts
- `future-architect/vuls`: 62 instances, 2 parts
- `gravitational/teleport`: 76 instances, 2 parts
- `internetarchive/openlibrary`: 91 instances, 3 parts
- `navidrome/navidrome`: 57 instances, 2 parts
- `protonmail/webclients`: 65 instances, 2 parts
- `qutebrowser/qutebrowser`: 79 instances, 2 parts
- `tutao/tutanota`: 20 instances, 1 part

## Output Contract

Each instance writes:

- `input/instance.json`
- `input/repo_session_config.json`
- `input/session_state.json` when there is prior repo-session context for this
  session part
- `out/result.json`
- `out/session_state.json`
- `out/trajectory.jsonl` only when `--write-trajectories` is set

Each `result.json` includes `model_patch`, status, error fields, session part
metadata, compression metrics, and the session event span for that instance.
This means partial runs can still be converted into predictions for completed
or skipped instances by collecting existing `result.json` files.

Repo-session continuity is an artifact contract, not a host-side live Python
object. For each session part, the host passes the previous
`out/session_state.json` to the next instance as `input/session_state.json`.
The payload carries the active context messages, task seed, and run metadata
needed to continue the repository session while keeping the agent loop inside
the current instance container. Compressed-away inactive transcript entries are
not carried forward in this continuation artifact; durable information should
survive through the active summary context.

The repo-session runner also refreshes `<run_id>_predictions.jsonl` after each
completed instance. This file is therefore a partial prediction snapshot while
the run is active and a final prediction file after all workers finish.

Batch-level outputs include:

- `experiment.json`
- `instances.jsonl`
- `<run_id>_predictions.jsonl`
- `skipped_instances.jsonl` when any instances are skipped
- `_repo_sessions/<session>/summary.json`

## Invalid Prompt Handling

If the provider raises `invalid_prompt` or code `-4321`, the in-container
repo-session runner classifies the latest active user-visible message:

- If it is the current instance prompt, the instance is skipped and that prompt
  is dropped from active context so the next instance can proceed.
- If it is a tool output, the latest connected tool-call/tool-result exchange
  is removed from active context and replaced with a short context note:
  `刚刚的工具调用及其输出会触发 invalid_prompt，已从上下文移除。请使用其他命令继续。`
  The model request is then retried.
- Tool-output invalid-prompt rewrites are capped at 20 per instance.
- Invalid-prompt retries share the same 250-turn per-instance budget; failed
  invalid-prompt attempts do not reset the turn budget.
- If retries exhaust the retry cap or turn budget, the instance is skipped and
  the active context is cleared so the next instance starts from an empty
  repo-session context.
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
uv run --extra swebench python runs/swebench/run_swebench_pro_repo_sessions.py \
  --all \
  --parallel parts \
  --provider-auth-envs OPENAI_AUTH_TOKEN:12,OPENAI_AUTH_TOKEN2:11 \
  --api-kind openai-responses \
  --max-turns 250 \
  --run-official-eval
```

Do not pass `--write-trajectories` for the formal run unless debugging a small
subset. Full trajectories can grow to hundreds of GB because every model
request payload can include a large repo-session context.

## Verification Status

Completed:

- A lightweight single-instance smoke wrote `result.json` and converted it to a
  predictions JSONL without writing full trajectories
  (`smoke-responses-20260628-quick`).
- Unit tests cover planning, long-tail splitting, trajectory opt-in,
  invalid-prompt classification, invalid-prompt context edits, and turn-budget
  accounting for invalid-prompt retries.
- A deterministic low-threshold Responses smoke forced one `summarize`
  compression and verified the adapter could continue with the compressed
  active context.
- A single-instance end-to-end smoke ran the official SWE-bench Pro evaluator
  path and wrote one eval result (`e2e-responses-20260628-quick`). The instance
  was unresolved as expected for `max-turns=1`; the path, not score, was the
  smoke signal.
- A previous full run was launched as
  `pro-repo-summarize-responses-20260628-020217` with 23 session parts,
  `--parallel parts`, `openai-responses`, and full trajectories disabled; it was
  later paused, so do not treat it as an active run.

## Paused Run Review Notes

The paused run
`evals/out/swebench_pro/pro-repo-summarize-responses-20260628-020217` completed
338 of 731 planned instances before it was stopped. It wrote 317 ok results,
13 error results, and 8 skipped results; 23 additional instances had staged
`input/instance.json` files but no `result.json`, matching the 23 interrupted
workers.

Issues found from the paused run and code review:

- Instance prompts were appended as generic `message` entries, so the
  compression preserve list did not pin the current problem statement. They are
  now appended as `task` messages.
- The paused run used `preserve_kinds=["task", "system", "context"]`, which
  allowed prior summaries to be summarized again. The current runner reuses the
  repo-session default preserve list, including `summary`, so formal runs keep
  prior summaries verbatim.
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
  predictions are durable; exact repo-session resume is not.
- Some generated or dependency paths appeared in model patches, including
  `.pb.go` and `vendor/` files. The current ignore approach filters only what
  Git ignore rules can filter; tracked or force-staged generated files need a
  separate patch-filtering decision.
- The Responses adapter supports `store` and `previous_response_id` passthrough,
  but the repo-session runner does not automatically maintain a server-side
  response chain. Current runs are stateless replay runs unless explicit
  request extras are supplied.

Still useful while a formal run is active:

- Monitor `evals/out/swebench_pro/pro-repo-summarize-responses-20260628-020217.log`
  for `[DONE]`, `[FAIL]`, compression events, skipped instances, and final
  predictions/eval output.
