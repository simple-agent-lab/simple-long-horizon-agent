# Configuration Reference: Environment Variables

One place to see every environment variable the project reads, grouped by the
layer that owns it. This is a *discoverability* surface — the code is still the
source of truth for behavior. Each variable's **name constant** is declared in
exactly one module (the "Owner" column); other call sites import it rather than
re-declaring the string. See ADR `consolidate-provider-env` for why.

## Launching a run: the JSON run-profile (`--profile`)

To launch one agent-on-a-bench arm from a single committed file instead of a
remembered mix of exports and flags, pass `--profile PATH` to a run entry
(`runs/run_swebench_suite.py`, `runs/run_programbench_suite.py`). A profile is a
small JSON document with two sections (see ADR `run-profile-file`):

```json
{
  "env": { "AGENT_FLAVOR": "pdr", "SWE_PDR_WIDTH": "3" },
  "run": { "max-turns": 200, "network-mode": "host", "prepare-wheelhouse": true }
}
```

- `env` → the catalogued env names below, applied **fill-the-gaps** (a real
  export still wins, exactly like `.env`).
- `run` → `run_*_suite.py` long-option names (without `--`), injected as
  defaults that an explicit CLI flag overrides.

It is a *bundle* of the two existing surfaces (`.env` + CLI), not a new schema,
so there is no second source of truth. Keep secrets in `.env`; commit only
`runs/profiles/*.example.json` (the rest are gitignored). See
`runs/profiles/swebench-pdr.example.json`.

## Boundary rule (where to declare a new env var)

Env-var names live with the layer that owns the concern, in a *light* module so
host harnesses can forward names into a container without importing a heavy
graph:

- Provider / credentials / reasoning → `src/simple_agent_lab/llm/env.py`.
- Named model aliases (`<ALIAS>_*`) → `src/simple_agent_lab/llm/config.py`
  (`ModelRegistry`).
- Model pricing / context-window overrides → `src/simple_agent_lab/model_metadata.py`.
- Agent selection + agent-level knobs (flavor, compression) →
  `src/simple_agent_lab/agent_flavors.py` (light names module; reading and
  default values for the workflow/compression knobs live in
  `src/simple_agent_lab/agents/flavors.py`, which imports the names back).
- Suite-specific knobs → that suite's container module under
  `src/simple_agent_lab/evals/suites/<suite>/`.

Do **not** centralize all names into one module — that would couple unrelated
layers and break the inward dependency that keeps `llm/env.py` clean. Add a row
to this table instead.

## Provider / credentials

Owner: `src/simple_agent_lab/llm/env.py` (single source of truth).

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_MODEL` | required | Model id under test. |
| `OPENAI_AUTH_TOKEN` | required | Gateway bearer token (NOT `OPENAI_API_KEY`). |
| `OPENAI_BASE_URL` | unset | Endpoint override; must agree with `API_KIND`. |
| `API_KIND` | `openai-chat` | Wire adapter: `openai-chat`, `openai-responses`, `anthropic-messages`. |
| `OPENAI_SESSION_ID` | unset | Optional gateway trace header. |
| `OPENAI_LOG_ID` | unset | Optional gateway trace header (`X-TT-logid`). |
| `REASONING_EFFORT` | unset | Provider-agnostic reasoning depth. |
| `OPENAI_REASONING_EFFORT` | unset | Legacy alias for `REASONING_EFFORT`. |
| `JUDGE_MODEL` | falls back to `OPENAI_MODEL` | Eval-grader model. |
| `JUDGE_AUTH_TOKEN` | falls back to `OPENAI_AUTH_TOKEN` | Eval-grader token. |
| `JUDGE_BASE_URL` | falls back to `OPENAI_BASE_URL` | Eval-grader endpoint. |
| `JUDGE_API_KIND` | falls back to `API_KIND` | Eval-grader wire adapter. |

Note: the eval harnesses (`evals/swebench`, `evals/programbench`,
`evals/onemillion`) accept only `openai-chat` / `openai-responses` via their own
`API_KIND_CHOICES` tuple — an intentional narrower set than `llm.env`'s.

## Named model aliases (`ModelRegistry`)

Owner: `src/simple_agent_lab/llm/config.py`. See ADR `model-alias-registry`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_CONFIG` | unset | Path to a `models.json` mapping aliases (`strong`/`fast`) to provider specs. |
| `MODEL_CONFIG_JSON` | unset | Same schema inline, for sandboxes where writing a file is awkward. |
| `<ALIAS>_*` | falls back to `OPENAI_*` | Per-alias provider env (e.g. `STRONG_MODEL`); single-model setups collapse onto `OPENAI_*`. |

## Model pricing / context windows

Owner: `src/simple_agent_lab/model_metadata.py`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `SIMPLE_AGENT_LAB_PRICE_BOOK` | built-in table | Path to a JSON model→rate override, merged over the built-in price book. |
| `SIMPLE_AGENT_LAB_CONTEXT_WINDOW_BOOK` | built-in table | Path to model metadata with context windows (LiteLLM / models.dev formats). |

## Agent selection + agent-level knobs

Owner (names): `src/simple_agent_lab/agent_flavors.py`. Reading + defaults for
the workflow/compression knobs: `src/simple_agent_lab/agents/flavors.py`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `AGENT_FLAVOR` | `bash` | Agent flavor: `bash`, `bash_task`, `bash_task_read`, `bash_skills`, `loop`, `pdr`. |
| `SAL_AGENT_COMPRESSION_THRESHOLD_TOKENS` | `window * ratio`, else `80000` | Token threshold that triggers context compression. |
| `SAL_AGENT_COMPRESSION_WINDOW_RATIO` | `0.8` | Fraction of the context window used as the threshold when no explicit threshold is set. |
| `SAL_AGENT_COMPRESSION_KEEP_RECENT` | `4` | Recent turns kept verbatim during compression. |
| `SWE_PDR_ROUNDS` | `2` | PDR workflow: distill/refine rounds. |
| `SWE_PDR_WIDTH` | `3` | PDR workflow: parallel attempts per round. |
| `SWE_PDR_ATTEMPT_TURNS` | = `SWE_WORKER_MAX_TURNS` | PDR workflow: per-attempt turn budget (cost guard). |
| `SWE_LOOP_MAX_TURNS` | `6` | Loop workflow: judge-gated outer iterations. |
| `SWE_WORKER_MAX_TURNS` | `40` | Per-worker inner turn budget. |

## Suite-specific knobs

| Variable | Owner | Default | Purpose |
| --- | --- | --- | --- |
| `SWE_REPO_LANGUAGE` | `evals/suites/swebench/container.py` | unset | Repo language hint for the SWE-bench container. |
| `PROGRAMBENCH_REQUIRE_NET_ISOLATION` | `evals/suites/programbench/container.py` | on | Require per-command network isolation (`unshare --net`). |
| `OMB_WORKFLOW` | `evals/suites/onemillion/workflow_container.py` | `single` | OneMillion workflow selector. |
| `OMB_REFLECTION_ROUNDS` | same | — | Reflection rounds. |
| `OMB_PARALLEL_WORKERS` | same | — | Parallel workers. |
| `OMB_PDR_ROUNDS` | same | — | PDR rounds for the OneMillion arm. |
| `OMB_PDR_WIDTH` | same | — | PDR width for the OneMillion arm. |
| `OMB_TIMEOUT` | same | `600.0`s | Per-request timeout for every sub-agent. |
| `MCP_CONFIG` | `evals/swebench/harness.py` | unset | MCP server config forwarded into the container. |

## Memory

Owner: `src/simple_agent_lab/evals/protocols.py`. See `docs/agent-native/memory.md`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `SAL_MEMORY_HOME` | unset (memory off) | Filesystem memory root; presence opts the run into memory. |
| `SAL_MEMORY_NAME` | unset | Memory namespace. |
| `SAL_MEMORY_RUN_ID` | unset | Run id scoping memory artifacts. |

## Trace

| Variable | Owner | Default | Purpose |
| --- | --- | --- | --- |
| `LIVE_TRACE_PATH` | `src/simple_agent_lab/trace/live.py` | unset | Bind-mounted path for incremental live trace output. |
| `E2E_TRACE_PATH` | `tests/e2e/test_live_openai_responses_e2e.py` | unset | Live e2e test trace output path (test-only). |
