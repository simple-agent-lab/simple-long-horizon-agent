# Configuration Reference: Environment Variables

One place to see every environment variable the project reads, grouped by the
layer that owns it. This is a *discoverability* surface — the code is still the
source of truth for behavior. Each variable's **name constant** is declared in
exactly one module (the "Owner" column); other call sites import it rather than
re-declaring the string. See ADR `consolidate-provider-env` for why.

## Launching a run: the JSON run-profile (`--profile`)

To launch one agent-on-a-bench arm from a single committed file instead of a
remembered mix of exports and flags, pass `--profile PATH` to any bench through
the unified entry (`runs/run_bench.py <bench> --profile PATH`; the per-bench
modules live in `runs/_benches/`). A profile is a small JSON document with two
sections (see ADR `run-profile-file`):

```json
{
  "env": { "AGENT_FLAVOR": "pdr", "SAL_WORKFLOW_PDR_WIDTH": "3" },
  "run": { "max-turns": 200, "network-mode": "host", "prepare-wheelhouse": true }
}
```

- `env` → the catalogued env names below, applied **fill-the-gaps** (a real
  export still wins, exactly like `.env`).
- `run` → `run_*_suite.py` long-option names (without `--`), injected as
  defaults that an explicit CLI flag overrides.

It is a *bundle* of the two existing surfaces (`.env` + CLI), not a new schema,
so there is no second source of truth. Keep secrets in `.env`; commit only
`runs/profiles/*.example.json` (the rest are gitignored). Each benchmark ships a
default example to copy and edit: `swebench-pdr.example.json`,
`programbench.example.json`, `onemillion.example.json`, and
`onemillion-workflow.example.json`.

## One entry point for every bench (`run_bench.py`)

`runs/run_bench.py` is a single dispatcher over all benches, built so a thin
dashboard can drive everything by shelling out and reading JSON. The per-bench
`runs/run_<bench>.py` scripts stay directly runnable (they are imported by the
dispatcher); `run_bench.py` is the one surface to learn:

```bash
uv run python runs/run_bench.py list [--json]            # discover benches
uv run python runs/run_bench.py setup [bench ...] [--oracle] [--json]
uv run python runs/run_bench.py <bench> [bench args ...] [--json]
uv run python runs/run_bench.py all --manifest M.json [--parallel N]
```

- `list` — registered benches (name, description, whether Docker is needed).
- `setup` — probe the environment (Python/uv, `.env` + provider creds, Docker
  daemon, datasets) and report per-bench readiness; `--oracle` additionally runs
  a model-free oracle smoke where a bench supports it (OneMillion). A fast "is my
  environment wired correctly?" check before launching real runs.
- `<bench>` — delegate to that bench's own parser (same flags, incl. `--profile`);
  with `--json` it prints one result object to stdout (human logs go to stderr).
- `all` — run every entry of a JSON manifest, each as an isolated subprocess, and
  print a combined JSON summary. See `runs/bench-manifest.example.json`; a
  manifest entry is `{bench, args:[...]}` (plus optional `profile`), so the
  dashboard composes one click from the per-bench profiles above.

## Boundary rule (where to declare a new env var)

Env-var names live with the layer that owns the concern, in a *light* module so
host harnesses can forward names into a container without importing a heavy
graph:

- Provider / credentials / reasoning → `src/simple_agent_lab/llm/env.py`.
- Named model aliases (`<ALIAS>_*`) → `src/simple_agent_lab/llm/config.py`
  (`ModelRegistry`).
- Model pricing / context-window overrides → `src/simple_agent_lab/model_metadata.py`.
- Agent selection (flavor) → `src/simple_agent_lab/agent_flavors.py`. The
  agent-level workflow/compression knobs are registry-backed (declared in
  `src/simple_agent_lab/config.py`; see the generated table below).
- Suite-specific knobs not yet in the registry → that suite's container module
  under `src/simple_agent_lab/evals/suites/<suite>/`.

Do **not** centralize all names into one module *if doing so couples unrelated
layers* — that would break the inward dependency that keeps `llm/env.py` clean.

> Direction (ADR `centralized-env-config`, in progress): env knobs are moving
> into one declarative registry, `src/simple_agent_lab/config.py`. It is a
> FOUNDATION-zone leaf that *declares* each name (with default/parser/group)
> and imports nothing internal, so it centralizes without the coupling this
> rule guards against — layers depend on `config`, never the reverse. Several
> groups are migrated already (see the generated table below); until a knob
> moves, its name still lives with the layer below.

## Registry-backed configuration

These knobs live in the central registry (`src/simple_agent_lab/config.py`); the
table below is generated from `REGISTRY` by `scripts/build_config_reference.py`
and validated in CI, so it cannot drift. Groups follow the `domain.subsystem`
hierarchy. Knobs not yet migrated stay in the hand-written sections that follow.

<!-- BEGIN GENERATED: config-registry (scripts/build_config_reference.py) -->
<!-- Generated from simple_agent_lab.config.REGISTRY — do not edit by hand; run scripts/build_config_reference.py. -->

### `agent.compression`

| Variable | Default | Purpose |
| --- | --- | --- |
| `SAL_AGENT_COMPRESSION_THRESHOLD_TOKENS` | unset | Token threshold that triggers compression; default is window * ratio, else a fixed fallback. |
| `SAL_AGENT_COMPRESSION_WINDOW_RATIO` | `0.8` | Fraction of the context window used as the threshold when none is set. |
| `SAL_AGENT_COMPRESSION_KEEP_RECENT` | `4` | Recent turns kept verbatim during compression. |

### `agent.workflow`

| Variable | Default | Purpose |
| --- | --- | --- |
| `SAL_WORKFLOW_WORKER_MAX_TURNS` | `40` | Per-worker inner turn budget. |
| `SAL_WORKFLOW_LOOP_MAX_TURNS` | `6` | Loop workflow: judge-gated outer iterations. |
| `SAL_WORKFLOW_PDR_ROUNDS` | `2` | PDR workflow: distill/refine rounds. |
| `SAL_WORKFLOW_PDR_WIDTH` | `3` | PDR workflow: parallel attempts per round. |
| `SAL_WORKFLOW_PDR_ATTEMPT_TURNS` | unset | PDR workflow: per-attempt turn budget; defaults to SAL_WORKFLOW_WORKER_MAX_TURNS. |

### `eval.onemillion`

| Variable | Default | Purpose |
| --- | --- | --- |
| `OMB_WORKFLOW` | `single` | OneMillion workflow selector. |
| `OMB_REFLECTION_ROUNDS` | `2` | Reflection rounds. |
| `OMB_PARALLEL_WORKERS` | `3` | Parallel workers. |
| `OMB_PDR_ROUNDS` | `2` | PDR rounds. |
| `OMB_PDR_WIDTH` | `3` | PDR width. |
| `OMB_TIMEOUT` | `600.0` | Per-request timeout for every sub-agent (seconds). |

### `eval.swebench`

| Variable | Default | Purpose |
| --- | --- | --- |
| `SWE_REPO_LANGUAGE` | `python` | Repo language hint for the SWE-bench container. |

### `trace`

| Variable | Default | Purpose |
| --- | --- | --- |
| `LIVE_TRACE_PATH` | unset | Bind-mounted path for incremental live trace output (unset = off). |

<!-- END GENERATED: config-registry -->

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

## Agent selection

Owner: `src/simple_agent_lab/agent_flavors.py`. (The agent compression +
workflow knobs are registry-backed — see the generated table above.)

| Variable | Default | Purpose |
| --- | --- | --- |
| `AGENT_FLAVOR` | `bash` | Agent flavor: `bash`, `bash_task`, `bash_task_read`, `bash_skills`, `loop`, `pdr`. |

## Suite-specific knobs

The SWE-bench and OneMillion suite knobs are registry-backed (see the generated
table above). This bespoke default-closed bool stays with its suite:

| Variable | Owner | Default | Purpose |
| --- | --- | --- | --- |
| `PROGRAMBENCH_REQUIRE_NET_ISOLATION` | `src/simple_agent_lab/evals/suites/programbench/container.py` | on | Require per-command network isolation (`unshare --net`). |

## Memory

Owner: `src/simple_agent_lab/evals/protocols.py`. See `docs/agent-native/memory.md`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `SAL_MEMORY_HOME` | unset (memory off) | Filesystem memory root; presence opts the run into memory. |
| `SAL_MEMORY_NAME` | unset | Memory namespace. |
| `SAL_MEMORY_RUN_ID` | unset | Run id scoping memory artifacts. |

## Trace

`LIVE_TRACE_PATH` is registry-backed (see the generated table above). This
test-only var is not part of the runtime registry:

| Variable | Owner | Default | Purpose |
| --- | --- | --- | --- |
| `E2E_TRACE_PATH` | `tests/e2e/test_live_openai_responses_e2e.py` | unset | Live e2e test trace output path (test-only). |
