# Harbor Eval Integration

The `harbor` bench runs Simple Agent Lab agents through Harbor's own harness.
Harbor resolves datasets and tasks, starts the task environment, runs the
verifier, downloads logs/artifacts, and writes the canonical job `result.json`.

SAL supplies one Harbor installed agent:

```text
simple_agent_lab.evals.harbor.agent:SimpleAgentLabHarborAgent
```

That installed agent starts a SAL runner inside the Harbor task environment.
The SAL runner then builds the normal SAL agent and executes bash/read/task
tools locally in that same container.

When the adapter is imported from a local Simple Agent Lab checkout, it uploads a
minimal source archive (`pyproject.toml`, root metadata files, and `src/`) into
the Harbor task environment and installs that source into the in-container venv.
If no local checkout is available, it falls back to the configurable
`sal_package` value.

The adapter first reuses an existing working SAL venv when one is present. If it
needs to install, it prefers the task image's own Python 3.10+ plus `venv`/`pip`
so benchmark images that already include Python do not need a separate `uv`
bootstrap. Only when no suitable system Python path is available does it fall
back to installing `uv` and using the configured `python_version`.

There is no `harbor_exec` tool and no fallback where a host-side SAL agent
forwards individual tool commands into the Harbor container.

## Boundary

This integration treats Harbor as the dataset and environment harness, not as a
single dataset source to translate into SAL suites.

Harbor owns:

- dataset/task resolution (`--dataset`, `--path`, `--repo`, `--task`);
- task environment lifecycle and network/resource policy;
- verifier execution;
- artifact download;
- job-level `result.json` aggregation.

SAL owns:

- the installed-agent adapter Harbor imports;
- the container-local SAL runner;
- SAL bash/read/task tool behavior after the runner starts;
- SAL debug artifacts under Harbor agent logs.

The first target slice is Harbor datasets that fit SAL's current code and
terminal/tool-use agent shape. Do not add a SAL-level Windows exclusion here;
Harbor task and agent compatibility checks remain the authority for whether a
particular dataset can run.

The `--agent-env KEY=${KEY}` entries in dry-run output are intentional. Harbor
resolves those templates from the host process environment before constructing
the agent, so dry-run JSON does not expose secret values while real runs still
receive the actual credentials.

## Install

Harbor is optional and should be installed in a Python version that satisfies
Harbor's requirements:

```bash
uv sync --extra harbor
```

## Dry Run

Use `--dry-run` to inspect the Harbor command without requiring Harbor, Docker,
or model credentials:

```bash
uv run python runs/run_bench.py harbor \
  --dataset demo \
  --n-tasks 1 \
  --dry-run \
  --json
```

The JSON result includes a `command` array beginning with:

```text
harbor run --dataset demo --n-tasks 1 --agent simple_agent_lab.evals.harbor.agent:SimpleAgentLabHarborAgent
```

## Real Run

For a Harbor registry dataset:

```bash
uv run python runs/run_bench.py harbor \
  --dataset <dataset-name> \
  --n-tasks 1 \
  --model <provider/model>
```

For a local Harbor dataset or task directory:

```bash
uv run python runs/run_bench.py harbor \
  --path /path/to/harbor/dataset-or-task \
  --n-tasks 1 \
  --model <provider/model>
```

Useful selectors are forwarded directly to Harbor:

```text
--dataset, --path, --repo, --task, --registry-url, --registry-path,
--include-task-name, --exclude-task-name, --n-tasks
```

SAL-specific runner controls are passed as Harbor agent kwargs:

```text
--agent-flavor bash|bash_task|bash_task_read|bash_skills
--max-turns 150
--provider openai|fake
--api-kind openai-responses|openai-chat|anthropic-messages  # default: openai-responses
--agent-kwarg install_timeout_sec=3000
--setup-proxy-from-env
```

`--setup-proxy-from-env` copies host proxy variables
(`HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`, and lowercase variants) into private
`SAL_HARBOR_SETUP_*` agent environment entries. The installed-agent adapter maps
those private entries back to normal proxy variables only while installing SAL
inside the task container, then filters them before the SAL runner starts. It
does not pass proxy variables to the task verifier.

## Outputs

The host wrapper defaults Harbor jobs to:

```text
evals/out/harbor/jobs/
```

Within each Harbor trial's agent logs, SAL writes:

```text
sal-instruction.txt
sal-summary.json
sal-trajectory.jsonl
simple-agent-lab.txt
```

Harbor's `result.json` remains the score source of truth. The SAL summary and
trajectory are debugging artifacts for understanding the agent loop.

## Terminal-Bench 2.1 Batch Runs

The checked-in Terminal-Bench 2.1 launchers use the model configured in the
root `.env`, force `OPENAI_REASONING_EFFORT=xhigh`, run 10 trials concurrently,
set the general and agent-setup timeout multipliers to 3, and give each SAL
agent 150 turns. They also forward `SAL_BASH_MAX_TIMEOUT_SECONDS=300`, matching
the Bash tool's 300-second default maximum command timeout:

```bash
bash runs/harbor/run_terminal_bench_2_1_bash.sh
bash runs/harbor/run_terminal_bench_2_1_bash_task.sh
```

Use the sequential launcher when the `bash_task` experiment must not start
until the `bash` Harbor process has exited successfully:

```bash
bash runs/harbor/run_terminal_bench_2_1_sequential.sh
```

Set `HARBOR_DRY_RUN=1` to inspect both generated Harbor commands without
starting task containers or model calls.
