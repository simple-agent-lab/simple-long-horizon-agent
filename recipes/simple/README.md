# Simple self-evolving recipe

The simple recipe is the config-backed entry point for the generic v1
self-evolving runner. It runs the benchmark-agnostic `simple` algorithm from a
YAML file, registers the SWE-bench factories, and delegates the actual CLI to
`simple_agent_lab.evolution.run`.

This recipe is intentionally smaller than the DGM recipe. The generic v1 builder
supports only `evolution.algorithm: simple`; DGM/open-ended scheduling remains
recipe-local under [`../dgm/`](../dgm/).

## Default config

The default config is [`../../configs/simple_swebench.yaml`](../../configs/simple_swebench.yaml).
It uses registry names for the SWE-bench suite and source-tree surface:

- suite: `swebench`
- surface: `source_tree`
- backend: `local_docker`
- store: `local_dir`
- strategy: `source_tree_agent`
- algorithm: `simple`

That means the meta-agent edits the real framework source under
`src/simple_agent_lab/**/*.py`, not a lightweight wrapper package. Each
candidate version is staged under `input/source_tree/src/simple_agent_lab/`, and
SWE-bench imports that candidate source before the installed package.

The default train and heldout slices point at the generated SWE-bench demo
split:

- [`../../configs/swebench/demo-train-60.jsonl`](../../configs/swebench/demo-train-60.jsonl)
- [`../../configs/swebench/demo-test-60.jsonl`](../../configs/swebench/demo-test-60.jsonl)

For a real run, copy the config or pass your own with `--config`, then edit the
`instances.train.path`, `instances.heldout.path`, model settings, output root,
`strategy.args.repo_root`, and execution settings when you want a different
split, source checkout, or runtime shape.

## Quick start

From the repo root:

```bash
uv sync --group dev --extra swebench
```

Put provider settings in `.env` or export them in the shell. The checked-in
simple config reads:

```bash
OPENAI_AUTH_TOKEN=...
OPENAI_MODEL=...
OPENAI_BASE_URL=...   # optional for compatible providers
```

Docker must be reachable before `--execute`. Docker Desktop works as-is; for
Colima, start it first. If Python cannot discover the daemon even though the
Docker CLI works, export the Colima socket:

```bash
colima start
export DOCKER_HOST=unix://$HOME/.colima/default/docker.sock
```

The run wrapper prepares the Linux `uv` helper and SWE-bench wheelhouse on the
first real run.

Dry-run the default config without Docker or credentials:

```bash
bash runs/run_self_evolving_simple.sh --run-id simple-smoke
```

Run the configured model + Docker loop with the default config:

```bash
bash runs/run_self_evolving_simple.sh --run-id simple-real --execute
```

For a smaller real smoke, copy the config and point train at a tiny JSONL split,
then disable heldout if you only want to test the training loop:

```bash
cp configs/simple_swebench.yaml configs/my_simple_swebench.yaml
```

Edit `configs/my_simple_swebench.yaml`:

```yaml
instances:
  train:
    path: configs/swebench/my-train-3.jsonl
  heldout: null
execution:
  parallel: 3
  max_turns: 75
evolution:
  rounds: 3
evaluation:
  baseline_heldout: false
  final_heldout: false
```

Then run it:

```bash
nohup bash runs/run_self_evolving_simple.sh \
  --config configs/my_simple_swebench.yaml \
  --run-id simple-real-custom \
  --execute \
  > simple-real.log 2>&1 &
```

For an interactive foreground run, use the same command without `nohup` and
redirection:

```bash
bash runs/run_self_evolving_simple.sh \
  --config configs/my_simple_swebench.yaml \
  --run-id simple-real-custom \
  --execute
```

The wrapper keeps the `uv run --extra swebench python` path when `uv` is
available, then calls `recipes/simple/evolve.py "$@"`. You can call the recipe
directly with the same flags.

## Flags

| Flag | Meaning |
| --- | --- |
| `--config PATH` | YAML run config. Defaults to `configs/simple_swebench.yaml` when omitted. |
| `--run-id ID` | Override `run.id` from the config; names one output child directory. |
| `--execute` | Run the configured evolution loop instead of printing the dry-run plan. |
| `--reset` | Clear stale state for this run before building it. |
| `--monitor` | Print the run root external monitors should watch. |

Other run shape settings live in the YAML. In particular, `rounds`,
`instances.train.path`, `instances.heldout.path`, `evaluation.*`,
`execution.parallel`, `execution.max_turns`, and backend or store options should
be edited in the config file.

## Outputs

Dry-runs print the resolved plan: run id, run root, suite, surface, editable
components, train slice, heldout slice when configured, counts, and rounds.
Executed runs write the generic evolution workspace under
`<output_root>/<run-id>/evolution` plus suite run artifacts under
`<output_root>/<run-id>/runs`.

For the default config, the most useful paths are:

- `evals/out/self_evolving/simple/<run-id>/evolution/versions/`: immutable
  candidate source-tree versions.
- `evals/out/self_evolving/simple/<run-id>/evolution/decisions.jsonl`: one
  baseline-vs-candidate record per proposed generation.
- `evals/out/self_evolving/simple/<run-id>/runs/`: per-version, per-instance
  SWE-bench artifacts.
- `evals/out/self_evolving/simple/<run-id>/evaluation/summary.json`: optional
  heldout before/final summary.

During a background run, monitor:

```bash
tail -f simple-real.log
find evals/out/self_evolving/simple/simple-real-custom -name result.json | wc -l
tail -n 20 evals/out/self_evolving/simple/simple-real-custom/evolution/decisions.jsonl
```

When `evaluation.baseline_heldout` or `evaluation.final_heldout` is enabled,
the simple runner evaluates the current agent on `instances.heldout` before
and/or after evolution and writes a generic performance claim to
`<output_root>/<run-id>/evaluation/summary.json`. The report is benchmark-agnostic:
it uses the configured suite rollout and the configured reward function, then
records reward means, resolved counts when the suite provides `resolved`, and
the before/final delta.

For SWE-bench, the default config sets `suite.args.in_env_scoring: true` so the
suite stages its in-environment scoring inputs and `result.json` contains a
gradable reward. The DGM recipe still owns its archive-specific official
baseline/final workflow and extra reporting artifacts.
