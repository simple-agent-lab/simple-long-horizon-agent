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
It uses registry names for the SWE-bench suite and Python-agent surface:

- suite: `swebench`
- surface: `python_agent_package`
- backend: `local_docker`
- store: `local_dir`
- strategy: `model_program`
- algorithm: `simple`

The default train slice points at the checked-in tiny JSONL example
[`../../configs/examples/swebench_train_tiny.jsonl`](../../configs/examples/swebench_train_tiny.jsonl)
so a dry-run works in a clean checkout. For a real run, copy the config or pass
your own with `--config`, then edit the `instances.train.path`, model settings,
output root, and execution settings for your environment.

## Run it

Dry-run the default config without Docker or credentials:

```bash
bash runs/run_self_evolving_simple.sh --run-id simple-smoke
```

Use an edited config:

```bash
bash runs/run_self_evolving_simple.sh \
  --config configs/my_simple_swebench.yaml \
  --run-id simple-real
```

Run the configured model + Docker loop:

```bash
bash runs/run_self_evolving_simple.sh \
  --config configs/my_simple_swebench.yaml \
  --run-id simple-real \
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
`instances.train.path`, `execution.parallel`, `execution.max_turns`, and backend
or store options should be edited in the config file.

## Outputs

Dry-runs print the resolved plan: run id, run root, suite, surface, editable
components, train slice, train count, and rounds. Executed runs write the
generic evolution workspace under `<output_root>/<run-id>/evolution` plus suite
run artifacts under `<output_root>/<run-id>/runs`.

The generic simple runner does not yet implement held-out before/after official
scoring. Keep held-out evaluation in your own follow-up command or use the DGM
recipe when you need its recipe-local scoring workflow.
