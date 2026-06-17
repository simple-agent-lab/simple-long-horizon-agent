# Simple self-evolving recipe

The framework is good enough that one short script starts a real self-evolving
run. A model rewrites the whole agent program under `agent/`, the evolution
kernel compares each candidate on a train slice in a SWE-bench Docker sandbox,
and the best valid agent is scored on a held-out test slice.

This recipe is the **ergonomics showcase**: it runs the evolution kernel
sequentially via `Experiment.run`, with `best` parent selection and a single
container at a time. Read `evolve.py` top-to-bottom — it is the whole story.

## Prerequisites

See [`../README.md`](../README.md): a `.env` with provider keys, a reachable
Docker daemon, a wheelhouse, and train/test SWE-bench JSONL splits.

## Run it

Dry plan (no model, no Docker — validates inputs and prints the run layout):

```bash
uv run --extra swebench python recipes/simple/evolve.py \
  --run-id simple-smoke \
  --train-dataset evals/out/dgm_swebench/splits/headroom-train-20.jsonl \
  --test-dataset evals/out/dgm_swebench/splits/headroom-test-20.jsonl
```

Real run (model + Docker) — the `runs/` wrapper prepares Docker and the
wheelhouse for you, then calls the recipe with `--execute`:

```bash
bash runs/run_self_evolving_simple.sh \
  --run-id simple-real \
  --train-dataset evals/out/dgm_swebench/splits/headroom-train-20.jsonl \
  --test-dataset evals/out/dgm_swebench/splits/headroom-test-20.jsonl \
  --rounds 4 \
  --execute
```

Or call the recipe directly with
`uv run --extra swebench python recipes/simple/evolve.py ...` once the wheelhouse
exists.

## Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--run-id` | required | Reproducible run id (names the output subdirectory). |
| `--train-dataset` | required | SWE-bench JSONL slice the candidates evolve against. |
| `--test-dataset` | required | Held-out JSONL slice the best agent is scored on. |
| `--rounds` | `4` | Number of sequential evolution generations. |
| `--max-turns` | `75` | Per-instance agent turn budget. |
| `--output-root` | `evals/out/self_evolving/simple` | Where run artifacts land. |
| `--wheelhouse` | `evals/out/swebench/wheelhouse/cp311-manylinux` | Container wheelhouse. |
| `--dotenv` | `.env` | Provider env file. |
| `--execute` | off | Run the real model + Docker (otherwise dry plan). |

The model name comes from `OPENAI_MODEL` in your environment (default
`evolving-swebench`).

## What you get

Under `evals/out/self_evolving/simple/<run-id>/` you'll find the evolution
workspace (version store, pointers, `decisions.jsonl`) and the SWE-bench run
artifacts. The script prints the final generation count and the current agent
hash; for a richer summary, point the DGM recipe's
[`report.py`](../dgm/report.py) at the run root.
