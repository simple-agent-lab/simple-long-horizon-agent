# DGM self-evolving recipe

A faithful [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) reproduction
on SWE-bench. A model-driven meta-agent rewrites the whole agent program under
`agent/`; the evolution kernel runs a **parallel open-ended archive-admission
loop** (branches per round, best-valid promotion, archive parent selection); and
the seed and best-on-train agents are scored on a held-out test split so the run
reports a before/after delta. Every DGM knob is exposed.

This recipe is the **faithfulness showcase** — the counterpart to the
[`simple`](../simple/README.md) recipe's minimalism.

## Layout

- `evolve.py` — the recipe (proposal, archive admission, train rollout, criterion,
  promotion, held-out scoring).
- `../../configs/dgm_swebench.yaml` — the default editable run config.
- `config.py` — recipe-local YAML schema and CLI-override resolver.
- `algorithm/archive.py` / `algorithm/open_ended.py` — DGM-local archive
  reconstruction, parent selection, and parallel branch admission.
- `algorithm/repo_edits.py` — optional DGM helper for turning full repository
  tree changes into version proposals.
- `ops/baseline.py` — measure the seed agent per-instance, then build a balanced
  "headroom" train/test split (pure selection/splitting logic is unit-tested).
- `ops/report.py` — summarize a run. The final held-out scoring step writes one
  `generation_metrics.jsonl` row (aggregated from
  `official/final/eval_results.jsonl`) capturing the best version's official
  resolved rate, selector, and held-out test score; `report.py` headlines that
  row alongside the `decisions.jsonl` monitor (accepted/rejected decisions,
  current version, selector distribution, test-leakage monitor).

## Prerequisites

See [`../README.md`](../README.md): a `.env` with provider keys, a reachable
Docker daemon, a wheelhouse, and train/test SWE-bench JSONL splits.

## 1. Build a headroom split (optional)

The default verified splits can be too easy/small to show signal. Build a
repo-balanced candidate pool, roll the seed scaffold once over that pool, and
carve a balanced headroom split:

```bash
uv run --extra swebench python recipes/dgm/ops/baseline.py \
  --run-id baseline-demo-160 \
  --pool-size 160 \
  --pool-out evals/out/dgm_swebench/splits/demo-pool-160.jsonl \
  --baseline-out evals/out/dgm_swebench/splits/demo-baseline-160.jsonl \
  --train-out evals/out/dgm_swebench/splits/demo-train-60.jsonl \
  --test-out evals/out/dgm_swebench/splits/demo-test-60.jsonl \
  --train-size 60 \
  --test-size 60 \
  --parallel 3
```

This writes a selected pool, per-instance seed resolve records, and disjoint
train/test JSONL files under `evals/out/dgm_swebench/splits/`. To skip the
Docker baseline pass after a previous measurement, pass `--reuse-baseline` with
the saved baseline JSONL.

## 2. Run the evolution

The default config is [`../../configs/dgm_swebench.yaml`](../../configs/dgm_swebench.yaml).
It points at the generated demo train/test split under `configs/swebench/`.
You can run it directly, or copy it when you want to change the train/test
paths, rounds, branch count, parallel worker cap, model, or wheelhouse settings:

```bash
cp configs/dgm_swebench.yaml configs/my_dgm_swebench.yaml
```

Dry plan first:

```bash
bash runs/run_dgm_swebench.sh \
  --config configs/my_dgm_swebench.yaml \
  --run-id dgm-smoke \
  --rounds 3
```

Then the real run (the wrapper ensures Docker/Linux `uv`; the recipe refreshes
the configured wheelhouse, then runs `python -m recipes.dgm.evolve --execute`):

```bash
bash runs/run_dgm_swebench.sh --run-id dgm-real --execute

bash runs/run_dgm_swebench.sh \
  --config configs/my_dgm_swebench.yaml \
  --run-id dgm-real-custom \
  --rounds 5 --branches 3 --parent-selection score_child_prop \
  --execute
```

You can also call the recipe directly with
`uv run --extra swebench python -m recipes.dgm.evolve ...` if you have already
prepared the wheelhouse. Use module mode (`-m`) rather than
`python recipes/dgm/evolve.py`; script-path execution can put `recipes/dgm/` on
`sys.path` and shadow the installed `swebench` package with
`recipes/dgm/swebench.py`.

## 3. Monitor / report

```bash
# during or after a run
uv run python recipes/dgm/ops/report.py evals/out/dgm_swebench/<run-id> \
  --test-dataset evals/out/dgm_swebench/splits/headroom-test-20.jsonl

# or via the recipe's built-in monitor flag
bash runs/run_dgm_swebench.sh --run-id dgm-real \
  --config configs/my_dgm_swebench.yaml --monitor
```

## DGM Config

Most run shape lives in YAML now:

| YAML field | Default | Meaning |
| --- | --- | --- |
| `dataset.train_path` / `dataset.test_path` | `configs/swebench/demo-*.jsonl` | Train and held-out before/final scoring JSONL slices. |
| `dgm.rounds` | `4` | Sequential evolution rounds. Total candidates = `rounds × branches`. |
| `dgm.branches` | `3` | Candidate branches evaluated concurrently per round. |
| `dgm.meta_concurrency` | `0` (= branches) | Concurrent meta-agent LLM calls per round. |
| `dgm.parent_selection` | `score_child_prop` | `latest` \| `best` \| `score_prop` \| `score_child_prop`. |
| `execution.parallel` | `3` | Global Docker worker cap. Must be at least `dgm.branches`. |
| `execution.max_turns` | `75` | Per-instance agent turn budget. |
| `execution.wheelhouse` | `evals/out/swebench/wheelhouse/cp311-manylinux` | Container wheelhouse. |
| `model.default_model` | `dgm-swebench` | Provider model written into `provider.json` when `OPENAI_MODEL` is unset. |
| `model.api_kind` | `openai-chat` | `openai-chat` \| `openai-responses`. |
| `dataset.name` | `princeton-nlp/SWE-bench_Verified` | Source dataset for official scoring. |

CLI flags are overrides for quick experiments. The common ones are `--config`,
`--run-id`, `--execute`, `--reset`, and `--monitor`; lower-level flags such as
`--rounds`, `--branches`, and `--parallel` remain available but should usually be
edited in the YAML for reproducibility.

## Output

Run artifacts land under `evals/out/dgm_swebench/<run-id>/` (gitignored). See
[`evals/out/dgm_swebench/README.md`](../../evals/out/dgm_swebench/README.md) for
the directory layout. Official performance claims should use
`official/baseline/eval_results.jsonl`, `official/final/eval_results.jsonl`, and
`test_summary.json`, not only the recipe-level `generation_metrics.jsonl`.
