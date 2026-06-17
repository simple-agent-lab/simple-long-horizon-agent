# DGM self-evolving recipe

A faithful [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) reproduction
on SWE-bench. A model-driven meta-agent rewrites the whole agent program under
`agent/`; the evolution kernel runs a **parallel open-ended archive-admission
loop** (branches per round, best-valid promotion, archive parent selection); and
the best-on-train agent is scored on a held-out test split. Every DGM knob is
exposed.

This recipe is the **faithfulness showcase** — the counterpart to the
[`simple`](../simple/README.md) recipe's minimalism.

## Layout

- `evolve.py` — the recipe (proposal, archive admission, train rollout, criterion,
  promotion, held-out scoring).
- `archive.py` / `open_ended.py` — DGM-local archive reconstruction, parent
  selection, and parallel branch admission.
- `repo_edits.py` — optional DGM helper for turning full repository tree changes
  into version proposals.
- `baseline.py` — measure the seed agent per-instance, then build a balanced
  "headroom" train/test split (pure selection/splitting logic is unit-tested).
- `report.py` — summarize a run. The held-out scoring step writes one
  `generation_metrics.jsonl` row (aggregated from `official/eval_results.jsonl`)
  capturing the best version's official resolved rate, selector, and held-out
  test score; `report.py` headlines that row alongside the `decisions.jsonl`
  monitor (accepted/rejected decisions, current version, selector distribution,
  test-leakage monitor).

## Prerequisites

See [`../README.md`](../README.md): a `.env` with provider keys, a reachable
Docker daemon, a wheelhouse, and train/test SWE-bench JSONL splits.

## 1. Build a headroom split (optional)

The default verified splits can be too easy/small to show signal. Roll the seed
scaffold once over a candidate pool and carve a balanced split:

```bash
uv run --extra swebench python recipes/dgm/baseline.py \
  --run-id baseline-headroom \
  --train-size 20 --test-size 20 \
  --execute
```

This writes disjoint `*-train-*.jsonl` / `*-test-*.jsonl` files under
`evals/out/dgm_swebench/splits/`.

## 2. Run the evolution

Dry plan first:

```bash
bash runs/run_dgm_swebench.sh \
  --run-id dgm-smoke \
  --train-dataset evals/out/dgm_swebench/splits/headroom-train-20.jsonl \
  --test-dataset evals/out/dgm_swebench/splits/headroom-test-20.jsonl \
  --rounds 3
```

Then the real run (the wrapper prepares Docker + wheelhouse, then calls
`recipes/dgm/evolve.py --execute`):

```bash
bash runs/run_dgm_swebench.sh \
  --run-id dgm-real \
  --train-dataset evals/out/dgm_swebench/splits/headroom-train-20.jsonl \
  --test-dataset evals/out/dgm_swebench/splits/headroom-test-20.jsonl \
  --rounds 5 --branches 3 --parent-selection score_child_prop \
  --execute
```

You can also call the recipe directly with
`uv run --extra swebench python recipes/dgm/evolve.py ...` if you have already
prepared the wheelhouse.

## 3. Monitor / report

```bash
# during or after a run
uv run python recipes/dgm/report.py evals/out/dgm_swebench/<run-id> \
  --test-dataset evals/out/dgm_swebench/splits/headroom-test-20.jsonl

# or via the recipe's built-in monitor flag
bash runs/run_dgm_swebench.sh --run-id dgm-real \
  --train-dataset ... --test-dataset ... --monitor
```

## DGM knobs

| Flag | Default | Meaning |
| --- | --- | --- |
| `--run-id` | required | Reproducible run id. |
| `--train-dataset` / `--test-dataset` | required | Train (evolution) and held-out (scoring) JSONL slices. |
| `--rounds` | `4` | Sequential evolution rounds. Total candidates = `rounds × branches`. |
| `--branches` | `3` | Candidate branches evaluated concurrently per round. |
| `--meta-concurrency` | `0` (= branches) | Concurrent meta-agent LLM calls per round. |
| `--parent-selection` | `score_child_prop` | `latest` \| `best` \| `score_prop` \| `score_child_prop`. |
| `--parallel` | `auto` | Global Docker worker cap, or `auto` to size to the Docker VM. |
| `--model-name` | `OPENAI_MODEL` | Provider model written into `provider.json`. |
| `--api-kind` | `openai-chat` | `openai-chat` \| `openai-responses`. |
| `--max-turns` | `75` | Per-instance agent turn budget. |
| `--dataset-name` | `princeton-nlp/SWE-bench_Verified` | Source dataset for official scoring. |
| `--reset` | off | Remove the run root before starting. |
| `--monitor` | off | Print the report for an existing run and exit. |
| `--execute` | off | Run the real model + Docker (otherwise dry plan). |
| `--generations` | — | Deprecated alias for `--rounds`. |

## Output

Run artifacts land under `evals/out/dgm_swebench/<run-id>/` (gitignored). See
[`evals/out/dgm_swebench/README.md`](../../evals/out/dgm_swebench/README.md) for
the directory layout. Official performance claims should use
`official/eval_results.jsonl`, not the recipe-level `generation_metrics.jsonl`.
