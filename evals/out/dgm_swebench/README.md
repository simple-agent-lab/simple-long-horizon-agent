# DGM SWE-bench Output

This directory holds local artifacts for DGM-style performance runs over
SWE-bench. Contents are gitignored except for this README.

## Layout

```text
evals/out/dgm_swebench/
├── splits/
│   ├── macbook-train.jsonl          # Optional tiny local evolution dataset
│   └── macbook-test.jsonl           # Optional tiny held-out scoring dataset
└── <run-id>/
    ├── evolution/                 # Version store, pointers, runs, decisions.jsonl
    ├── swebench_runs/             # Generic SWE-bench run_dataset artifacts
    │   └── <run-id>/<instance-id>/
    │       ├── input/instance.json
    │       └── out/
    │           ├── result.json
    │           └── trajectory.jsonl
    ├── official/
    │   ├── <run-id>_predictions.jsonl
    │   ├── eval_results.jsonl
    │   └── harness/               # Official SWE-bench reports
    └── generation_metrics.jsonl    # One summary record per generation
```

`generation_metrics.jsonl` is the recipe-level summary. Official performance
claims should use `official/eval_results.jsonl` or the corresponding official
SWE-bench reports.

## Smoke Command

```bash
bash runs/run_dgm_swebench.sh \
  --run-id dgm-swebench-smoke \
  --train-dataset evals/out/dgm_swebench/splits/headroom-train-20.jsonl \
  --test-dataset evals/out/dgm_swebench/splits/headroom-test-20.jsonl \
  --rounds 3 \
  --parent-selection score_child_prop
```

The command above is a dry plan by default. Add `--execute` only after the
SWE-bench run artifacts exist and Docker/provider prerequisites are ready. To
build a balanced train/test split under `splits/`, see `recipes/dgm/baseline.py`
(`recipes/dgm/README.md`).
