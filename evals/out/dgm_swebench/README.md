# DGM SWE-bench Output

This directory holds local artifacts for DGM-style performance runs over
SWE-bench. Contents are gitignored except for this README.

## Layout

```text
evals/out/dgm_swebench/
├── splits/
│   ├── demo-train-60.jsonl          # Optional local/custom evolution dataset
│   └── demo-test-60.jsonl           # Optional local/custom held-out dataset
└── <run-id>/
    ├── evolution/                 # Version store, pointers, runs, decisions.jsonl
    ├── swebench_runs/             # Generic SWE-bench run_dataset artifacts
    │   └── <run-id>/<instance-id>/
    │       ├── input/instance.json
    │       └── out/
    │           ├── result.json
    │           └── trajectory.jsonl
    ├── official/
    │   ├── baseline/
    │   │   ├── baseline_predictions.jsonl
    │   │   ├── eval_results.jsonl
    │   │   └── harness/           # Official SWE-bench reports
    │   └── final/
    │       ├── final_predictions.jsonl
    │       ├── eval_results.jsonl
    │       └── harness/
    ├── test_summary.json          # Baseline/final held-out delta
    └── generation_metrics.jsonl    # Held-out summary row written after official
                                    # scoring (one row for the best held-out eval)
```

`generation_metrics.jsonl` is the recipe-level summary. Official performance
claims should use `official/baseline/eval_results.jsonl`,
`official/final/eval_results.jsonl`, `test_summary.json`, or the corresponding
official SWE-bench reports.

## Smoke Command

```bash
bash runs/run_dgm_swebench.sh --run-id dgm-swebench-smoke
```

The command above is a dry plan by default and uses the tracked
`configs/swebench/demo-*.jsonl` split from `configs/dgm_swebench.yaml`. Add
`--execute` only after Docker/provider prerequisites are ready. To build a
balanced custom train/test split under `splits/`, see
`recipes/dgm/ops/baseline.py` (`recipes/dgm/README.md`).
