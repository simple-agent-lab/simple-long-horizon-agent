# SWE-bench Outputs

Each run is named `swebench-<timestamp>` or a custom label.

## Layout

```text
evals/out/swebench/
├── README.md                       ← this file
├── verified/
│   ├── instances/                  ← cached Verified dataset rows
│   ├── container_runs/             ← per-instance agent run dirs
│   ├── predictions/                ← collected prediction JSONL files
│   ├── eval_results/               ← normalized eval-result outputs
│   └── official/                   ← official SWE-bench harness outputs
├── pro/
│   ├── instances/                  ← cached Pro dataset rows
│   ├── container_runs/             ← per-instance agent run dirs
│   ├── predictions/                ← collected prediction JSONL files
│   ├── eval_results/               ← normalized eval-result outputs
│   ├── official/                   ← official Pro harness outputs
│   └── SWE-bench_Pro-os/           ← optional local Pro evaluator checkout
└── shared/
    └── wheelhouse/                 ← pre-built wheels for container installs
        └── cp311-manylinux/*.whl
```

## Generating a Run

```bash
# Quick smoke run on a single instance
bash runs/run_swebench_container.sh sympy__sympy-23824

# Batch runs
bash runs/run_swebench_verified.sh --all --parallel 4
bash runs/run_swebench_pro.sh --all --parallel 4
```

## Evaluating Predictions

```bash
python evals/swebench/evaluate_predictions.py \
  --predictions evals/out/swebench/verified/predictions/<run-id>_predictions.jsonl \
  --jsonl evals/out/swebench/verified/eval_results/<run-id>_eval_results.jsonl
```

## File Sizes

- `trajectory.jsonl`: 50 KB – 5 MB per instance (contains raw LLM I/O)
- `prediction.jsonl`: < 10 KB per instance
- `instance.json`: < 50 KB per instance
- `pro/SWE-bench_Pro-os/`: external checkout; intentionally empty in git until
  you clone `scaleapi/SWE-bench_Pro-os` there.
