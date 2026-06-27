# SWE-bench Verified Outputs

Single-instance smoke runs are named `swebench-<timestamp>` by default.
Verified batch runs are named `verified-<timestamp>`.

## Layout

```text
evals/out/swebench/
├── README.md                       ← this file
├── instance_<id>.jsonl             ← fetched instance records
├── wheelhouse/                     ← pre-built wheels for container installs
│   └── cp311-manylinux/*.whl
└── <run-id>/
    └── <instance-id>/              (e.g. django__django-12113)
        ├── input/
        │   └── instance.json       sanitized instance fed to the agent
        └── out/
            ├── trajectory.jsonl    three-layer trace (schema v3)
            └── prediction.jsonl    {instance_id, model_name_or_path, model_patch}
```

## Generating a Run

```bash
# Quick smoke run on a single instance
bash runs/swebench/run_swebench_verified.sh sympy__sympy-23824

# Full SWE-bench Verified split
bash runs/swebench/run_swebench_verified.sh --all --parallel 4
```

## Evaluating Predictions

```bash
python evals/swebench/evaluate_predictions.py \
  --predictions evals/out/swebench/<run-id>/<id>/out/prediction.jsonl \
  --instance    evals/out/swebench/<run-id>/<id>/input/instance.json
```

## File Sizes

- `trajectory.jsonl`: 50 KB – 5 MB per instance (contains raw LLM I/O)
- `prediction.jsonl`: < 10 KB per instance
- `instance.json`: < 50 KB per instance
