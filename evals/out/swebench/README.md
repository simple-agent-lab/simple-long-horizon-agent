# SWE-bench Verified Outputs

Single-instance smoke runs are named `swebench-<timestamp>` by default.
Verified batch runs are named `verified-<timestamp>`.

## Layout

```text
evals/out/swebench/
├── README.md
├── instance_<id>.jsonl
├── wheelhouse/
│   └── cp311-manylinux/*.whl
└── <run-id>/
    └── <instance-id>/
        ├── input/
        │   └── instance.json
        └── out/
            ├── trajectory.jsonl    three-layer trace (schema v5)
            └── prediction.jsonl    model_patch prediction
```

## Generating a Run

```bash
bash runs/swebench/run_swebench.sh sympy__sympy-23824
bash runs/swebench/run_swebench.sh --all --parallel 4
```

## Evaluating Predictions

```bash
uv run python -m evals.swebench.evaluate_predictions \
  --predictions evals/out/swebench/<run-id>/<id>/out/prediction.jsonl \
  --instance evals/out/swebench/<run-id>/<id>/input/instance.json
```
