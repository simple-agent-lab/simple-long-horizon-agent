# SWE-bench Pro Outputs

Each run is named `pro-<timestamp>` or a custom label.

## Layout

```text
evals/out/swebench_pro/
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
bash runs/swebench/run_swebench.sh --variant pro
bash runs/swebench/run_swebench.sh --variant pro --all --parallel 4
```
