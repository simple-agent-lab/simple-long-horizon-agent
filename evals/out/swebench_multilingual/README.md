# SWE-bench Multilingual Outputs

Each run is named `multilingual-<timestamp>` or a custom label.

## Layout

```text
evals/out/swebench_multilingual/
├── README.md                       <- this file
├── instance_<id>.jsonl             <- fetched instance records
├── wheelhouse/                     <- pre-built wheels for container installs
│   └── cp311-manylinux/*.whl
└── <run-id>/
    └── <instance-id>/
        ├── input/
        │   └── instance.json       sanitized instance fed to the agent
        └── out/
            ├── trajectory.jsonl    three-layer trace (schema v3)
            └── prediction.jsonl    {instance_id, model_name_or_path, model_patch}
```

## Generating a Run

```bash
bash runs/swebench/run_swebench_multilingual.sh
bash runs/swebench/run_swebench_multilingual.sh --all --parallel 4
```

## File Sizes

- `trajectory.jsonl`: 50 KB - 5 MB per instance (contains raw LLM I/O)
- `prediction.jsonl`: < 10 KB per instance
- `instance.json`: < 50 KB per instance
