# OneMillion-Bench Outputs

Per-run artifacts for the `onemillion` suite. Runs are named
`onemillion-<timestamp>` by default. Everything here except this README is
gitignored.

## Layout

```text
evals/out/onemillion/
├── README.md
└── <run-id>/
    └── <instance-id>/
        ├── input/
        │   ├── instance.json       sanitized case fed to the agent
        │   └── eval.json           staged rubrics for the judge
        └── out/
            ├── trajectory.jsonl    generation trace (schema v5)
            └── result.json         model response and rubric verdict
```

## Generating a Run

```bash
# One case
uv run python -m runs.run_bench onemillion case_2860 \
  --dataset datasets/OneMillion-Bench/healthcare_and_medicine

# A whole domain or dataset
uv run python -m runs.run_bench onemillion --all \
  --dataset datasets/OneMillion-Bench --concurrency 8
```

See `evals/onemillion/README.md` for setup and the environment contract.
