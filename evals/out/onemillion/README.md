# OneMillion-Bench Outputs

Per-run artifacts for the `onemillion` suite. Runs are named
`onemillion-<timestamp>` by default. Everything here except this README is
git-ignored.

## Layout

```text
evals/out/onemillion/
├── README.md                       ← this file
└── <run-id>/
    └── <instance-id>/              (e.g. case_2860)
        ├── input/
        │   ├── instance.json       sanitized case fed to the agent (no rubrics)
        │   └── eval.json           staged rubrics for the judge (gold)
        └── out/
            ├── trajectory.jsonl    generation trace (schema v3)
            └── result.json         model_response + rubric verdict (score, ...)
```

## Generating a Run

```bash
# One case
uv run python runs/run_bench.py onemillion case_2860 \
  --dataset datasets/OneMillion-Bench/healthcare_and_medicine

# A whole domain / the full dataset
uv run python runs/run_bench.py onemillion --all \
  --dataset datasets/OneMillion-Bench --concurrency 8
```

See `evals/onemillion/README.md` for setup and the environment contract.
