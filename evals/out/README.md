# Eval Output

Local artifacts produced by eval runs. Everything here except this README is
gitignored — nothing in this tree is source of truth.

## Directory Layout

Every suite writes the same per-instance shape under its own root:

```text
evals/out/<suite>/
├── instance_<id>.jsonl              staged dataset row (SWE-bench variants)
├── wheelhouse/cp311-manylinux/*.whl offline wheels for the container venv
└── <run-id>/<instance-id>/
    ├── input/
    │   ├── instance.json            sanitized case fed to the agent
    │   └── eval.json                staged gold/rubrics, when a suite scores in place
    └── out/
        ├── trajectory.jsonl         three-layer trace (schema v5)
        ├── result.json              suite-specific run product
        └── prediction.jsonl         model_patch prediction (SWE-bench variants)
```

The suite roots are `swebench/`, `swebench_multilingual/`, `swebench_pro/`,
`programbench/`, `onemillion/`, and `harbor/`. SWE-bench Verified, Multilingual,
and Pro share the host adapter under `evals/swebench/` but keep separate output
roots. Harbor is the exception: it delegates to Harbor's own job layout under
`harbor/jobs/`.

Run ids default to `<suite>-<timestamp>`. A scoring pass writes alongside the
run it scored, as `<run-id>_eval/`.

## Generating and Scoring Runs

The commands live with each suite, not here:

- [`evals/README.md`](../README.md) — the eval framework and shared entry points.
- [`evals/swebench/README.md`](../swebench/README.md),
  [`evals/programbench/README.md`](../programbench/README.md),
  [`evals/onemillion/README.md`](../onemillion/README.md),
  [`evals/harbor/README.md`](../harbor/README.md) — per-suite setup, run, and
  scoring commands.
- `uv run python -m runs.run_bench list` — every registered bench entry point.

## Adding a New Benchmark

1. Create `evals/<suite>/` with adapter code and a README.
2. Add `runs/_benches/<suite>.py` and register it in `runs/run_bench.py`.
3. Add a setup shell only when external images or datasets require one.
4. Note the suite root above if its output layout differs from the shape here.
