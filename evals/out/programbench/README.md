# ProgramBench Outputs

Local artifacts from ProgramBench runs. Run ids default to
`programbench-<timestamp>`.

## Layout

```text
evals/out/programbench/
├── README.md
├── wheelhouse/
│   └── cp311-manylinux/*.whl
├── <run-id>/
│   └── <instance-id>/
│       ├── input/
│       │   └── instance.json
│       └── out/
│           ├── trajectory.jsonl    three-layer trace (schema v5)
│           └── result.json         submission_tar_b64 and run metadata
└── <run-id>_eval/
    ├── <instance-id>/
    │   ├── submission.tar.gz
    │   └── <instance-id>.eval.json
    └── scores.json
```

The ProgramBench product is the whole workspace. The container half stores it
as a base64-encoded tarball in `result.json`; the official scorer reconstructs
the submission archive.

## Generating a Run

```bash
bash runs/programbench/run_programbench.sh abishekvashok__cmatrix.5c082c6
bash runs/programbench/run_programbench.sh --all --parallel 4
```

## Scoring

```bash
uv run --extra programbench python -m evals.programbench.evaluate_submissions \
  --run-id <run-id> --workers 4
```

The official evaluator requires Docker, the `programbench` package, and access
to its test data.
