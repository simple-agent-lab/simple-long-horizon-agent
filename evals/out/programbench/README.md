# ProgramBench Outputs

Local artifacts from ProgramBench runs. Run ids default to
`programbench-<timestamp>`.

## Layout

```text
evals/out/programbench/
├── README.md                       ← this file
├── wheelhouse/                     ← pre-built wheels for container installs
│   └── cp311-manylinux/*.whl
├── <run-id>/
│   └── <instance-id>/              (e.g. abishekvashok__cmatrix.5c082c6)
│       ├── input/
│       │   └── instance.json       sanitized instance (no repo/commit/gold)
│       └── out/
│           ├── trajectory.jsonl    three-layer trace (schema v3)
│           └── result.json         {instance_id, submission_tar_b64, network_isolated, ...}
└── <run-id>_eval/                  ← built by evaluate_submissions.py
    ├── <instance-id>/
    │   ├── submission.tar.gz       decoded from result.json's submission_tar_b64
    │   └── <instance-id>.eval.json official `programbench eval` result
    └── scores.json                 machine-readable manifest
```

Unlike SWE-bench (whose product is a `model_patch` diff), the ProgramBench
product is the **whole workspace**: the container half tars + gzips it and
returns it base64-encoded as `submission_tar_b64` in `result.json` (a container
half can only return bytes through that file). `evaluate_submissions.py` decodes
it back into the `<id>/submission.tar.gz` layout the official scorer expects.

## Generating a Run

```bash
# One instance
bash runs/run_programbench_suite.sh abishekvashok__cmatrix.5c082c6

# Whole task set, 4 at a time
bash runs/run_programbench.sh --all --parallel 4
```

## Scoring

```bash
uv run python evals/programbench/evaluate_submissions.py \
  --run-id <run-id> --workers 4
```

This rebuilds each `submission.tar.gz` under `<run-id>_eval/` and runs the
official `programbench eval` (needs Docker + the `programbench` package + access
to the HF test blobs).

## File Sizes

- `trajectory.jsonl`: 50 KB – 5 MB per instance (raw LLM I/O).
- `result.json`: a few KB to a few MB (holds the gzipped workspace, base64).
- `instance.json`: < 5 KB per instance.
