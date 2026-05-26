# Eval Output

This directory holds local artifacts produced by eval runs. Contents are
gitignored except for README files that document the expected structure.

## Directory Layout

```text
evals/out/
├── README.md                          ← this file
└── swebench/                          ← SWE-bench suite outputs
    ├── README.md                      ← per-suite structure docs
    ├── wheelhouse/                    ← pre-built wheels for container installs
    │   └── cp311-manylinux/
    │       └── *.whl
    ├── instance_<id>.jsonl            ← fetched instance records
    └── <run-id>/
        └── <instance-id>/
            ├── input/
            │   └── instance.json      ← sanitized instance (no gold patch)
            └── out/
                ├── trajectory.jsonl   ← three-layer trace (events, spans, model_turns)
                └── prediction.jsonl   ← SWE-bench prediction with model_patch
```

Each benchmark suite gets its own subdirectory under `evals/out/`, matching
the adapter directory name under `evals/`.

## Reproducing the Structure

```bash
# 1. Fetch an instance
bash runs/setup_swebench_docker.sh django__django-12113

# 2. Run the agent
bash runs/run_swebench_container.sh django__django-12113
```

Outputs land under `swebench/<run-id>/<instance-id>/out/`.

## Adding a New Benchmark

When adding a new benchmark suite (e.g. `aider_bench`, `humanevalfix`):

1. Create `evals/<suite>/` with adapter code and a README.
2. Create `evals/out/<suite>_runs/README.md` documenting the output layout.
3. Add a gitignore exception in `.gitignore` so the README survives.
4. Add a `runs/run_<suite>.sh` convenience script.
5. If setup requires external resources (git clones, docker images, datasets),
   put that in `runs/setup_<suite>.sh` and document prerequisites in
   `evals/<suite>/README.md`.
