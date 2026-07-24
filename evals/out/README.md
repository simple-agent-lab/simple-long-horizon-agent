# Eval Output

This directory holds local artifacts produced by eval runs. Contents are
gitignored except for the README files that document the expected structure.

## Directory Layout

```text
evals/out/
├── README.md
├── harbor/
│   ├── README.md
│   └── jobs/
├── onemillion/
│   ├── README.md
│   └── <run-id>/<instance-id>/{input,out}/
├── programbench/
│   ├── README.md
│   ├── wheelhouse/
│   └── <run-id>/<instance-id>/{input,out}/
├── swebench/
│   ├── README.md
│   ├── instance_<id>.jsonl
│   ├── wheelhouse/
│   └── <run-id>/<instance-id>/{input,out}/
├── swebench_multilingual/
│   ├── README.md
│   ├── instance_<id>.jsonl
│   ├── wheelhouse/
│   └── <run-id>/<instance-id>/{input,out}/
└── swebench_pro/
    ├── README.md
    ├── instance_<id>.jsonl
    ├── wheelhouse/
    └── <run-id>/<instance-id>/{input,out}/
```

Each benchmark run family gets its own subdirectory. SWE-bench Verified,
Multilingual, and Pro share the host adapter under `evals/swebench/`, but keep
separate output roots.

## Reproducing the Structure

Run commands create artifact directories beneath these documented roots:

```bash
uv run python -m runs.run_bench list
bash runs/swebench/run_swebench.sh sympy__sympy-23824
```

## Adding a New Benchmark

1. Create `evals/<suite>/` with adapter code and a README.
2. Create `evals/out/<suite>/README.md` documenting its output layout.
3. Add a `.gitignore` exception so that README survives.
4. Add `runs/_benches/<suite>.py` and register it in `runs/run_bench.py`.
5. Add a setup shell only when external images or datasets require one.
