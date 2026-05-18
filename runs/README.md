# Runs

This directory contains small shell scripts for reproducible example runs.

The style follows nanochat's `runs/` convention: a script should be readable, copy-pasteable, and useful as the reference way to run an experiment.

## Available Runs

```bash
bash runs/run_ci.sh
bash runs/run_examples.sh
bash runs/run_bash_agent_demo.sh
bash runs/run_swebench_smoke.sh
```

`runs/run_ci.sh` mirrors the GitHub Actions workflow at
`.github/workflows/ci.yml`: it syncs the dev dependency group, checks Ruff
formatting, runs `ty` on `src/`, and runs the full unittest suite. Use it as
the local pre-push gate.

The focused tests cover the canonical runtime:

```bash
uv run python -m unittest discover -s tests/unit
```

This runs the recipe demo on the canonical runtime:

```bash
bash runs/run_examples.sh
```

This runs a deterministic mini-SWE-style bash-use agent demo:

```bash
bash runs/run_bash_agent_demo.sh
```

This verifies the SWE-bench adapter's local unit-smoke path without installing
SWE-bench or running Docker:

```bash
bash runs/run_swebench_smoke.sh
```

To inspect context management behavior:

```bash
uv run python scripts/run_tiny_demo.py --recipe debate --last-messages 1
```
