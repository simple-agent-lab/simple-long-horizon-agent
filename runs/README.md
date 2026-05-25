# Runs

This directory contains small shell scripts for reproducible example runs.

The style follows nanochat's `runs/` convention: a script should be readable, copy-pasteable, and useful as the reference way to run an experiment.

## Available Runs

```bash
bash runs/run_ci.sh
bash runs/run_docs_lint.sh
bash runs/run_bash_agent_demo.sh
bash runs/run_swebench_smoke.sh
bash runs/run_swebench_verified.sh
bash runs/run_swebench_pro.sh
bash runs/eval_swebench.sh
```

`runs/run_ci.sh` mirrors the GitHub Actions workflow at
`.github/workflows/ci.yml`: it syncs the dev dependency group, checks Ruff
formatting, runs docs lint, runs `ty` on `src/`, runs the full unittest suite,
and runs the deterministic bash-agent demo smoke. Use it as the local pre-push
gate.

The focused tests cover the canonical runtime:

```bash
uv run python -m unittest discover -s tests/unit
```

This checks local Markdown links and backticked path references:

```bash
bash runs/run_docs_lint.sh
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

These run the containerized SWE-bench agent for one default instance, one named
instance, or the full dataset split. Full-split runs use `--all` and can limit
Docker/model concurrency with `--parallel N`:

```bash
bash runs/run_swebench_verified.sh
bash runs/run_swebench_verified.sh sympy__sympy-23824
bash runs/run_swebench_verified.sh --all --parallel 4

bash runs/run_swebench_pro.sh
bash runs/run_swebench_pro.sh instance_navidrome__navidrome-8e640bb8580affb7e0ea6225c0bbe240186b6b08
bash runs/run_swebench_pro.sh --all --parallel 4
```

Both scripts cache downloaded instance records under `evals/out/`, write per-run
container logs under `evals/out/swebench_container_runs/`, and collect generated
prediction rows into `evals/out/<run-id>_predictions.jsonl`. When records are
not cached, the scripts use `uv run --with datasets python` to fetch the
HuggingFace dataset rows without making `datasets` a project dependency.

This wrapper runs or normalizes official SWE-bench / SWE-bench Pro evaluation
results for an existing predictions file:

```bash
bash runs/eval_swebench.sh --run-official --predictions evals/out/swebench_predictions.jsonl
bash runs/eval_swebench.sh --pro --predictions evals/out/pro-20260525-120000_predictions.jsonl --results-json evals/out/swebench_pro_eval/eval_results.json
```
