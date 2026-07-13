# Runs

This directory contains small shell scripts for reproducible example runs.

The style follows nanochat's `runs/` convention: a script should be readable, copy-pasteable, and useful as the reference way to run an experiment.

## Available Runs

```bash
bash runs/run_ci.sh
bash runs/run_docs_lint.sh
bash runs/run_bash_agent_demo.sh
bash runs/run_swebench_verified.sh
bash runs/run_swebench_multilingual.sh
bash runs/run_swebench_pro.sh
bash runs/eval_swebench.sh
bash runs/run_programbench_suite.sh \
  abishekvashok__cmatrix.5c082c6 --dynamic-workflow
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

To verify SWE-bench adapter tests specifically (already included in `run_ci.sh`):

```bash
uv run python -m unittest \
  tests.unit.test_swebench_patch_extract \
  tests.unit.test_swebench_harness \
  tests.unit.test_swebench_evaluate_predictions
```

## SWE-bench (Containerized)

One-time setup: install Docker, build SWE-bench images for an instance:

```bash
bash runs/setup_swebench_docker.sh sympy__sympy-23824
```

These run the containerized SWE-bench agent for one default instance, one named
instance, or the full dataset split. Full-split runs use `--all` and can limit
Docker/model concurrency with `--parallel N`:

```bash
bash runs/run_swebench_verified.sh
bash runs/run_swebench_verified.sh sympy__sympy-23824
bash runs/run_swebench_verified.sh --all --parallel 4

bash runs/run_swebench_multilingual.sh
bash runs/run_swebench_multilingual.sh --all --parallel 4

bash runs/run_swebench_pro.sh
bash runs/run_swebench_pro.sh instance_navidrome__navidrome-8e640bb8580affb7e0ea6225c0bbe240186b6b08
bash runs/run_swebench_pro.sh --all --parallel 4
```

SWE-bench Verified artifacts land under `evals/out/swebench/`; SWE-bench
Multilingual and Pro artifacts land under sibling roots
`evals/out/swebench_multilingual/` and `evals/out/swebench_pro/`. All roots use
the same flat layout: `instance_<id>.jsonl` for cached records, `wheelhouse/`
for wheels, and `<run-id>/<instance-id>/` for per-instance outputs. When records
are not cached, the scripts use the `datasets` package from `uv sync --extra
swebench` to fetch the HuggingFace dataset rows.

This wrapper runs or normalizes official SWE-bench / SWE-bench Pro evaluation
results for an existing predictions file:

```bash
bash runs/eval_swebench.sh --run-official --predictions evals/out/swebench_predictions.jsonl
bash runs/eval_swebench.sh --multilingual --predictions evals/out/swebench_multilingual/swebench_multilingual_predictions.jsonl
bash runs/eval_swebench.sh --pro --predictions evals/out/swebench_pro/swebench_pro_predictions.jsonl --results-json evals/out/swebench_pro_eval/eval_results.json
```

See `evals/swebench/README.md` for detailed Docker setup, macOS arm64
workarounds, and troubleshooting.

## ProgramBench Dynamic Workflows

The ProgramBench runner can select an agent-written JavaScript workflow while
preserving ProgramBench's sealed network boundary and official workspace
submission format. Use the shell wrapper so macOS hosts get the required Linux
`uv` binary automatically:

```bash
bash runs/run_programbench_suite.sh \
  abishekvashok__cmatrix.5c082c6 \
  --dynamic-workflow
```

See `evals/programbench/README.md` for Node provisioning, workflow controls,
artifact layout, and official scoring.
