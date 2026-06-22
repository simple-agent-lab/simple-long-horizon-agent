# Runs

This directory contains small shell scripts for reproducible example runs.

The style follows nanochat's `runs/` convention: a script should be readable, copy-pasteable, and useful as the reference way to run an experiment.

## Available Runs

```bash
bash runs/run_ci.sh
bash runs/run_docs_lint.sh
bash runs/run_bash_agent_demo.sh
bash runs/run_self_evolving_simple.sh --run-id simple-smoke
bash runs/run_dgm_swebench.sh --run-id dgm-swebench-smoke
bash runs/run_swebench_verified.sh
bash runs/run_swebench_multilingual.sh
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

### Self-Evolving SWE-bench (simple recipe)

This runs the config-backed simple self-evolving recipe
(`recipes/simple/evolve.py`). The wrapper registers the SWE-bench factories,
uses `configs/simple_swebench.yaml` when `--config` is omitted, and delegates to
the generic self-evolving runner. The default config points at checked-in tiny
train and heldout JSONL examples so the dry-run works in a clean checkout:

```bash
bash runs/run_self_evolving_simple.sh --run-id simple-smoke
```

For a real run, copy or create a config and edit the YAML paths and execution
settings:

```bash
bash runs/run_self_evolving_simple.sh \
  --config configs/my_simple_swebench.yaml \
  --run-id simple-real

bash runs/run_self_evolving_simple.sh \
  --config configs/my_simple_swebench.yaml \
  --run-id simple-real \
  --execute
```

Train and heldout paths, rounds, `evaluation.*`, `parallel` / `max_turns`,
backend options, model settings, and output root live in YAML now. The
`--execute` command starts real model + Docker work, evolves on the train split,
and writes the generic heldout performance report to
`<output_root>/<run-id>/evaluation/summary.json` when heldout evaluation is
enabled. It expects `.env` or the shell environment to provide
`OPENAI_AUTH_TOKEN` and, when needed, `OPENAI_BASE_URL`. For the faithful DGM
variant with all knobs exposed, see the DGM recipe below. Both recipes are
documented under `recipes/`.
The checked-in `configs/examples/swebench_*_tiny.jsonl` paths are dry-run
examples only; both self-evolving recipes reject `--execute` until those paths
are replaced with real SWE-bench JSONL files.

Build the stronger demo split first with `recipes/dgm/ops/baseline.py`; it can fetch
SWE-bench Verified, select a repo-balanced pool, measure seed resolves, and
write disjoint headroom train/test files.

```bash
uv run --extra swebench python recipes/dgm/ops/baseline.py \
  --run-id baseline-demo-160 \
  --pool-size 160 \
  --pool-out evals/out/dgm_swebench/splits/demo-pool-160.jsonl \
  --baseline-out evals/out/dgm_swebench/splits/demo-baseline-160.jsonl \
  --train-out evals/out/dgm_swebench/splits/demo-train-60.jsonl \
  --test-out evals/out/dgm_swebench/splits/demo-test-60.jsonl \
  --train-size 60 \
  --test-size 60 \
  --parallel 3
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

## DGM SWE-bench Recipe

This runs the faithful DGM self-evolving recipe over SWE-bench, with artifacts
under `evals/out/dgm_swebench/` (see `recipes/dgm/README.md` for the full
quick-start and config reference). The default config is
`configs/dgm_swebench.yaml`; copy it for real runs and edit the train/test paths,
rounds, branch count, worker cap, model, and wheelhouse settings there:

```bash
cp configs/dgm_swebench.yaml configs/my_dgm_swebench.yaml
```

```bash
bash runs/run_dgm_swebench.sh \
  --config configs/my_dgm_swebench.yaml \
  --run-id dgm-demo \
  --rounds 2 \
  --parent-selection score_child_prop
```

By default it prints a dry plan. Add `--execute` to run real model + Docker
evolution. Use `--monitor` with the same run id and config to print the current
report. Use the train dataset for evolution and the test dataset for held-out
official scoring; this avoids reporting on the same instances used for
selection. To build a balanced train/test split, see
`recipes/dgm/ops/baseline.py`.
DGM writes scoped official artifacts under `official/baseline/` and
`official/final/`, plus `test_summary.json` with the held-out delta. The simple
wrapper writes the generic evolution workspace, suite run artifacts, and, when
enabled, the suite-scored `evaluation/summary.json` described in its
YAML-backed runner docs.
