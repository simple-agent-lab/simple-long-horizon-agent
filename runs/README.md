# Runs

Small, copy-pasteable entry points for reproducible runs. The style follows
nanochat's `runs/` convention: each script should be readable and usable as the
reference way to run an experiment.

## Conventions

- `run_*.sh` / `run_*.py`, `eval_*.sh`, `setup_*.sh` — entry points you invoke
  directly.
- `_*.sh` (leading underscore) — shared helpers meant to be `source`d from an
  entry point, **not** run on their own:
  - `runs/_python.sh` — pick `uv run python` when `uv` is available, else
    `python3`; sets the `$PYTHON` array.
  - `runs/_swebench_uv.sh` — fetch a Linux `uv` binary for the SWE-bench
    container; sets `$SWEBENCH_UV_BIN`.
  - `runs/_swebench_run.sh` — the shared driver behind the three SWE-bench
    dataset launchers (arg parsing, dataset fetch+cache, the per-instance
    container run, the `--all` batch loop, prediction collection).

## Quality gate

Run before pushing — `runs/run_ci.sh` mirrors `.github/workflows/ci.yml`: it
syncs the dev dependency group, checks Ruff formatting, runs docs lint, runs
`ty` on `src/`, runs the full unittest suite, and runs the deterministic
bash-agent demo smoke.

```bash
bash runs/run_ci.sh
bash runs/run_docs_lint.sh   # local Markdown links + backticked path references
```

The focused unittest suite covers the canonical runtime:

```bash
uv run python -m unittest discover -s tests/unit
```

## Demos

```bash
bash runs/run_bash_agent_demo.sh   # deterministic bash-use agent on the canonical runtime
bash runs/run_mcp_agent_demo.sh    # multimodal MCP tool demo (needs the `mcp` extra)
bash runs/run_trace_viewer.sh      # Observatory trace viewer over evals/out/ (http://127.0.0.1:8765)
```

## SWE-bench (containerized eval)

One-time setup: install Docker, then build the SWE-bench image for an instance:

```bash
bash runs/setup_swebench_docker.sh sympy__sympy-23824
```

Run one instance directly through the framework
(`run_suite_instance(SwebenchSuite, LocalDockerBackend, LocalDirStore)`):

```bash
bash runs/run_swebench_suite.sh sympy__sympy-23824
uv run python runs/run_swebench_suite.py sympy__sympy-23824 --max-turns 20
```

The three dataset launchers are thin wrappers over `runs/_swebench_run.sh` —
each sets only its dataset constants. They run one default instance, one named
instance, or the full split with `--all` (limit concurrency with `--parallel N`):

```bash
bash runs/run_swebench_verified.sh
bash runs/run_swebench_verified.sh sympy__sympy-23824
bash runs/run_swebench_verified.sh --all --parallel 4

bash runs/run_swebench_multilingual.sh --all --parallel 4
bash runs/run_swebench_pro.sh instance_navidrome__navidrome-8e640bb8580affb7e0ea6225c0bbe240186b6b08
```

SWE-bench Verified artifacts land under `evals/out/swebench/`; Multilingual and
Pro under the sibling roots `evals/out/swebench_multilingual/` and
`evals/out/swebench_pro/`. All roots use the same flat layout:
`instance_<id>.jsonl` for cached records, `wheelhouse/` for wheels, and
`<run-id>/<instance-id>/` for per-instance outputs. When records are not cached,
the launchers fetch the HuggingFace rows with the `datasets` package from
`uv sync --extra swebench`.

Score or normalize predictions with the official harness (`runs/eval_swebench.sh`),
or run the gold-patch oracle smoke (`runs/run_swebench_gold_smoke.sh`):

```bash
bash runs/run_swebench_gold_smoke.sh
bash runs/eval_swebench.sh --run-official --predictions evals/out/swebench_predictions.jsonl
bash runs/eval_swebench.sh --multilingual --predictions evals/out/swebench_multilingual/swebench_multilingual_predictions.jsonl
bash runs/eval_swebench.sh --pro --predictions evals/out/swebench_pro/swebench_pro_predictions.jsonl --results-json evals/out/swebench_pro_eval/eval_results.json
```

To verify the SWE-bench adapter tests specifically (also part of `run_ci.sh`):

```bash
uv run python -m unittest \
  tests.unit.test_swebench_patch_extract \
  tests.unit.test_swebench_harness \
  tests.unit.test_swebench_evaluate_predictions
```

See `evals/swebench/README.md` for Docker setup, macOS arm64 workarounds, and
troubleshooting.

## OneMillion-Bench (Docker-free eval)

A light, Docker-free suite run through `LocalProcessBackend`: generation is one
tool-free model turn, graded in-environment by a judge model.

```bash
# one case by id
uv run python runs/run_onemillion_suite.py case_2860 \
  --dataset datasets/OneMillion-Bench/healthcare_and_medicine

# a whole domain (or the full dataset), fanned out
uv run python runs/run_onemillion_suite.py --all \
  --dataset datasets/OneMillion-Bench --concurrency 8
```
