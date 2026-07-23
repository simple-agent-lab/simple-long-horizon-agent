# Runs

Launch surface for the project: one Python entry over every benchmark, plus
small copy-pasteable shell scripts (nanochat's `runs/` convention — readable and
useful as the reference way to run an experiment), grouped by concern.

## Layout

```
runs/
  run_bench.py               # one entry (list/setup/<bench>/batch/score/oracle/all)
  bench-manifest.example.json
  profiles/                  # per-bench run-profiles (*.example.json)
  _benches/                  # internal per-bench modules (imported by run_bench.py)
  lib/                       # small shared launch helpers
  harbor/                    # Terminal-Bench 2.1 Harbor batch launchers
  swebench/                  # SWE-bench batch/setup/eval scripts
  programbench/              # ProgramBench batch/setup scripts
  demos/                     # bash-agent / MCP / trace-viewer demos
  dev/                       # run_ci.sh, run_docs_lint.sh
```

Start with the unified entry:

```bash
uv run python runs/run_bench.py list            # what benches exist
uv run python runs/run_bench.py setup           # is my environment ready?
uv run python runs/run_bench.py <bench> ...     # run one bench
uv run python runs/run_bench.py batch <bench> ...   # select/run a concurrent batch
uv run python runs/run_bench.py score <bench> ...   # official scorer (or note inline scoring)
uv run python runs/run_bench.py oracle <bench> ...  # gold/model-free reference run (wiring check)
uv run python runs/run_bench.py all --manifest runs/bench-manifest.json
```

`score` reaches a bench's official scorer (SWE-bench / ProgramBench delegate to
their `evals/<suite>/evaluate_*.py`; a bench that grades inline says so and does
nothing). `oracle` is sugar for the run path with `--provider oracle`, applying
the reference solution model-free as a deterministic wiring check (only benches
whose `--provider` accepts `oracle`).

## Available shell runs

```bash
bash runs/dev/run_ci.sh
bash runs/dev/run_docs_lint.sh
bash runs/demos/run_bash_agent_demo.sh
bash runs/harbor/run_terminal_bench_2_1_sequential.sh
bash runs/swebench/run_swebench.sh
bash runs/swebench/run_swebench.sh --variant multilingual
bash runs/swebench/run_swebench.sh --variant pro
bash runs/swebench/eval_swebench.sh
```

`runs/dev/run_ci.sh` mirrors the GitHub Actions workflow at
`.github/workflows/ci.yml`: it syncs the dev dependency group, checks Ruff
formatting and lint, checks docs plus generated references, enforces
architecture and environment-variable boundaries, runs `ty` on `src/`, runs
the full unittest suite, and runs the deterministic bash-agent demo smoke. Use
it as the local pre-push gate.

The Terminal-Bench 2.1 Harbor launchers run `bash` and `bash_task` experiments
with separate job names. The sequential entry waits for the complete blocking
Harbor process for `bash` before starting `bash_task`:

```bash
bash runs/harbor/run_terminal_bench_2_1_bash.sh
bash runs/harbor/run_terminal_bench_2_1_bash_task.sh
bash runs/harbor/run_terminal_bench_2_1_sequential.sh
```

Use `HARBOR_DRY_RUN=1` with any launcher to validate its command without
starting an experiment.

The focused tests cover the canonical runtime:

```bash
uv run python -m unittest discover -s tests/unit
```

This checks local Markdown links and backticked path references:

```bash
bash runs/dev/run_docs_lint.sh
```

This runs a deterministic mini-SWE-style bash-use agent demo:

```bash
bash runs/demos/run_bash_agent_demo.sh
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
bash runs/swebench/setup_swebench_docker.sh sympy__sympy-23824
```

These run the containerized SWE-bench agent for one default instance, one named
instance, a selected subset, or the full dataset split. Subset and full-split
runs can limit Docker/model concurrency with `--parallel N`; subset files use
one instance id per line and may include `#` comments. The shell script is a
six-line compatibility wrapper over `run_bench.py batch swebench`:

```bash
bash runs/swebench/run_swebench.sh
bash runs/swebench/run_swebench.sh sympy__sympy-23824
bash runs/swebench/run_swebench.sh --ids-file evals/out/swebench/ids.txt --parallel 4
bash runs/swebench/run_swebench.sh --all --parallel 4

bash runs/swebench/run_swebench.sh --variant multilingual
bash runs/swebench/run_swebench.sh --variant multilingual --ids-file ids.txt --parallel 4
bash runs/swebench/run_swebench.sh --variant multilingual --all --parallel 4

bash runs/swebench/run_swebench.sh --variant pro
bash runs/swebench/run_swebench.sh --variant pro instance_navidrome__navidrome-8e640bb8580affb7e0ea6225c0bbe240186b6b08
bash runs/swebench/run_swebench.sh --variant pro --ids-file ids.txt --parallel 4
bash runs/swebench/run_swebench.sh --variant pro --all --parallel 4

# The same Python entry, without the compatibility wrapper:
uv run --extra swebench python runs/run_bench.py batch swebench \
  --variant pro --all --parallel 4
```

SWE-bench Verified artifacts land under `evals/out/swebench/`; SWE-bench
Multilingual and Pro artifacts land under sibling roots
`evals/out/swebench_multilingual/` and `evals/out/swebench_pro/`. All roots use
the same flat layout: optional `instance_<id>.jsonl` setup inputs,
`wheelhouse/` for wheels, and `<run-id>/<instance-id>/` for per-instance
outputs. The Python batch entry loads rows with the `datasets` package from
`uv sync --extra swebench` (which reuses the HuggingFace cache).

This wrapper runs or normalizes official SWE-bench / SWE-bench Pro evaluation
results for an existing predictions file:

```bash
bash runs/swebench/eval_swebench.sh --run-official --predictions evals/out/swebench_predictions.jsonl
bash runs/swebench/eval_swebench.sh --multilingual --predictions evals/out/swebench_multilingual/swebench_multilingual_predictions.jsonl
bash runs/swebench/eval_swebench.sh --pro --predictions evals/out/swebench_pro/swebench_pro_predictions.jsonl --results-json evals/out/swebench_pro_eval/eval_results.json
```

See `evals/swebench/README.md` for detailed Docker setup, macOS arm64
workarounds, and troubleshooting.
