# Runs

Launch surface for the project: one Python entry over every benchmark, plus
small copy-pasteable shell scripts (nanochat's `runs/` convention — readable and
useful as the reference way to run an experiment), grouped by concern.

## Layout

```
runs/
  run_bench.py               # the one entry over every bench (list/setup/<bench>/score/oracle/all)
  bench-manifest.example.json
  profiles/                  # per-bench run-profiles (*.example.json)
  _benches/                  # internal per-bench modules (imported by run_bench.py)
  lib/                       # shared sourced helpers (_python.sh, _swebench_uv.sh)
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
bash runs/demos/run_evolve_demo.sh
bash runs/swebench/run_swebench.sh
bash runs/swebench/run_swebench.sh --variant multilingual
bash runs/swebench/run_swebench.sh --variant pro
bash runs/swebench/eval_swebench.sh
```

`runs/dev/run_ci.sh` mirrors the GitHub Actions workflow at
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
instance, or the full dataset split. Full-split runs use `--all` and can limit
Docker/model concurrency with `--parallel N`:

```bash
bash runs/swebench/run_swebench.sh
bash runs/swebench/run_swebench.sh sympy__sympy-23824
bash runs/swebench/run_swebench.sh --all --parallel 4

bash runs/swebench/run_swebench.sh --variant multilingual
bash runs/swebench/run_swebench.sh --variant multilingual --all --parallel 4

bash runs/swebench/run_swebench.sh --variant pro
bash runs/swebench/run_swebench.sh --variant pro instance_navidrome__navidrome-8e640bb8580affb7e0ea6225c0bbe240186b6b08
bash runs/swebench/run_swebench.sh --variant pro --all --parallel 4
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
bash runs/swebench/eval_swebench.sh --run-official --predictions evals/out/swebench_predictions.jsonl
bash runs/swebench/eval_swebench.sh --multilingual --predictions evals/out/swebench_multilingual/swebench_multilingual_predictions.jsonl
bash runs/swebench/eval_swebench.sh --pro --predictions evals/out/swebench_pro/swebench_pro_predictions.jsonl --results-json evals/out/swebench_pro_eval/eval_results.json
```

See `evals/swebench/README.md` for detailed Docker setup, macOS arm64
workarounds, and troubleshooting.
