# Evals

This directory holds higher-level behavior checks and comparison cases.

Unit tests live in `tests/`. Evals are for feedback that compares agent
behavior across prompts, recipes, context views, runtime designs, or model
adapters.

Trajectory collection for shared demos can live outside this directory because
trajectories are fact records, not scores. Scene-level suite adapters can keep
their collector beside their scorer when that makes the suite easier to read.
The first suite example is the local SWE-bench adapter smoke check:

```bash
uv run python -m unittest \
  tests.unit.test_swebench_patch_extract \
  tests.unit.test_swebench_containerized_agent \
  tests.unit.test_swebench_evaluate_predictions
```

That script runs the focused patch extraction, containerized-agent, and
prediction-evaluation unit tests without installing SWE-bench or running
Docker.

The second suite is the context-compression validation in
[`compression/`](compression/). It grades the two shipped strategies
(`ToolCompactStrategy`, `SummarizeStrategy`) on effectiveness rather than
mechanics: threshold-trigger behavior, real compression ratios, tool-pair
and pinned-kind safety, how close the char/4 token estimate is to real
provider tokens (the number `threshold_tokens` is really measured in), and
how many durable facts survive a real summary. The offline half runs in CI
(`tests.unit.test_compression_eval`); the live half needs an OpenAI provider:

```bash
bash runs/run_compression_eval.sh          # offline only, no model
bash runs/run_compression_eval.sh --live   # + token-estimate + summary fidelity
```

Generated files under `evals/out/` are local artifacts and are ignored by git.
Each subdirectory keeps a committed README describing the expected output layout
so that every user sees the same directory skeleton after cloning. See
[`evals/out/README.md`](out/README.md) for the full structure.

Record types:

- `trajectory`: raw messages/events/model turns for one run, written by scripts
- `eval_result`: pass/fail score and compact metrics, written by evals
- `training_example`: model-visible input plus assistant output, written by export scripts

Evals should help answer:

- Does the agent follow the expected loop?
- Does tool use behave predictably?
- Can we score model-call trajectories without scraping terminal output?
- Are changes making the system easier or harder to understand?
- Do reference architecture ideas improve the project in practice?
