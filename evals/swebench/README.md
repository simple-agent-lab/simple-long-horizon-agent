# SWE-bench Eval Adapter

This directory is the first scene-level evaluation suite adapter.

The adapter keeps the existing Simple Agent Lab split:

- `collect_trajectories.py` records what happened and writes the official
  SWE-bench prediction JSONL shape.
- `prepare_workspace.py` creates the per-instance repository workspace the
  agent may inspect and edit.
- `evaluate_predictions.py` runs or normalizes the official SWE-bench harness
  result into `EvalResult` records.
- Training examples are still exported by the shared
  `scripts/export_training_examples.py` script after eval labels exist.

## Local Adapter Smoke

This command does not install SWE-bench and does not run Docker. It checks that
the local adapter can produce trajectory, prediction, and eval-result records:

```bash
bash runs/run_swebench_smoke.sh
```

The default smoke prediction is an empty patch and the eval row is explicitly
marked as a missing official report. Do not treat it as a benchmark score.

## Workspace Run Shape

The real SWE-bench path has two separate workspaces:

```text
agent workspace:
  checkout repo at base_commit
  let the Simple Agent Lab agent inspect/edit files
  collect git diff as model_patch

official judge:
  rebuild a clean SWE-bench environment
  apply model_patch
  run official tests
  emit report
```

Prepare an agent workspace from a SWE-bench instance record:

```bash
uv run python evals/swebench/prepare_workspace.py \
  --instance-json path/to/instances.jsonl \
  --instance-id sympy__sympy-20590
```

Then collect a trajectory and prediction from that workspace:

```bash
uv run python evals/swebench/collect_trajectories.py \
  --instance-json path/to/instances.jsonl \
  --instance-id sympy__sympy-20590 \
  --workspace evals/out/swebench_workspaces/sympy__sympy-20590/repo
```

The current workspace collector is intentionally minimal. It exercises the same
runtime/tool/trajectory path that real agents will use, and the final
`model_patch` is taken from `git diff` in the prepared repository.

## Official Harness

SWE-bench itself remains an optional external dependency because it is heavy and
uses Docker. Install it in the repo Python environment before running official
evaluation:

```bash
git clone https://github.com/princeton-nlp/SWE-bench.git /tmp/SWE-bench
uv pip install -e /tmp/SWE-bench
```

Verify the official setup with the gold patch smoke:

```bash
bash runs/run_swebench_gold_smoke.sh
```

Then evaluate local predictions:

```bash
uv run python evals/swebench/evaluate_predictions.py \
  --run-official \
  --predictions evals/out/swebench_predictions.jsonl \
  --instance-ids sympy__sympy-20590
```

Official prediction rows must contain:

```json
{"instance_id": "sympy__sympy-20590", "model_name_or_path": "simple-agent-lab", "model_patch": "diff --git ..."}
```

Do not pass SWE-bench gold `patch` or `test_patch` fields into the model-visible
task. They belong to the official harness and scoring path, not trajectory
collection.
