# SWE-bench Eval Adapter

This directory is the first scene-level evaluation suite adapter.

The adapter keeps the existing Simple Agent Lab split:

- `containerized_agent.py` starts the SWE-bench instance container and runs the
  Simple Agent Lab runner inside it.
- `in_container_runner.py` records what happened and writes the official
  SWE-bench prediction JSONL shape from inside the container.
- `patch_extract.py` collects the final `model_patch` while filtering generated
  files.
- `evaluate_predictions.py` runs or normalizes the official SWE-bench harness
  result into `EvalResult` records.
- Training examples are still exported by the shared
  `scripts/export_training_examples.py` script after eval labels exist.

## Local Adapter Smoke

This command does not install SWE-bench and does not run Docker. It runs the
focused unit-smoke checks for the containerized adapter:

```bash
bash runs/run_swebench_smoke.sh
```

## Containerized Agent

SWE-bench patch generation runs inside the SWE-bench instance container. The
host launcher mounts a run directory and optional wheelhouse, copies in the
small eval runner from `evals/`, installs the `simple-agent-lab` wheel, passes
model environment variables, and collects `prediction.jsonl` plus
`trajectory.jsonl`.

From the agent's point of view, `/testbed` is a normal local repository and the
bash tool is the normal local bash tool. The eval runner stays outside `src/`;
the installed wheel supplies the runtime modules.

The runner collects `model_patch` from a staged git diff after installing
SWALM-style generated-file ignore rules in `.git/info/exclude`; this keeps
build artifacts such as `build/`, `dist/`, `node_modules/`, and compiled
language outputs out of the prediction without adding a `.gitignore` change to
the patch.

Prepare provider wheels once on the host:

```bash
uv run python - <<'PY'
from pathlib import Path
from evals.swebench.containerized_agent import prepare_wheelhouse
prepare_wheelhouse(Path("evals/out/wheelhouse/cp311-manylinux"))
PY
```

Use `.env` for provider settings:

```bash
OPENAI_MODEL=gpt-test-1
OPENAI_AUTH_TOKEN=...
OPENAI_BASE_URL=https://api.openai.com/v1
API_KIND=openai-chat
```

`OPENAI_BASE_URL` is optional. `API_KIND` is optional and defaults to
`openai-chat`; set it to `openai-responses` to use the OpenAI Responses API.
The launcher also passes `NO_PROXY` and `no_proxy` through when they exist.

Run one containerized agent:

```bash
uv run python evals/swebench/containerized_agent.py \
  --instance-json evals/out/swebench_verified_sympy__sympy-23824.jsonl \
  --instance-id sympy__sympy-23824 \
  --dataset-name princeton-nlp/SWE-bench_Verified \
  --split test \
  --model-name simple-agent-lab-containerized-mimo-v2.5-pro \
  --provider openai \
  --api-kind openai-responses \
  --dotenv .env \
  --max-turns 20 \
  --run-id containerized-openai-sympy-23824 \
  --force
```

Outputs land under
`evals/out/swebench_container_runs/<run-id>/<instance-id>/out/`. The official
judge should still run in a separate clean container using the generated
`prediction.jsonl`.

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

Official harness outputs are intentionally run from
`evals/out/swebench_official/<run-id>/` so summary JSON, harness logs, and report
files stay under the ignored eval output tree instead of the repo root. When
calling `evaluate_predictions.py --run-official`, the default report directory
is `<official-output-dir>/<run-id>/reports`; override `--official-output-dir`
only when you want a different local artifact root.

Do not pass SWE-bench gold `patch` or `test_patch` fields into the model-visible
task. They belong to the official harness and scoring path, not trajectory
collection.
