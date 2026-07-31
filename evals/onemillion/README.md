# OneMillion-Bench Eval Adapter

OneMillion-Bench is a **rubric-graded Q&A** benchmark across professional domains: a model answers a conceptual prompt, and a *judge* model scores the answer against the case's weighted rubrics.

- `suite.py` — `OneMillionSuite`. Drops the rubrics before the agent sees the case (`task_input`) and stages them as gold scoring inputs (`eval_inputs`) so the container-half `evaluate` hook grades the answer in the run environment.
- `harness.py` — host-side helpers the suite and run entry share: dataset loading (single-object or list JSON files), agent-visible sanitization, dotenv loading, and the generator/judge environment.
- The container half ships in the wheel at `simple_long_horizon_agent.evals.suites.onemillion`:
  - `container.py` — a tool-free LLM agent that answers the prompt and persists its answer to `model_response.txt`; `extract_result` collects it; `evaluate` grades it with the judge.
  - `grading.py` — the rubric prompt, response parser, and weighted scoring **ported verbatim** from `omb.grading` (stdlib-only), so the verdict matches upstream without depending on the vendored checkout.

Unlike SWE-bench there is **no Docker image**: generation is one tool-free model turn graded by a judge, so runs go through `LocalProcessBackend` (in-process).

## Quick Start

```bash
# 1. Download the dataset (see the upstream README) into datasets/OneMillion-Bench/
hf download humanlaya-data-lab/OneMillion-Bench --repo-type=dataset \
  --local-dir datasets/OneMillion-Bench

# 2. Configure .env with the generator (model under test) and the judge.
cat > .env <<'EOF'
# Generator — the model being evaluated
OPENAI_MODEL=your-model-name
OPENAI_AUTH_TOKEN=your-api-key
OPENAI_BASE_URL=https://your-provider/v1
API_KIND=openai-chat
# Judge — the grader (falls back to the OPENAI_* values if omitted)
JUDGE_MODEL=your-judge-model
JUDGE_AUTH_TOKEN=your-judge-key
JUDGE_BASE_URL=https://your-provider/v1
EOF

# 3. Run one case, or a whole domain.
uv run python -m runs.run_bench onemillion case_2860 \
  --dataset datasets/OneMillion-Bench/healthcare_and_medicine

uv run python -m runs.run_bench onemillion --all \
  --dataset datasets/OneMillion-Bench --concurrency 8
```

Outputs land under `evals/out/onemillion/<run-id>/<instance-id>/out/`:

- `trajectory.jsonl` — the generation trajectory (one tool-free model turn).
- `result.json` — the `model_response` plus the rubric verdict (`score`, `total_score` / `max_score`, per-rubric `rubric_scores`, and the judge's per-rubric `judge_verdicts`).

## Environment

| Variable | Role |
| --- | --- |
| `OPENAI_MODEL` / `OPENAI_AUTH_TOKEN` | Generator (model under test). Required. |
| `OPENAI_BASE_URL` / `API_KIND` | Generator endpoint / `openai-chat` vs `openai-responses`. |
| `JUDGE_MODEL` / `JUDGE_AUTH_TOKEN` | Judge (grader). Each falls back to the matching `OPENAI_*`. |
| `JUDGE_BASE_URL` / `JUDGE_API_KIND` | Judge endpoint / API kind (default `openai-chat`). |

Secrets stay in the environment — only the non-secret rubrics and prompt are staged into `input/eval.json` for the judge.

## Local Adapter Smoke

No network and no dataset needed — the focused unit checks for the adapter:

```bash
uv run python -m unittest \
  tests.unit.test_onemillion_grading \
  tests.unit.test_onemillion_container \
  tests.unit.test_onemillion_harness
```

## Scoring

Scoring uses the in-environment `evaluate` hook and is on by default. The host
stages `{prompt, rubrics, human_scores}` via `eval_inputs`; the hook builds the
upstream judge prompt, parses the per-rubric yes/no verdict, and converts hits
to weighted scores. The per-case `score` is `total_score / max_score`
(positive-weight total), matching `omb`'s per-task accuracy. Pass
`--no-scoring` (or `OneMillionSuite(in_env_scoring=False)`) to capture answers
only and grade later.
