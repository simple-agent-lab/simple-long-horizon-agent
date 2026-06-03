# GDPVal

This is a first-version GDPVal integration for Simple Agent Lab.

Scope:

- Solver by default, with optional first-version rubric judge via `--judge`.
- Agent mode: `tool-call-context-managed` only.
- Runtime dependency: standard library plus the installed `simple-agent-lab`
  wheel. The implementation does not import `swalm`.
- Tool surface: local file/read/search/write/edit/bash/todo tools. Web tools
  are intentionally absent in this first version.
- Judge scope: rubric-only, second-stage agent judge. It compares candidate
  deliverables against rubrics and any staged gold/reference files, then writes
  weighted per-rubric scores. It does not yet implement the full swalm
  forward/reverse A/B GSB scoring path.

The host half lives in `evals/gdpval/suite.py`. The container half ships in the
wheel at `simple_agent_lab.evals.suites.gdpval.container`, so Docker backends
run it through the generic eval runner.

## Run

By default, the runner downloads `openai/gdpval` from Hugging Face, uses the
`train` split, and keeps only rows whose `deliverable_files` field is non-empty:

```bash
uv run --with datasets python runs/run_gdpval.py --limit 10
```

Add `--judge` to run the rubric judge after successful solver cases:

```bash
uv run --with datasets python runs/run_gdpval.py --limit 10 --judge
```

To run through Docker with a local GDPVal base image, build the image first:

```bash
bash runs/build_gdpval_agent_base.sh
```

Then pass it to the runner. The host-side command needs the Docker Python SDK
because `LocalDockerBackend` uses docker-py; `--pull never` keeps Docker from
trying to pull the local tag from a registry. Docker runs mount a local
wheelhouse for the `simple-agent-lab` bootstrap; by default this lives at
`evals/out/gdpval/wheelhouse/cp311-manylinux` and is prepared automatically on
first use.

```bash
uv run --with docker --with datasets python runs/run_gdpval.py \
  --backend local-docker \
  --image gdpval-agent-base:latest \
  --pull never \
  --limit 10
```

You can still pass a local JSONL/JSON/Parquet file for debugging:

```bash
uv run python runs/run_gdpval.py path/to/gdpval.jsonl \
  --task-ids 83d10b06-26d1-4636-a32c-23f92c57f30b \
  --reference-root /path/to/reference_task_id_files \
  --max-turns 100 \
  --concurrency 1
```

Parquet input is accepted when the caller environment has `pyarrow` installed.
Hugging Face input uses `datasets`; pass `--hf-cache-dir` to control where the
cache lands.

The runner reads `OPENAI_MODEL`, `OPENAI_AUTH_TOKEN`, and optional
`OPENAI_BASE_URL` from `.env` or the environment. It uses `openai-chat` by
default; pass `--api-kind openai-responses` to use the Responses adapter.
The judge uses the same provider by default; override with `--judge-provider`,
`--judge-api-kind`, `--judge-max-turns`, and `--judge-concurrency`.

Artifacts land under `evals/out/gdpval/<run-id>/<task-id>/out/`:

- `result.json`: manifest plus workspace archive metadata.
- `trajectory.jsonl`: Simple Agent Lab trace.
- `workspace.tar.gz`: archive written through the artifact store in Docker.

With `--judge`, judge artifacts land under
`evals/out/gdpval/<run-id>-judge/<task-id>/out/`, and aggregate files are also
written beside the solver run:

- `evals/out/gdpval/<run-id>/judge_summary.jsonl`
- `evals/out/gdpval/<run-id>/judge_summary.json`

## Input Shape

Rows are normalized from common GDPVal fields:

- `task_id` or `instance_id`
- `prompt` or `prompt_en`
- `reference_files` or `reference_file_urls`
- `deliverable_files`, `deliverable_file_urls`, `rubric_json`, and `rubrics`
  are treated as judge-only fields and are not shown to the solver.

When reading directly from Hugging Face, reference files are downloaded from
`reference_file_urls` and staged privately into `REFERENCE_DIR`. When reading a
local file, use `--reference-root` if the row stores relative paths.
For judge runs, gold deliverables are downloaded from `deliverable_file_urls` by
default; use `--deliverable-root` for local gold deliverable paths.
