# GDPVal

This directory contains the GDPVal integration for Simple Agent Lab.

Scope:

- Solver by default, with optional second-stage judge via `--judge`.
- Agent: a plain Simple Agent Lab `make_llm_agent` tool-calling agent.
- GDPVal source mode: rows are normalized for the `tool-call-context-managed`
  setting, but this integration does not add a custom summarizer or context
  compression layer.
- Runtime dependency: standard library plus the installed `simple-agent-lab`
  wheel. The implementation does not import `swalm`.
- Solver tool surface: `execute_bash`, `TodoWrite`, `multi_edit_file`,
  `view_image`, plus optional `WebSearch` and `WebFetch`. `WebSearch` uses
  Serper (`SERPER_API_KEY`); `WebFetch` uses Jina Reader (`JINA_API_KEY`
  optional). Pass `--disable-web-tools` for offline/controlled runs.
- Judge scope: `--judge-mode gsb` compares candidate deliverables against
  `deliverable_files` with forward/reverse A/B GSB scoring. `--judge-mode
  rubric` uses the direct rubric score path.

The host half lives in `evals/gdpval/suite.py`. The container half ships in the
wheel at `simple_agent_lab.evals.suites.gdpval.container`, so Docker backends
run it through the generic eval runner.

## Run

By default, the runner downloads `openai/gdpval` from Hugging Face, uses the
`train` split, keeps only rows whose `deliverable_files` field is non-empty,
and skips known rows whose gold deliverables are unreadable:

```bash
uv run --with datasets python runs/run_gdpval.py --limit 10
```

Use `--task-ids-file evals/gdpval/task_ids/gdpval_181.txt` to select the saved
181-id GDPVal task list for repeatable evaluations. The normal known-bad filter
still applies unless `--include-known-bad-tasks` is also passed.

By default the solver may use `WebSearch` and `WebFetch` when external public
information is needed. Set `SERPER_API_KEY` to enable Serper search. Jina fetch
works through `https://r.jina.ai/` by default and can use `JINA_API_KEY` or
`JINA_ENDPOINT` from the environment. To keep the solver fully offline, pass:

```bash
uv run --with datasets python runs/run_gdpval.py --limit 10 --disable-web-tools
```

Known-bad GDPVal rows are excluded so default full runs do not hang on broken
gold artifacts. As of this integration, `0e386e32-df20-4d1f-b536-7159bc409ad5`
is skipped because its standard-answer `PrivateCrypMixV2.zip` object is not a
valid zip archive. Pass `--include-known-bad-tasks` only when you explicitly
need to reproduce the raw upstream row set.

Add `--judge` to run the GSB judge after successful solver cases:

```bash
uv run --with datasets python runs/run_gdpval.py --limit 10 --judge
```

The judge tool surface is shared by `--judge-mode gsb` and
`--judge-mode rubric`. It defaults to `--judge-tool-mode hybrid`: local Excel
inspection helpers are available, and local stdio MCP servers are added when
the image can start them. Only read/inspection MCP tools are exposed to the
model; MCP tools that write, edit, create, delete, move, or format files are
filtered out. Use `--judge-tool-mode local` to disable MCP tools, or
`--judge-tool-mode mcp` to require MCP startup and fail loudly if a server is
missing.

Use `--judge-mode rubric` for the direct rubric judge:

```bash
uv run --with datasets python runs/run_gdpval.py --limit 10 --judge \
  --judge-mode rubric
```

To run through Docker with a local GDPVal base image, build the image first:

```bash
bash runs/build_gdpval_agent_base.sh
```

For a full direct build that includes the judge MCP runtimes and starts from
`python:3.11-slim-bookworm`, use `Dockerfile.full`:

```bash
docker build \
  -f docker/gdpval-agent-base/Dockerfile.full \
  -t gdpval-agent-base:latest \
  docker/gdpval-agent-base
```

If your network needs mirrors, pass them as build args. For open-source/default
builds, omit these args and Docker uses the upstream Debian and PyPI sources:

```bash
docker build \
  -f docker/gdpval-agent-base/Dockerfile.full \
  --build-arg DEBIAN_MIRROR=http://mirror.example/debian \
  --build-arg PIP_INDEX_URL=https://pypi.example/simple/ \
  -t gdpval-agent-base:latest \
  docker/gdpval-agent-base
```

The full image includes the common GDPVal task libraries, LibreOffice,
PDF/OCR/font system packages, Node 22, npm document packages, and the local MCP
server runtimes used by the GSB judge. Smoke check it with:

```bash
docker run --rm gdpval-agent-base:latest sh -lc '
python - <<'"'"'PY'"'"'
import pandas, scipy, openpyxl, pypdf, docx, pptx, PIL, reportlab
import mcp, fastmcp, pytesseract
print("python imports ok")
PY
node --version
npm --version
which excel-mcp-server word_mcp_server ppt_mcp_server pdf-reader-mcp mcp-server-filesystem
'
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
`OPENAI_BASE_URL` from `.env` or the environment. It uses `openai-responses` by
default; pass `--api-kind openai-chat` to use the Chat Completions adapter.
The judge uses the same provider by default; override solver and judge settings
separately with `--solver-model`, `--solver-api-key`, `--solver-base-url`,
`--judge-model`, `--judge-api-key`, `--judge-base-url`, `--judge-provider`,
`--judge-api-kind`, `--judge-max-turns`, `--judge-concurrency`, `--judge-mode`,
and `--judge-tool-mode`.

Before a Docker judge run that uses MCP tools, smoke-test the image:

```bash
docker run --rm -i \
  -v "$PWD:/repo" -w /repo -e PYTHONPATH=/repo/src \
  gdpval-agent-base:latest \
  python runs/smoke_gdpval_mcp.py
```

The smoke test starts the filesystem, PDF, Excel, Word, and PowerPoint MCP
servers, lists their tools, performs one read-only call on sample files, and
checks that the filesystem MCP server denies a read outside WORKDIR/reference
roots. It also verifies that the final AgentTool surface contains only the
GDPVal judge MCP read/inspection allowlist.

Artifacts land under `evals/out/gdpval/<run-id>/<task-id>/out/`:

- `result.json`: manifest plus workspace archive metadata.
- `trajectory.jsonl`: Simple Agent Lab trace.
- `workspace.tar.gz`: archive written through the artifact store in Docker.

With `--judge`, judge artifacts land under
`evals/out/gdpval/<run-id>-judge/<task-id>/out/`, and aggregate files are also
written beside the solver run:

- `evals/out/gdpval/<run-id>/judge_summary.jsonl`
- `evals/out/gdpval/<run-id>/judge_summary.json`

GSB judge result files include `combined_weighted_score`, `llm_score`,
`score_process`, forward/reverse rubric GSB details, and the raw parsed
direction payloads under `rm_eval_result`.

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
