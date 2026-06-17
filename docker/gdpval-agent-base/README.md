# GDPVal Agent Base Image

This image is the GDPVal sandbox base for Simple Agent Lab Docker runs. It
starts from `python:3.11-slim-bookworm`, adds shell/search utilities, and
installs common Python, Office/PDF, OCR, Node, and MCP runtimes used by GDPVal
solver and judge runs.

Docker repository names must be lowercase, so use the tag
`gdpval-agent-base:latest` even though the image label is `GDPval-Agent-Base`.

Build it from the repo root:

```bash
bash runs/build_gdpval_agent_base.sh
```

The build script uses
[`Dockerfile.full`](Dockerfile.full), the single complete Dockerfile for this
image.

The build script defaults to `DEBIAN_MIRROR=http://mirrors.aliyun.com` for apt
and `PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/` for pip. Use the
official sources instead with:

```bash
DEBIAN_MIRROR= PIP_INDEX_URL= bash runs/build_gdpval_agent_base.sh
```

Smoke check:

```bash
docker run --rm gdpval-agent-base:latest python - <<'PY'
import pandas, scipy, openpyxl, pypdf, docx, pptx, PIL, reportlab
print("gdpval-agent-base imports ok")
PY
```

MCP runtime smoke check:

```bash
docker run --rm gdpval-agent-base:latest sh -lc \
  'node --version && npm --version && which excel-mcp-server word_mcp_server ppt_mcp_server pdf-reader-mcp mcp-server-filesystem'
```

Use it with the GDPVal runner:

```bash
uv run --with docker --with datasets \
  python runs/run_gdpval.py \
  --backend local-docker \
  --image gdpval-agent-base:latest \
  --pull never \
  --limit 10
```

The runner mounts a local wheelhouse at `/agent/wheelhouse` for the
`simple-agent-lab` bootstrap. By default it uses
`evals/out/gdpval/wheelhouse/cp311-manylinux` and prepares it on first use.
Add the same `--judge` and judge flags you use for normal GDPVal runs.
