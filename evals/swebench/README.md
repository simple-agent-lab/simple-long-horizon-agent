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
- Training/export flows should build from shared trajectory records after eval
  labels exist.

## Quick Start

One-time setup and a single-instance run from scratch:

```bash
# 1. Install Python deps (datasets + Docker SDK + SWE-bench)
uv sync --extra swebench

# 2. Set up Docker (see "Docker Setup" below for details)
bash runs/setup_swebench_docker.sh sympy__sympy-23824

# 3. Configure .env with your model provider
cat > .env <<'EOF'
OPENAI_MODEL=your-model-name
OPENAI_AUTH_TOKEN=your-api-key
OPENAI_BASE_URL=https://your-provider/v1
API_KIND=openai-chat
EOF

# 4. Fetch an instance, build images, and run the agent
bash runs/run_swebench_container.sh sympy__sympy-23824
```

## Docker Setup

SWE-bench runs agents inside Docker containers with pre-installed repo
dependencies. Docker is required for the containerized path.

### macOS (Apple Silicon / arm64)

Docker Desktop requires sudo and has licensing constraints. Colima is a
lightweight alternative that works without either:

```bash
brew install docker colima
```

SWE-bench images are x86_64. On Apple Silicon, Colima must start with
**Rosetta 2** emulation (not QEMU — QEMU fails on Miniconda's embedded
binaries):

```bash
colima start --cpu 4 --memory 8 --arch aarch64 --vm-type vz --vz-rosetta
```

Set the Docker socket for all subsequent commands:

```bash
export DOCKER_HOST=unix://$HOME/.colima/default/docker.sock
```

**Known issue: `docker build` + Rosetta.** The Docker BuildKit builder fails
to run x86_64 ELF binaries through Rosetta during image builds (the dynamic
linker `/lib64/ld-linux-x86-64.so.2` is not accessible in the build overlay).
The same binaries work fine in `docker run`. The setup script works around this
by building the SWE-bench base image via `docker run` + `docker commit` instead
of `docker build`. The SWE-bench env and instance layers build normally on top
of the committed base.

### Linux (x86_64)

Standard Docker Engine works directly:

```bash
sudo apt-get install docker.io
# or: curl -fsSL https://get.docker.com | sh
```

No emulation or workarounds needed.

## Instance Data

SWE-bench instance records can be fetched from HuggingFace. The instance JSONL
must include `repo`, `instance_id`, `base_commit`, `problem_statement`, and the
private fields (`patch`, `test_patch`, `FAIL_TO_PASS`, `PASS_TO_PASS`) needed by
`make_test_spec` for image builds.

Fetch a single instance using the HuggingFace datasets API:

```bash
uv run python - <<'PY'
import json
from pathlib import Path
from datasets import load_dataset
ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
row = ds.filter(lambda x: x["instance_id"] == "sympy__sympy-23824")[0]
out = Path("evals/out/swebench/instance_sympy__sympy-23824.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(dict(row), ensure_ascii=False) + "\n", encoding="utf-8")
print("Saved instance to evals/out/swebench/instance_sympy__sympy-23824.jsonl")
PY
```

Or use the REST API without installing `datasets`:

```bash
mkdir -p evals/out/swebench
curl -s 'https://datasets-server.huggingface.co/rows?dataset=princeton-nlp/SWE-bench_Verified&config=default&split=test&offset=0&length=500' \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
for row in data.get('rows', []):
    r = row.get('row', {})
    if r.get('instance_id') == 'sympy__sympy-23824':
        print(json.dumps(r, ensure_ascii=False))
        break
" > evals/out/swebench/instance_sympy__sympy-23824.jsonl
```

## Building Docker Images

SWE-bench images are built locally (not available on Docker Hub). Three layers:

1. **Base image** (`sweb.base.py.x86_64:latest`): Ubuntu 22.04 + Miniconda +
   build tools.
2. **Env image** (`sweb.env.py.x86_64.<hash>:latest`): base + conda environment
   matching the instance's Python version and repo dependencies.
3. **Instance image** (`sweb.eval.x86_64.<instance-id>:latest`): env + repo
   cloned at `base_commit`.

On x86_64 Linux, the standard build works:

```bash
uv run python - <<'PY'
import docker, json
from swebench.harness.docker_build import build_instance_images
with open("evals/out/swebench/instance_sympy__sympy-23824.jsonl") as f:
    instance = json.loads(f.readline())
client = docker.from_env()
build_instance_images(client=client, dataset=[instance], tag="latest", env_image_tag="latest")
PY
```

On macOS arm64 with Colima + Rosetta, the base image must be built via the
`docker run` + `docker commit` workaround. The setup script
`runs/setup_swebench_docker.sh` handles this automatically.

After building, tag the instance image for the `swebench` namespace that
`containerized_agent.py` expects:

```bash
docker tag sweb.eval.x86_64.sympy__sympy-23824:latest \
  "swebench/sweb.eval.x86_64.sympy_$(echo sympy__sympy-23824 | md5sum | head -c4)_sympy-23824:latest"
```

The convenience scripts handle this automatically.

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
prepare_wheelhouse(Path("evals/out/swebench/wheelhouse/cp311-manylinux"))
PY
```

The container launcher refreshes the local `simple-agent-lab` wheel every time
it mounts a wheelhouse. This keeps cached third-party wheels reusable while
preventing the container from installing an older build of the current checkout.
If you see an import error for a symbol that exists in `src/simple_agent_lab/`,
rerun the container command; the launcher should rebuild the project wheel
before starting Docker.

The core runtime and normal CI do not require Docker or SWE-bench. To run the
containerized SWE-bench adapters, install the optional SWE-bench dependencies in
your local environment and make sure Docker is running:

```bash
uv sync --extra swebench
docker info
```

Use `.env` for provider settings:

```bash
OPENAI_MODEL=...
OPENAI_AUTH_TOKEN=...
OPENAI_BASE_URL=https://api.openai.com/v1
API_KIND=openai-chat
```

`OPENAI_BASE_URL` is optional. `API_KIND` is optional and defaults to
`openai-chat`; set it to `openai-responses` to use the OpenAI Responses API.
The launcher also passes `NO_PROXY` and `no_proxy` through when they exist.

The recommended entry points are the run scripts. With no instance argument,
each script runs a small default instance. Passing one instance id runs that
instance. Passing `--all` runs the full dataset split; use `--parallel N` to
limit concurrent Docker/model runs:

```bash
bash runs/run_swebench_verified.sh
bash runs/run_swebench_verified.sh sympy__sympy-23824
bash runs/run_swebench_verified.sh --all --parallel 4

bash runs/run_swebench_pro.sh
bash runs/run_swebench_pro.sh instance_navidrome__navidrome-8e640bb8580affb7e0ea6225c0bbe240186b6b08
bash runs/run_swebench_pro.sh --all --parallel 4
```

The scripts keep each suite under its own flat output root. SWE-bench Verified
uses `evals/out/swebench/`; SWE-bench Pro uses the sibling
`evals/out/swebench_pro/`. Each root has `instance_<id>.jsonl` caches,
`wheelhouse/` provider wheels, and `<run-id>/<instance-id>/` per-instance run
outputs. When instance records are not cached, the scripts fetch HuggingFace
rows with the `datasets` package from `uv sync --extra swebench`.

The lower-level launcher is still useful when you already have a prepared
instance JSONL and want full control over arguments:

```bash
uv run python evals/swebench/containerized_agent.py \
  --instance-json evals/out/swebench/instance_sympy__sympy-23824.jsonl \
  --instance-id sympy__sympy-23824 \
  --dataset-name princeton-nlp/SWE-bench_Verified \
  --split test \
  --model-name simple-agent-lab-local \
  --provider openai \
  --api-kind openai-chat \
  --dotenv .env \
  --max-turns 20 \
  --run-id my-run \
  --network-mode host \
  --force
```

Or use the convenience script:

```bash
bash runs/run_swebench_container.sh sympy__sympy-23824
```

Outputs land under
`evals/out/swebench/<run-id>/<instance-id>/out/`:

- `trajectory.jsonl`: full agent trajectory (messages, events, model turns).
- `prediction.jsonl`: SWE-bench prediction record with `model_patch`.

The official judge should still run in a separate clean container using the
generated `prediction.jsonl`.

## Official Harness

SWE-bench itself remains an optional dependency because it is heavy and uses
Docker. Install the extra in the repo Python environment before running official
evaluation:

```bash
uv sync --extra swebench
```

Verify the official setup with the gold patch smoke:

```bash
bash runs/run_swebench_gold_smoke.sh
```

Then evaluate local predictions:

```bash
bash runs/eval_swebench.sh \
  --run-official \
  --predictions evals/out/swebench_predictions.jsonl \
  --instance-ids sympy__sympy-23824
```

For SWE-bench Pro predictions, pass `--pro` and either run the official Pro
harness or normalize an existing Pro result file:

```bash
bash runs/eval_swebench.sh --pro \
  --predictions evals/out/swebench_pro/swebench_pro_predictions.jsonl \
  --results-json evals/out/swebench_pro_eval/eval_results.json
```

Official SWE-bench Pro evaluation additionally requires a local checkout of
`scaleapi/SWE-bench_Pro-os`. The current default expects it at
`/tmp/SWE-bench_Pro-os`:

```bash
git clone https://github.com/scaleapi/SWE-bench_Pro-os.git /tmp/SWE-bench_Pro-os
```

If your checkout is elsewhere, pass both Pro harness paths explicitly:

```bash
bash runs/eval_swebench.sh --pro --run-official \
  --predictions evals/out/swebench_pro/swebench_pro_predictions.jsonl \
  --instances evals/out/swebench_pro/instance_all-test.jsonl \
  --pro-eval-script /path/to/SWE-bench_Pro-os/swe_bench_pro_eval.py \
  --scripts-dir /path/to/SWE-bench_Pro-os/run_scripts
```

Official prediction rows must contain:

```json
{"instance_id": "sympy__sympy-23824", "model_name_or_path": "simple-agent-lab", "model_patch": "diff --git ..."}
```

Official harness outputs are intentionally run from
`evals/out/swebench_official/<run-id>/` so summary JSON, harness logs, and
report files stay under the ignored eval output tree instead of the repo root.
When calling `evaluate_predictions.py --run-official`, the default report
directory is `<official-output-dir>/<run-id>/reports`; override
`--official-output-dir`
only when you want a different local artifact root.

Do not pass SWE-bench gold `patch` or `test_patch` fields into the model-visible
task. They belong to the official harness and scoring path, not trajectory
collection.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `rosetta error: failed to open elf` during `docker build` | Rosetta + BuildKit overlay incompatibility | Use `runs/setup_swebench_docker.sh` (builds base via `docker run` + `docker commit`) |
| `CondaHTTPError: HTTP 000 CONNECTION FAILED` | Container has no network | Check `colima status`; restart with `colima start` |
| `exit code 255` from Miniconda installer | QEMU x86_64 emulation failure | Switch to Rosetta: `colima start --vm-type vz --vz-rosetta` |
| `Missing SWE-bench image swebench/sweb.eval...` | Namespace mismatch between built images and `containerized_agent.py` | Pass matching `--namespace` or re-tag images (convenience scripts handle this) |
| `OPENAI_AUTH_TOKEN` / `OPENAI_MODEL` missing | `.env` not configured | Create `.env` with provider credentials (see above) |
