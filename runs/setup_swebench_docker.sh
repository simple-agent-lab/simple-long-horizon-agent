#!/usr/bin/env bash
# One-time setup: install Docker deps, start Colima (macOS), and build
# SWE-bench images for a given instance.
#
# Usage:
#   bash runs/setup_swebench_docker.sh <instance-id>
#
# Example:
#   bash runs/setup_swebench_docker.sh sympy__sympy-23824
#
# This script is idempotent: it skips steps that are already done.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

source runs/_python.sh

INSTANCE_ID="${1:?Usage: $0 <instance-id>}"
DATASET="princeton-nlp/SWE-bench_Verified"
SPLIT="test"
INSTANCE_JSONL="evals/out/swebench/verified/instances/${INSTANCE_ID}.jsonl"

# ──────────────────────────────────────────────────────────────────────
# Step 1: Python dependencies
# ──────────────────────────────────────────────────────────────────────
echo "==> [1/5] Checking Python dependencies..."
"${PYTHON[@]}" -c "import datasets; import docker; import swebench" 2>/dev/null \
  || { echo "  Installing SWE-bench extra..."; uv sync --extra swebench; }

# ──────────────────────────────────────────────────────────────────────
# Step 2: Docker runtime
# ──────────────────────────────────────────────────────────────────────
echo "==> [2/5] Checking Docker runtime..."

resolve_docker_host() {
  if [ -z "${DOCKER_HOST:-}" ]; then
    local sock="$HOME/.colima/default/docker.sock"
    if [ -S "$sock" ]; then
      export DOCKER_HOST="unix://$sock"
    fi
  fi
}

resolve_docker_host

if docker info >/dev/null 2>&1; then
  echo "  Docker is running."
elif [ "$(uname -s)" = "Darwin" ]; then
  echo "  Docker not running. Starting Colima with Rosetta..."
  if ! command -v colima >/dev/null 2>&1; then
    echo "  Installing colima + docker CLI via Homebrew..."
    brew install docker colima
  fi
  if colima status 2>/dev/null | grep -q "running"; then
    echo "  Colima already running."
  else
    colima start --cpu 4 --memory 8 --arch aarch64 --vm-type vz --vz-rosetta
  fi
  resolve_docker_host
  if ! docker info >/dev/null 2>&1; then
    echo "Error: Docker still not reachable after starting Colima." >&2
    exit 1
  fi
else
  echo "Error: Docker is not running. Install and start Docker Engine." >&2
  exit 1
fi

# ──────────────────────────────────────────────────────────────────────
# Step 3: Instance data
# ──────────────────────────────────────────────────────────────────────
echo "==> [3/5] Checking instance data..."
mkdir -p "$(dirname "$INSTANCE_JSONL")"

if [ -f "$INSTANCE_JSONL" ]; then
  echo "  Found $INSTANCE_JSONL"
else
  echo "  Fetching $INSTANCE_ID from $DATASET..."
  DATASET="$DATASET" SPLIT="$SPLIT" "${PYTHON[@]}" - "$INSTANCE_ID" "$INSTANCE_JSONL" <<'PY'
import json, os, sys
from pathlib import Path
from datasets import load_dataset

instance_id, out_path = sys.argv[1], Path(sys.argv[2])
dataset = os.environ["DATASET"]
split = os.environ["SPLIT"]
for row in load_dataset(dataset, split=split):
    if row.get("instance_id") == instance_id:
        out_path.write_text(json.dumps(dict(row), ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"  Saved {instance_id} to {out_path}")
        sys.exit(0)
print(f"Error: {instance_id} not found in {dataset} {split}.", file=sys.stderr)
print("Download the full instance manually. See evals/swebench/README.md.", file=sys.stderr)
sys.exit(1)
PY
fi

# ──────────────────────────────────────────────────────────────────────
# Step 4: Build Docker images
# ──────────────────────────────────────────────────────────────────────
echo "==> [4/5] Building SWE-bench Docker images..."

BASE_IMAGE="sweb.base.py.x86_64:latest"

if docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  echo "  Base image exists, skipping."
else
  ARCH=$(uname -m)
  if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
    echo "  Building base image via docker run + commit (Rosetta workaround)..."
    docker rm -f sweb-base-build 2>/dev/null || true
    docker run --platform linux/amd64 --name sweb-base-build \
      -e DEBIAN_FRONTEND=noninteractive -e TZ=Etc/UTC \
      ubuntu:22.04 bash -c '
set -ex
apt update && apt install -y \
  wget git build-essential libffi-dev libtiff-dev \
  python3 python3-pip python-is-python3 jq curl \
  locales locales-all tzdata \
  && rm -rf /var/lib/apt/lists/*
wget -q "https://repo.anaconda.com/miniconda/Miniconda3-py311_23.11.0-2-Linux-x86_64.sh" -O miniconda.sh
bash miniconda.sh -b -p /opt/miniconda3
export PATH=/opt/miniconda3/bin:$PATH
conda init --all
conda config --append channels conda-forge
adduser --disabled-password --gecos "dog" nonroot
echo "DONE"
'
    docker commit \
      --change 'ENV PATH=/opt/miniconda3/bin:$PATH' \
      --change 'ENV TZ=Etc/UTC' \
      --change 'ENV DEBIAN_FRONTEND=noninteractive' \
      sweb-base-build "$BASE_IMAGE"
    docker rm sweb-base-build
    echo "  Base image committed as $BASE_IMAGE"
  else
    echo "  Base image will be built by swebench (x86_64 native)."
  fi
fi

# Build env + instance images via the SWE-bench Python API
"${PYTHON[@]}" - "$INSTANCE_JSONL" <<'PY'
import docker, json, sys

with open(sys.argv[1]) as f:
    instance = json.loads(f.readline())
instance_id = instance["instance_id"]

client = docker.from_env()
from swebench.harness.docker_build import build_instance_images

print(f"  Building env + instance images for {instance_id}...")
build_instance_images(
    client=client,
    dataset=[instance],
    force_rebuild=False,
    max_workers=1,
    tag="latest",
    env_image_tag="latest",
)
print("  All images built successfully.")

from swebench.harness.test_spec.test_spec import make_test_spec
plain_key = make_test_spec(instance).instance_image_key
ns_key = make_test_spec(instance, namespace="swebench").instance_image_key
if plain_key != ns_key:
    client.images.get(plain_key).tag(ns_key.rsplit(":", 1)[0], ns_key.rsplit(":", 1)[1])
    print(f"  Tagged {plain_key} -> {ns_key}")
PY

# ──────────────────────────────────────────────────────────────────────
# Step 5: Prepare wheelhouse
# ──────────────────────────────────────────────────────────────────────
echo "==> [5/5] Preparing wheelhouse..."
WHEELHOUSE="evals/out/swebench/shared/wheelhouse/cp311-manylinux"

if [ -d "$WHEELHOUSE" ] && [ -n "$(ls -A "$WHEELHOUSE" 2>/dev/null)" ]; then
  echo "  Wheelhouse already populated; refreshing project wheel."
  "${PYTHON[@]}" - <<'PY'
from pathlib import Path
from evals.swebench.containerized_agent import prepare_project_wheel
prepare_project_wheel(Path("evals/out/swebench/shared/wheelhouse/cp311-manylinux"))
PY
else
  "${PYTHON[@]}" - <<'PY'
from pathlib import Path
from evals.swebench.containerized_agent import prepare_wheelhouse
prepare_wheelhouse(Path("evals/out/swebench/shared/wheelhouse/cp311-manylinux"))
PY
fi

echo ""
echo "Setup complete! Run the agent with:"
echo "  bash runs/run_swebench_container.sh $INSTANCE_ID"
