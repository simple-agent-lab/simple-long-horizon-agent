#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${1:-gdpval-agent-base:latest}"
DEBIAN_MIRROR="${DEBIAN_MIRROR:-http://mirrors.aliyun.com}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"

docker build \
  -f "$ROOT/docker/gdpval-agent-base/Dockerfile.full" \
  --build-arg "DEBIAN_MIRROR=$DEBIAN_MIRROR" \
  --build-arg "PIP_INDEX_URL=$PIP_INDEX_URL" \
  -t "$IMAGE_TAG" \
  "$ROOT/docker/gdpval-agent-base"
