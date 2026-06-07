#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building clawbase-sal:v1 from clawbase-nanobot:v1..."
docker build -f Dockerfile.clawbase -t clawbase-sal:v1 .

echo "Done. Images:"
docker images | grep clawbase-sal
