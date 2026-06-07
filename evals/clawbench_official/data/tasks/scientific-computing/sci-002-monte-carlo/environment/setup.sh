#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"

mkdir -p "$WORKSPACE"

# No input files needed for this task, but we create an empty placeholder to satisfy environment if needed
: > "$WORKSPACE/.placeholder"

# We fix the random seed in the solution script, so no data generation needed here
