#!/usr/bin/env bash
# Run the config-backed DGM self-evolving SWE-bench recipe.

set -euo pipefail
cd "$(dirname "$0")/.."
source runs/_python.sh
source runs/_swebench_uv.sh
source runs/_docker.sh

if command -v uv >/dev/null 2>&1; then
  PYTHON=(uv run --extra swebench python)
fi

EXECUTE=0
for arg in "$@"; do
  if [ "$arg" = "--execute" ]; then
    EXECUTE=1
  fi
done

if [ "$EXECUTE" -eq 1 ]; then
  docker_resolve_host
  if ! docker_ensure_running; then
    echo "ERROR: Docker daemon is not reachable and could not be started automatically." >&2
    echo "Start Docker Desktop or Colima manually, then re-run (or set SAL_DOCKER_AUTOSTART=0 to skip auto-start)." >&2
    exit 1
  fi
  swebench_ensure_linux_uv
  export SWEBENCH_UV_BIN
fi

"${PYTHON[@]}" -m recipes.dgm.evolve "$@"
