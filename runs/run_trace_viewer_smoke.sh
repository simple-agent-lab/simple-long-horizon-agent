#!/usr/bin/env bash
# Headless page smoke for studio/trace-viewer (RPA via puppeteer-core).
#
# Starts serve.py on a free port, copies a sample trajectory into evals/out so
# the experiments sidebar can be exercised, runs smoke.mjs, then stops the server.
#
# Usage:
#   bash runs/run_trace_viewer_smoke.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VIEWER_DIR="${ROOT}/studio/trace-viewer"
PORT="${TRACE_VIEWER_PORT:-9876}"
BASE_URL="http://127.0.0.1:${PORT}"
export TRACE_VIEWER_URL="${BASE_URL}"

source "${ROOT}/runs/_python.sh"

if ! command -v node >/dev/null 2>&1; then
  echo "error: node is required for trace viewer smoke tests" >&2
  exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "error: npm is required for trace viewer smoke tests" >&2
  exit 1
fi

mkdir -p "${ROOT}/evals/out/_smoke"
cp "${VIEWER_DIR}/sample-trace.jsonl" "${ROOT}/evals/out/_smoke/trajectory.jsonl"

printf '=== trace viewer smoke: install node deps ===\n'
(cd "${VIEWER_DIR}" && npm ci)

printf '\n=== trace viewer smoke: start server on %s ===\n' "${PORT}"
(
  cd "${ROOT}"
  "${PYTHON[@]}" "${VIEWER_DIR}/serve.py" --port "${PORT}" --dir "${ROOT}/evals/out"
) &
SERVER_PID=$!

cleanup() {
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for _ in $(seq 1 50); do
  if curl -sf "${BASE_URL}/api/info" >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done
if ! curl -sf "${BASE_URL}/api/info" >/dev/null 2>&1; then
  echo "error: trace viewer server did not become ready on ${BASE_URL}" >&2
  exit 1
fi

printf '\n=== trace viewer smoke: headless page checks ===\n'
(cd "${VIEWER_DIR}" && npm run smoke)

printf '\nTrace viewer smoke passed.\n'
