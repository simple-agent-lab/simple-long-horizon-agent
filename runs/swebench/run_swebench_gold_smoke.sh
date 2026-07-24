#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT="$PWD"
RUN_ID="validate-gold"
OFFICIAL_OUTPUT_DIR="$ROOT/evals/out/swebench_official/$RUN_ID"

source runs/lib/_python.sh

"${PYTHON[@]}" - <<'PY'
import importlib.util
import subprocess

if importlib.util.find_spec("swebench") is None:
    raise SystemExit(
        "SWE-bench is not installed. See evals/swebench/README.md for setup."
    )

try:
    subprocess.run(
        ["docker", "info"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
except Exception as exc:
    raise SystemExit(
        "Docker is not available or not running. Start Docker before the gold smoke."
    ) from exc
PY

mkdir -p "$OFFICIAL_OUTPUT_DIR"

(
  cd "$OFFICIAL_OUTPUT_DIR"
  "${PYTHON[@]}" -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --split test \
    --max_workers 1 \
    --instance_ids sympy__sympy-23824 \
    --predictions_path gold \
    --run_id "$RUN_ID" \
    --report_dir "$OFFICIAL_OUTPUT_DIR/reports"
)
