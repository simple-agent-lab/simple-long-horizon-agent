#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source runs/_python.sh

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

"${PYTHON[@]}" -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --split test \
  --max_workers 1 \
  --instance_ids sympy__sympy-20590 \
  --predictions_path gold \
  --run_id validate-gold \
  --report_dir evals/out/swebench_official
