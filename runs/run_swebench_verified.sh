#!/usr/bin/env bash
# Run the agent on SWE-bench Verified through the Suite framework (ADR generic-containerized-eval-framework).
#
# Usage:
#   bash runs/run_swebench_verified.sh                          # default: sympy__sympy-23824
#   bash runs/run_swebench_verified.sh django__django-16379     # one custom instance
#   bash runs/run_swebench_verified.sh --all --parallel 4       # full split, 4 at a time
#
# Requires Docker, `uv sync --extra swebench`, and a .env with provider credentials.
# Downloading uncached dataset records uses `datasets`.

set -euo pipefail
cd "$(dirname "$0")/.."
source runs/_python.sh
source runs/_swebench_uv.sh

DATASET="princeton-nlp/SWE-bench_Verified"
SPLIT="test"
DEFAULT_INSTANCE_ID="sympy__sympy-23824"
INSTANCE_DIR="evals/out/swebench"
CONTAINER_RUN_ROOT="evals/out/swebench"
PREDICTION_DIR="evals/out/swebench"
MODEL_NAME="simple-agent-lab-verified"
MAX_TURNS=150
RUN_ID="verified-$(date +%Y%m%d-%H%M%S)"
RUN_ALL=0
PARALLEL=1
POSITIONAL=()
FETCH_PYTHON=()

usage() {
  sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --all)
      RUN_ALL=1
      shift
      ;;
    --parallel)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --parallel requires a positive integer." >&2
        exit 2
      fi
      PARALLEL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if ! [[ "$PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --parallel must be a positive integer; got ${PARALLEL@Q}." >&2
  exit 2
fi
if [ "$RUN_ALL" -eq 1 ] && [ "${#POSITIONAL[@]}" -gt 0 ]; then
  echo "ERROR: pass either --all or one INSTANCE_ID, not both." >&2
  exit 2
fi
if [ "$RUN_ALL" -eq 0 ] && [ "${#POSITIONAL[@]}" -gt 1 ]; then
  echo "ERROR: pass at most one INSTANCE_ID, or use --all for the full split." >&2
  exit 2
fi

ensure_fetch_python() {
  if [ "${#FETCH_PYTHON[@]}" -gt 0 ]; then
    return
  fi
  if command -v uv >/dev/null 2>&1; then
    FETCH_PYTHON=(uv run --extra swebench python)
    return
  fi
  if "${PYTHON[@]}" - <<'PY' >/dev/null 2>&1
import datasets
PY
  then
    FETCH_PYTHON=("${PYTHON[@]}")
    return
  fi
  echo "ERROR: fetching uncached SWE-bench records requires the Python 'datasets' package." >&2
  echo "Install it with: uv sync --extra swebench" >&2
  exit 1
}

mkdir -p "$INSTANCE_DIR"

fetch_one_instance() {
  local instance_id="$1"
  local instance_json="${INSTANCE_DIR}/instance_${instance_id}.jsonl"
  if [ ! -f "$instance_json" ]; then
    ensure_fetch_python
    echo "Fetching instance ${instance_id} from ${DATASET}..." >&2
    DATASET="$DATASET" SPLIT="$SPLIT" INSTANCE_ID="$instance_id" INSTANCE_JSON="$instance_json" \
      "${FETCH_PYTHON[@]}" - <<'PY'
import json
import os
from pathlib import Path
from datasets import load_dataset

dataset = os.environ["DATASET"]
split = os.environ["SPLIT"]
instance_id = os.environ["INSTANCE_ID"]
instance_json = Path(os.environ["INSTANCE_JSON"])

for row in load_dataset(dataset, split=split):
    if row["instance_id"] == instance_id:
        instance_json.write_text(json.dumps(dict(row)) + "\n", encoding="utf-8")
        break
else:
    raise SystemExit(f"Instance {instance_id} not found in {dataset}")
PY
  fi
  printf '%s\n' "$instance_json"
}

fetch_all_instances() {
  local all_json="${INSTANCE_DIR}/instance_all-${SPLIT}.jsonl"
  local ids_file="${INSTANCE_DIR}/instance_all-${SPLIT}.ids"
  if [ ! -f "$all_json" ] || [ ! -f "$ids_file" ]; then
    ensure_fetch_python
    echo "Fetching full ${DATASET} ${SPLIT} split..."
    DATASET="$DATASET" SPLIT="$SPLIT" ALL_JSON="$all_json" IDS_FILE="$ids_file" \
      "${FETCH_PYTHON[@]}" - <<'PY'
import json
import os
from pathlib import Path
from datasets import load_dataset

dataset = os.environ["DATASET"]
split = os.environ["SPLIT"]
all_json = Path(os.environ["ALL_JSON"])
ids_file = Path(os.environ["IDS_FILE"])

rows = [dict(row) for row in load_dataset(dataset, split=split)]
all_json.write_text(
    "".join(json.dumps(row) + "\n" for row in rows),
    encoding="utf-8",
)
ids_file.write_text(
    "".join(str(row["instance_id"]) + "\n" for row in rows),
    encoding="utf-8",
)
PY
  fi
  INSTANCE_JSON="$all_json"
  mapfile -t INSTANCE_IDS < "$ids_file"
}

run_container() {
  local instance_json="$1"
  local instance_id="$2"
  shift 2
  "${PYTHON[@]}" runs/run_bench.py swebench "$instance_id" \
    --instance-json "$instance_json" \
    --dataset-name "$DATASET" \
    --provider openai \
    --dotenv .env \
    --max-turns "$MAX_TURNS" \
    --run-id "$RUN_ID" \
    --run-root "$CONTAINER_RUN_ROOT" \
    --uv-binary "$SWEBENCH_UV_BIN" \
    --network-mode host \
    --force \
    "$@"
}

running_jobs() {
  jobs -pr | wc -l | tr -d ' '
}

collect_predictions() {
  mkdir -p "$PREDICTION_DIR"
  local pred_out="${PREDICTION_DIR}/${RUN_ID}_predictions.jsonl"
  "${PYTHON[@]}" evals/swebench/evaluate_predictions.py --collect-predictions \
    --run-root "$CONTAINER_RUN_ROOT" --run-id "$RUN_ID" \
    --dataset-name "$DATASET" --model-name "$MODEL_NAME" \
    --predictions "$pred_out"
}

if [ "$RUN_ALL" -eq 1 ]; then
  fetch_all_instances
  echo "=== SWE-bench Verified full split ==="
  echo "Run ID: $RUN_ID"
  echo "Instances: ${#INSTANCE_IDS[@]}"
  echo "Parallel: $PARALLEL"
  echo ""

  echo "Preparing wheelhouse and Linux uv once before launching batch..."
  swebench_ensure_linux_uv
  "${PYTHON[@]}" - <<'PY'
from evals.swebench.harness import DEFAULT_WHEELHOUSE, prepare_wheelhouse
prepare_wheelhouse(DEFAULT_WHEELHOUSE)
PY

  FAIL=0
  for instance_id in "${INSTANCE_IDS[@]}"; do
    while [ "$(running_jobs)" -ge "$PARALLEL" ]; do
      wait -n || FAIL=$((FAIL + 1))
    done
    log="${CONTAINER_RUN_ROOT}/${RUN_ID}/${instance_id}.log"
    mkdir -p "$(dirname "$log")"
    echo "Starting: ${instance_id}"
    run_container "$INSTANCE_JSON" "$instance_id" > "$log" 2>&1 &
  done

  while [ "$(running_jobs)" -gt 0 ]; do
    wait -n || FAIL=$((FAIL + 1))
  done

  collect_predictions
  echo "Outputs: ${CONTAINER_RUN_ROOT}/${RUN_ID}/"
  if [ "$FAIL" -gt 0 ]; then
    echo "Failed runs: $FAIL" >&2
    exit 1
  fi
else
  if [ "${#POSITIONAL[@]}" -eq 0 ]; then
    INSTANCE_IDS=("$DEFAULT_INSTANCE_ID")
  else
    INSTANCE_IDS=("${POSITIONAL[0]}")
  fi
  INSTANCE_JSON="$(fetch_one_instance "${INSTANCE_IDS[0]}")"
  echo "=== SWE-bench Verified: ${INSTANCE_IDS[0]} ==="
  echo "Run ID: $RUN_ID"
  swebench_ensure_linux_uv
  run_container "$INSTANCE_JSON" "${INSTANCE_IDS[0]}" --prepare-wheelhouse
  collect_predictions
  echo "Outputs: ${CONTAINER_RUN_ROOT}/${RUN_ID}/"
fi
