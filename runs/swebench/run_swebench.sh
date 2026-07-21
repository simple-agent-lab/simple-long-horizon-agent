#!/usr/bin/env bash
# Run the agent on a SWE-bench variant through the Suite framework (ADR generic-containerized-eval-framework).
#
# Usage:
#   bash runs/swebench/run_swebench.sh [--variant verified|multilingual|pro] [INSTANCE_ID]
#   bash runs/swebench/run_swebench.sh                                   # verified, default instance
#   bash runs/swebench/run_swebench.sh django__django-16379             # verified, one instance
#   bash runs/swebench/run_swebench.sh --ids-file ids.txt --parallel 4  # selected instances
#   bash runs/swebench/run_swebench.sh --variant pro --all --parallel 4 # pro, full split, 4 at a time
#
# Requires Docker, `uv sync --extra swebench`, and a .env with provider credentials.
# Optional: set OPENAI_AUTH_TOKEN2 in .env to alternate keys across batch runs.
# Downloading uncached dataset records uses `datasets`.

set -euo pipefail
cd "$(dirname "$0")/../.."
source runs/lib/_python.sh
source runs/lib/_swebench_uv.sh

VARIANT="verified"
RUN_ALL=0
IDS_FILE=""
PARALLEL=1
# Image pull policy. Empty = the run_bench default ('never'): opt-in, so a run
# never silently downloads multi-GB images. `--pull` opts in (a full split is
# hundreds of GB, so this is deliberately not the default).
PULL=""
POSITIONAL=()
FETCH_PYTHON=()
SECONDARY_OPENAI_AUTH_TOKEN="${OPENAI_AUTH_TOKEN2:-}"

usage() {
  sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
}

shell_quote() {
  printf "%q" "$1"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --variant)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --variant requires a value (verified|multilingual|pro)." >&2
        exit 2
      fi
      VARIANT="$2"
      shift 2
      ;;
    --all)
      RUN_ALL=1
      shift
      ;;
    --ids-file)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --ids-file requires a path." >&2
        exit 2
      fi
      IDS_FILE="$2"
      shift 2
      ;;
    --parallel)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --parallel requires a positive integer." >&2
        exit 2
      fi
      PARALLEL="$2"
      shift 2
      ;;
    --pull)
      # Opt in to downloading images. `--pull` alone = missing; an explicit
      # `--pull always|never|missing` is honored.
      if [ "$#" -ge 2 ] && [[ "$2" != -* ]]; then
        PULL="$2"
        shift 2
      else
        PULL="missing"
        shift
      fi
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

# Per-variant config — the only thing that differs between the three splits.
case "$VARIANT" in
  verified)
    DATASET="princeton-nlp/SWE-bench_Verified"
    DEFAULT_INSTANCE_ID="sympy__sympy-23824"
    OUT_ROOT="evals/out/swebench"
    WHEELHOUSE=""
    WHEELHOUSE_CONST="DEFAULT_WHEELHOUSE"
    MODEL_NAME="simple-agent-lab-verified"
    MAX_TURNS=150
    TITLE="SWE-bench Verified"
    ;;
  multilingual)
    DATASET="SWE-bench/SWE-bench_Multilingual"
    DEFAULT_INSTANCE_ID="${SWE_BENCH_MULTILINGUAL_DEFAULT_INSTANCE_ID:-}"
    OUT_ROOT="evals/out/swebench_multilingual"
    WHEELHOUSE="evals/out/swebench_multilingual/wheelhouse/cp311-manylinux"
    WHEELHOUSE_CONST="DEFAULT_MULTILINGUAL_WHEELHOUSE"
    MODEL_NAME="simple-agent-lab-multilingual"
    MAX_TURNS=150
    TITLE="SWE-bench Multilingual"
    ;;
  pro)
    DATASET="ScaleAI/SWE-bench_Pro"
    DEFAULT_INSTANCE_ID="instance_navidrome__navidrome-8e640bb8580affb7e0ea6225c0bbe240186b6b08"
    OUT_ROOT="evals/out/swebench_pro"
    WHEELHOUSE="evals/out/swebench_pro/wheelhouse/cp311-manylinux"
    WHEELHOUSE_CONST="DEFAULT_PRO_WHEELHOUSE"
    MODEL_NAME="simple-agent-lab-pro"
    MAX_TURNS=250
    TITLE="SWE-bench Pro"
    ;;
  *)
    echo "ERROR: unknown --variant: ${VARIANT} (expected verified|multilingual|pro)." >&2
    exit 2
    ;;
esac

MAX_TURNS="${SWEBENCH_MAX_TURNS:-$MAX_TURNS}"

SPLIT="test"
INSTANCE_DIR="$OUT_ROOT"
CONTAINER_RUN_ROOT="$OUT_ROOT"
PREDICTION_DIR="$OUT_ROOT"
RUN_ID="${VARIANT}-$(date +%Y%m%d-%H%M%S)"

if ! [[ "$PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --parallel must be a positive integer; got $(shell_quote "$PARALLEL")." >&2
  exit 2
fi
if ! [[ "$MAX_TURNS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: SWEBENCH_MAX_TURNS must be a positive integer; got $(shell_quote "$MAX_TURNS")." >&2
  exit 2
fi
if [ "$RUN_ALL" -eq 1 ] && [ -n "$IDS_FILE" ]; then
  echo "ERROR: pass only one of --all or --ids-file." >&2
  exit 2
fi
if { [ "$RUN_ALL" -eq 1 ] || [ -n "$IDS_FILE" ]; } && [ "${#POSITIONAL[@]}" -gt 0 ]; then
  echo "ERROR: pass only one of --all, --ids-file, or one INSTANCE_ID." >&2
  exit 2
fi
if [ "${#POSITIONAL[@]}" -gt 1 ]; then
  echo "ERROR: pass at most one INSTANCE_ID, or use --all/--ids-file for a batch." >&2
  exit 2
fi
if [ -n "$IDS_FILE" ] && [ ! -f "$IDS_FILE" ]; then
  echo "ERROR: --ids-file does not exist: $(shell_quote "$IDS_FILE")." >&2
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

load_instance_ids() {
  local ids_path="$1"
  local instance_id=""
  INSTANCE_IDS=()
  while IFS= read -r instance_id || [ -n "$instance_id" ]; do
    if [ -n "$instance_id" ]; then
      INSTANCE_IDS+=("$instance_id")
    fi
  done < "$ids_path"
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
  load_instance_ids "$ids_file"
}

fetch_ids_instances() {
  local ids_path="$1"
  local selected_json="${INSTANCE_DIR}/instance_ids-${RUN_ID}.jsonl"
  local selected_ids="${INSTANCE_DIR}/instance_ids-${RUN_ID}.ids"
  ensure_fetch_python
  echo "Fetching selected instances from ${DATASET} ${SPLIT} split..."
  DATASET="$DATASET" SPLIT="$SPLIT" IDS_FILE="$ids_path" SELECTED_JSON="$selected_json" SELECTED_IDS="$selected_ids" \
    "${FETCH_PYTHON[@]}" - <<'PY'
import json
import os
from pathlib import Path

from datasets import load_dataset

dataset = os.environ["DATASET"]
split = os.environ["SPLIT"]
ids_file = Path(os.environ["IDS_FILE"])
selected_json = Path(os.environ["SELECTED_JSON"])
selected_ids = Path(os.environ["SELECTED_IDS"])

ids = []
seen = set()
for lineno, raw_line in enumerate(ids_file.read_text(encoding="utf-8").splitlines(), 1):
    instance_id = raw_line.split("#", 1)[0].strip()
    if not instance_id:
        continue
    if instance_id in seen:
        raise SystemExit(
            f"Duplicate instance id in {ids_file} on line {lineno}: {instance_id}"
        )
    seen.add(instance_id)
    ids.append(instance_id)

if not ids:
    raise SystemExit(f"--ids-file {ids_file} did not contain any instance ids")

rows_by_id = {
    str(row["instance_id"]): dict(row)
    for row in load_dataset(dataset, split=split)
}
missing = [instance_id for instance_id in ids if instance_id not in rows_by_id]
if missing:
    preview = ", ".join(missing[:10])
    suffix = "" if len(missing) <= 10 else f" ... (+{len(missing) - 10} more)"
    raise SystemExit(f"Instance id(s) not found in {dataset}: {preview}{suffix}")

selected_json.write_text(
    "".join(json.dumps(rows_by_id[instance_id]) + "\n" for instance_id in ids),
    encoding="utf-8",
)
selected_ids.write_text("".join(instance_id + "\n" for instance_id in ids), encoding="utf-8")
PY
  INSTANCE_JSON="$selected_json"
  load_instance_ids "$selected_ids"
}

ensure_default_instance_id() {
  # Variants without a hard-coded default (multilingual) resolve it to the
  # first instance of the split.
  if [ -n "$DEFAULT_INSTANCE_ID" ]; then
    return
  fi
  fetch_all_instances
  if [ "${#INSTANCE_IDS[@]}" -eq 0 ]; then
    echo "ERROR: ${DATASET} ${SPLIT} returned no instances." >&2
    exit 1
  fi
  DEFAULT_INSTANCE_ID="${INSTANCE_IDS[0]}"
}

load_secondary_openai_auth_token() {
  if [ -n "$SECONDARY_OPENAI_AUTH_TOKEN" ] || [ ! -f .env ]; then
    return
  fi
  SECONDARY_OPENAI_AUTH_TOKEN="$("${PYTHON[@]}" - <<'PY'
from simple_agent_lab.llm.env import load_dotenv

env = {}
load_dotenv(".env", environ=env)
print(env.get("OPENAI_AUTH_TOKEN2", ""))
PY
)"
}

run_container() {
  local instance_json="$1"
  local instance_id="$2"
  shift 2
  local args=(
    runs/run_bench.py swebench "$instance_id"
    --instance-json "$instance_json"
    --dataset-name "$DATASET"
    --provider openai
    --dotenv .env
    --max-turns "$MAX_TURNS"
    --run-id "$RUN_ID"
    --agent-flavor "${AGENT_FLAVOR:-bash}"
    --run-root "$CONTAINER_RUN_ROOT"
    --uv-binary "$SWEBENCH_UV_BIN"
    --network-mode host
    --force
  )
  # Verified uses the harness default wheelhouse; the others pin a per-variant one.
  if [ -n "$WHEELHOUSE" ]; then
    args+=(--wheelhouse "$WHEELHOUSE")
  fi
  if [ -n "$PULL" ]; then
    args+=(--pull "$PULL")
  fi
  "${PYTHON[@]}" "${args[@]}" "$@"
}

run_container_for_index() {
  local job_index="$1"
  shift
  if [ -n "$SECONDARY_OPENAI_AUTH_TOKEN" ] && [ $((job_index % 2)) -eq 1 ]; then
    OPENAI_AUTH_TOKEN="$SECONDARY_OPENAI_AUTH_TOKEN"
    export OPENAI_AUTH_TOKEN
  fi
  run_container "$@"
}

running_jobs() {
  jobs -pr | wc -l | tr -d ' '
}

collect_predictions() {
  local expected_ids="${CONTAINER_RUN_ROOT}/${RUN_ID}/expected_instance_ids.txt"
  mkdir -p "$(dirname "$expected_ids")"
  printf '%s\n' "${INSTANCE_IDS[@]}" > "$expected_ids"
  local pred_out="${PREDICTION_DIR}/${RUN_ID}_predictions.jsonl"
  "${PYTHON[@]}" evals/swebench/evaluate_predictions.py --collect-predictions \
    --run-root "$CONTAINER_RUN_ROOT" --run-id "$RUN_ID" \
    --dataset-name "$DATASET" --model-name "$MODEL_NAME" \
    --expected-ids-file "$expected_ids" \
    --predictions "$pred_out"
}

if [ "$RUN_ALL" -eq 1 ] || [ -n "$IDS_FILE" ]; then
  if [ "$RUN_ALL" -eq 1 ]; then
    fetch_all_instances
    BATCH_LABEL="full split"
  else
    fetch_ids_instances "$IDS_FILE"
    BATCH_LABEL="ids file: $IDS_FILE"
  fi
  load_secondary_openai_auth_token
  echo "=== ${TITLE} ${BATCH_LABEL} ==="
  echo "Run ID: $RUN_ID"
  echo "Instances: ${#INSTANCE_IDS[@]}"
  echo "Parallel: $PARALLEL"
  if [ -n "$SECONDARY_OPENAI_AUTH_TOKEN" ]; then
    echo "OpenAI auth tokens: 2 (round-robin per instance)"
  else
    echo "OpenAI auth tokens: 1"
  fi
  echo ""

  echo "Preparing wheelhouse and Linux uv once before launching batch..."
  swebench_ensure_linux_uv
  WHEELHOUSE_CONST="$WHEELHOUSE_CONST" "${PYTHON[@]}" - <<'PY'
import os

import evals.swebench.harness as harness

harness.prepare_wheelhouse(getattr(harness, os.environ["WHEELHOUSE_CONST"]))
PY

  FAIL=0
  # Jobs record their own exit codes because the throttle polls `jobs -pr`, which
  # can reap completed jobs before the final wait.
  STATUS_DIR="${CONTAINER_RUN_ROOT}/${RUN_ID}/.exit_codes"
  mkdir -p "$STATUS_DIR"
  job_index=0
  for instance_id in "${INSTANCE_IDS[@]}"; do
    # Throttle to PARALLEL. macOS ships bash 3.2, which has no `wait -n`, so poll
    # the running-job count and sleep until a slot frees instead.
    while [ "$(running_jobs)" -ge "$PARALLEL" ]; do
      sleep 0.3
    done
    log="${CONTAINER_RUN_ROOT}/${RUN_ID}/${instance_id}.log"
    mkdir -p "$(dirname "$log")"
    echo "Starting: ${instance_id}"
    status_file="${STATUS_DIR}/${instance_id}.rc"
    rm -f "$status_file"
    # `|| rc=$?` tests the command so `set -e` does not abort the subshell before
    # the exit code is recorded (a bare `cmd; echo $?` would lose a failure code).
    (
      rc=0
      run_container_for_index "$job_index" "$INSTANCE_JSON" "$instance_id" \
        --reuse-prepared-wheelhouse || rc=$?
      echo "$rc" > "$status_file"
    ) > "$log" 2>&1 &
    job_index=$((job_index + 1))
  done

  wait 2>/dev/null || true
  # A missing status file means the job ended before it could record one.
  for instance_id in "${INSTANCE_IDS[@]}"; do
    rc="$(cat "${STATUS_DIR}/${instance_id}.rc" 2>/dev/null || echo missing)"
    if [ "$rc" != "0" ]; then
      FAIL=$((FAIL + 1))
      echo "Failed run: ${instance_id} (exit ${rc})" >&2
    fi
  done

  collect_predictions
  echo "Outputs: ${CONTAINER_RUN_ROOT}/${RUN_ID}/"
  if [ "$FAIL" -gt 0 ]; then
    echo "Failed runs: $FAIL" >&2
    exit 1
  fi
else
  if [ "${#POSITIONAL[@]}" -eq 0 ]; then
    ensure_default_instance_id
    INSTANCE_IDS=("$DEFAULT_INSTANCE_ID")
    INSTANCE_JSON="${INSTANCE_JSON:-$(fetch_one_instance "${INSTANCE_IDS[0]}")}"
  else
    INSTANCE_IDS=("${POSITIONAL[0]}")
    INSTANCE_JSON="$(fetch_one_instance "${INSTANCE_IDS[0]}")"
  fi
  echo "=== ${TITLE}: ${INSTANCE_IDS[0]} ==="
  echo "Run ID: $RUN_ID"
  swebench_ensure_linux_uv
  run_container "$INSTANCE_JSON" "${INSTANCE_IDS[0]}" --prepare-wheelhouse
  collect_predictions
  echo "Outputs: ${CONTAINER_RUN_ROOT}/${RUN_ID}/"
fi
