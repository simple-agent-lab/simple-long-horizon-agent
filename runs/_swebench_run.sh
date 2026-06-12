# Shared driver for the SWE-bench-family launchers (Verified / Multilingual / Pro).
#
# The three datasets differ only in DATA — dataset id, default instance, run
# root, model name, turn budget, run-id prefix, banner label — so each launcher
# sets those `SWEBENCH_*` variables and calls `swebench_main "$@"`. Everything
# else (argument parsing, dataset fetch + cache, the per-instance container run,
# the parallel batch loop, prediction collection) lives here, once.
#
# This mirrors a unification that already exists one layer down: `SwebenchSuite`
# / `run_swebench_suite.py --dataset-name` resolve the launch shape (image,
# workdir, run root, wheelhouse) per dataset as data. This driver just passes the
# dataset name through and lets the Python entry pick the matching defaults, so
# the per-dataset run root and wheelhouse need not be repeated in bash.
#
# Required variables (set by the launcher before calling `swebench_main`):
#   SWEBENCH_DATASET             HuggingFace dataset id
#   SWEBENCH_DEFAULT_INSTANCE_ID default single instance ("" = first of split)
#   SWEBENCH_RUN_ROOT            evals/out/<root> (cached records, wheelhouse, outputs)
#   SWEBENCH_MODEL_NAME          prediction model_name label
#   SWEBENCH_MAX_TURNS           per-instance turn budget
#   SWEBENCH_RUN_ID_PREFIX       run-id prefix, e.g. "verified"
#   SWEBENCH_LABEL               banner label, e.g. "SWE-bench Verified"
# Optional:
#   SWEBENCH_SPLIT               dataset split (default "test")
#   AGENT_FLAVOR                 bash | bash_task | bash_skills (default bash)
#
# Requires `runs/_python.sh` and `runs/_swebench_uv.sh` to be sourced first
# (for the PYTHON array and swebench_ensure_linux_uv / SWEBENCH_UV_BIN).

# Print the launcher's own usage header (lines 2-10 of the calling script).
swebench_usage() {
  sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
}

swebench_parse_args() {
  RUN_ALL=0
  PARALLEL=1
  POSITIONAL=()
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
        swebench_usage
        exit 0
        ;;
      --*)
        echo "ERROR: unknown option: $1" >&2
        swebench_usage >&2
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
}

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

fetch_one_instance() {
  local instance_id="$1"
  local instance_json="${SWEBENCH_RUN_ROOT}/instance_${instance_id}.jsonl"
  if [ ! -f "$instance_json" ]; then
    ensure_fetch_python
    echo "Fetching instance ${instance_id} from ${SWEBENCH_DATASET}..." >&2
    DATASET="$SWEBENCH_DATASET" SPLIT="$SWEBENCH_SPLIT" INSTANCE_ID="$instance_id" INSTANCE_JSON="$instance_json" \
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
  local all_json="${SWEBENCH_RUN_ROOT}/instance_all-${SWEBENCH_SPLIT}.jsonl"
  local ids_file="${SWEBENCH_RUN_ROOT}/instance_all-${SWEBENCH_SPLIT}.ids"
  if [ ! -f "$all_json" ] || [ ! -f "$ids_file" ]; then
    ensure_fetch_python
    echo "Fetching full ${SWEBENCH_DATASET} ${SWEBENCH_SPLIT} split..."
    DATASET="$SWEBENCH_DATASET" SPLIT="$SWEBENCH_SPLIT" ALL_JSON="$all_json" IDS_FILE="$ids_file" \
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

# Resolve an empty default instance id to the split's first instance. A no-op
# when SWEBENCH_DEFAULT_INSTANCE_ID is already set (Verified / Pro); only
# Multilingual leaves it empty.
ensure_default_instance_id() {
  if [ -n "$SWEBENCH_DEFAULT_INSTANCE_ID" ]; then
    return
  fi
  fetch_all_instances
  if [ "${#INSTANCE_IDS[@]}" -eq 0 ]; then
    echo "ERROR: ${SWEBENCH_DATASET} ${SWEBENCH_SPLIT} returned no instances." >&2
    exit 1
  fi
  SWEBENCH_DEFAULT_INSTANCE_ID="${INSTANCE_IDS[0]}"
}

run_container() {
  local instance_json="$1"
  local instance_id="$2"
  shift 2
  # --wheelhouse / --run-root are omitted on purpose: run_swebench_suite.py
  # derives both from --dataset-name (DEFAULT_*_WHEELHOUSE == the same path),
  # so the dataset name is the single source of truth.
  "${PYTHON[@]}" runs/run_swebench_suite.py "$instance_id" \
    --instance-json "$instance_json" \
    --dataset-name "$SWEBENCH_DATASET" \
    --provider openai \
    --dotenv .env \
    --max-turns "$SWEBENCH_MAX_TURNS" \
    --run-id "$RUN_ID" \
    --agent-flavor "${AGENT_FLAVOR:-bash}" \
    --run-root "$SWEBENCH_RUN_ROOT" \
    --uv-binary "$SWEBENCH_UV_BIN" \
    --network-mode host \
    --force \
    "$@"
}

running_jobs() {
  jobs -pr | wc -l | tr -d ' '
}

collect_predictions() {
  mkdir -p "$SWEBENCH_RUN_ROOT"
  local pred_out="${SWEBENCH_RUN_ROOT}/${RUN_ID}_predictions.jsonl"
  "${PYTHON[@]}" evals/swebench/evaluate_predictions.py --collect-predictions \
    --run-root "$SWEBENCH_RUN_ROOT" --run-id "$RUN_ID" \
    --dataset-name "$SWEBENCH_DATASET" --model-name "$SWEBENCH_MODEL_NAME" \
    --predictions "$pred_out"
}

swebench_run_all() {
  fetch_all_instances
  echo "=== ${SWEBENCH_LABEL} full split ==="
  echo "Run ID: $RUN_ID"
  echo "Instances: ${#INSTANCE_IDS[@]}"
  echo "Parallel: $PARALLEL"
  echo ""

  echo "Preparing wheelhouse and Linux uv once before launching batch..."
  swebench_ensure_linux_uv
  WHEELHOUSE="${SWEBENCH_RUN_ROOT}/wheelhouse/cp311-manylinux" "${PYTHON[@]}" - <<'PY'
import os
from pathlib import Path
from evals.swebench.harness import prepare_wheelhouse

prepare_wheelhouse(Path(os.environ["WHEELHOUSE"]))
PY

  local FAIL=0
  local instance_id log
  for instance_id in "${INSTANCE_IDS[@]}"; do
    while [ "$(running_jobs)" -ge "$PARALLEL" ]; do
      wait -n || FAIL=$((FAIL + 1))
    done
    log="${SWEBENCH_RUN_ROOT}/${RUN_ID}/${instance_id}.log"
    mkdir -p "$(dirname "$log")"
    echo "Starting: ${instance_id}"
    run_container "$INSTANCE_JSON" "$instance_id" > "$log" 2>&1 &
  done

  while [ "$(running_jobs)" -gt 0 ]; do
    wait -n || FAIL=$((FAIL + 1))
  done

  collect_predictions
  echo "Outputs: ${SWEBENCH_RUN_ROOT}/${RUN_ID}/"
  if [ "$FAIL" -gt 0 ]; then
    echo "Failed runs: $FAIL" >&2
    exit 1
  fi
}

swebench_run_one() {
  if [ "${#POSITIONAL[@]}" -eq 0 ]; then
    ensure_default_instance_id
    INSTANCE_IDS=("$SWEBENCH_DEFAULT_INSTANCE_ID")
    # ensure_default_instance_id may have fetched the full split (and set
    # INSTANCE_JSON); reuse it rather than fetching the single record again.
    INSTANCE_JSON="${INSTANCE_JSON:-$(fetch_one_instance "${INSTANCE_IDS[0]}")}"
  else
    INSTANCE_IDS=("${POSITIONAL[0]}")
    INSTANCE_JSON="$(fetch_one_instance "${INSTANCE_IDS[0]}")"
  fi
  echo "=== ${SWEBENCH_LABEL}: ${INSTANCE_IDS[0]} ==="
  echo "Run ID: $RUN_ID"
  swebench_ensure_linux_uv
  run_container "$INSTANCE_JSON" "${INSTANCE_IDS[0]}" --prepare-wheelhouse
  collect_predictions
  echo "Outputs: ${SWEBENCH_RUN_ROOT}/${RUN_ID}/"
}

swebench_main() {
  : "${SWEBENCH_SPLIT:=test}"
  RUN_ID="${SWEBENCH_RUN_ID_PREFIX}-$(date +%Y%m%d-%H%M%S)"
  FETCH_PYTHON=()
  INSTANCE_JSON=""
  INSTANCE_IDS=()

  swebench_parse_args "$@"

  mkdir -p "$SWEBENCH_RUN_ROOT"

  if [ "$RUN_ALL" -eq 1 ]; then
    swebench_run_all
  else
    swebench_run_one
  fi
}
