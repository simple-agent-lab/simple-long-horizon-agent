#!/usr/bin/env bash
# Compare agent flavors (bash | bash_task | bash_task_read | bash_skills | loop | pdr) on
# the SAME SWE-bench instances, with a REAL model. Pick the benchmark with
# --bench pro|verified. Each flavor runs over the chosen instance slice as its
# own run-id, so their predictions and trajectories are directly comparable.
# Resolved-rate grading is a separate official-harness step (printed at the
# end); this driver produces the real per-flavor trajectories + predictions.
#
# Usage:
#   # Pro, 20 instances, the two flavors we benchmark, 4-way Docker concurrency
#   bash runs/run_swebench_arms.sh --bench pro \
#     --arms "bash_task_read loop" --count 20 --parallel 4
#
#   # Verified, specific instances
#   bash runs/run_swebench_arms.sh --bench verified --arms "loop" sympy__sympy-23824
#
#   # tune per-agent turn budget
#   bash runs/run_swebench_arms.sh --count 8 --max-turns 50 --parallel 2
#
# Requires Docker, `uv sync --extra swebench`, and a .env with provider creds.
# One selector now: --agent-flavor. "baseline" is an alias for the strong
# single-agent flavor (bash_task_read); loop/pdr are the workflow arms. Raw
# flavor names (bash | bash_task | bash_task_read | bash_skills | loop | pdr) also work.
#
# If the Python docker SDK can't find the daemon (e.g. Colima, not Docker
# Desktop), export DOCKER_HOST first, e.g.:
#   export DOCKER_HOST="unix://$HOME/.colima/default/docker.sock"
# (confirm via `docker context inspect | grep Host`).

set -euo pipefail
cd "$(dirname "$0")/.."
source runs/_python.sh
source runs/_swebench_uv.sh

# One benchmark entrypoint, two presets. `--bench pro` (default) and
# `--bench verified` set the dataset / instance-dir / wheelhouse / model-name so
# the same flavor-sweep machinery drives both SWE-bench Pro and Verified.
BENCH="pro"
SPLIT="test"
MAX_TURNS=200
ARMS="baseline loop pdr"
COUNT=0
PARALLEL=1
TS="$(date +%Y%m%d-%H%M%S)"
POSITIONAL=()
FETCH_PYTHON=()

usage() { sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    --bench) BENCH="$2"; shift 2 ;;
    --ts) TS="$2"; shift 2 ;;
    --arms) ARMS="$2"; shift 2 ;;
    --count) COUNT="$2"; shift 2 ;;
    --max-turns) MAX_TURNS="$2"; shift 2 ;;
    --parallel) PARALLEL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

case "$BENCH" in
  pro)
    DATASET="ScaleAI/SWE-bench_Pro"
    INSTANCE_DIR="evals/out/swebench_pro"
    RUN_ROOT="evals/out/swebench_pro"
    WHEELHOUSE="evals/out/swebench_pro/wheelhouse/cp311-manylinux"
    MODEL_NAME="simple-agent-lab-pro"
    DEFAULT_INSTANCE_ID="instance_navidrome__navidrome-8e640bb8580affb7e0ea6225c0bbe240186b6b08"
    ;;
  verified)
    DATASET="princeton-nlp/SWE-bench_Verified"
    INSTANCE_DIR="evals/out/swebench"
    RUN_ROOT="evals/out/swebench"
    WHEELHOUSE="evals/out/swebench/wheelhouse/cp311-manylinux"
    MODEL_NAME="simple-agent-lab-verified"
    DEFAULT_INSTANCE_ID="sympy__sympy-23824"
    ;;
  *)
    echo "ERROR: --bench must be 'pro' or 'verified'; got ${BENCH@Q}." >&2; exit 2 ;;
esac

if ! [[ "$PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --parallel must be a positive integer; got ${PARALLEL@Q}." >&2; exit 2
fi

mkdir -p "$INSTANCE_DIR"

ensure_fetch_python() {
  if [ "${#FETCH_PYTHON[@]}" -gt 0 ]; then return; fi
  if command -v uv >/dev/null 2>&1; then FETCH_PYTHON=(uv run --extra swebench python); return; fi
  if "${PYTHON[@]}" - <<'PY' >/dev/null 2>&1
import datasets
PY
  then FETCH_PYTHON=("${PYTHON[@]}"); return; fi
  echo "ERROR: fetching uncached SWE-bench records requires the 'datasets' package." >&2
  echo "Install it with: uv sync --extra swebench" >&2
  exit 1
}

fetch_one_instance() {
  local instance_id="$1"
  local instance_json="${INSTANCE_DIR}/instance_${instance_id}.jsonl"
  if [ ! -f "$instance_json" ]; then
    ensure_fetch_python
    echo "Fetching instance ${instance_id} from ${DATASET}..." >&2
    DATASET="$DATASET" SPLIT="$SPLIT" INSTANCE_ID="$instance_id" INSTANCE_JSON="$instance_json" \
      "${FETCH_PYTHON[@]}" - <<'PY'
import json, os
from pathlib import Path
from datasets import load_dataset
dataset, split = os.environ["DATASET"], os.environ["SPLIT"]
instance_id = os.environ["INSTANCE_ID"]
out = Path(os.environ["INSTANCE_JSON"])
for row in load_dataset(dataset, split=split):
    if row["instance_id"] == instance_id:
        out.write_text(json.dumps(dict(row)) + "\n", encoding="utf-8"); break
else:
    raise SystemExit(f"Instance {instance_id} not found in {dataset}")
PY
  fi
  printf '%s\n' "$instance_json"
}

# Resolve the instance slice: explicit ids, or the first COUNT ids of the split.
INSTANCE_IDS=()
if [ "${#POSITIONAL[@]}" -gt 0 ]; then
  INSTANCE_IDS=("${POSITIONAL[@]}")
elif [ "$COUNT" -gt 0 ]; then
  ensure_fetch_python
  echo "Selecting first ${COUNT} instances from ${DATASET} ${SPLIT}..."
  while IFS= read -r line; do
    [ -n "$line" ] && INSTANCE_IDS+=("$line")
  done < <(
    DATASET="$DATASET" SPLIT="$SPLIT" COUNT="$COUNT" "${FETCH_PYTHON[@]}" - <<'PY'
import os
from datasets import load_dataset
dataset, split, count = os.environ["DATASET"], os.environ["SPLIT"], int(os.environ["COUNT"])
for i, row in enumerate(load_dataset(dataset, split=split)):
    if i >= count: break
    print(row["instance_id"])
PY
  )
else
  INSTANCE_IDS=("$DEFAULT_INSTANCE_ID")
fi

# Cache each instance record once up front (deterministic path per id, so the
# run loop re-derives it without an associative array — bash 3.2 friendly).
for instance_id in "${INSTANCE_IDS[@]}"; do
  fetch_one_instance "$instance_id" >/dev/null
done

instance_json_path() { printf '%s' "${INSTANCE_DIR}/instance_${1}.jsonl"; }

# Map an arm name to run_swebench_suite.py flags via the single --agent-flavor
# selector. "baseline" is an alias for the strong single-agent flavor
# (bash_task_read = bash + read + task explorer); loop/pdr are the workflow
# arms. A raw flavor name is passed through unchanged.
arm_flags() {
  case "$1" in
    baseline) printf '%s' "--agent-flavor bash_task_read" ;;
    bash|bash_task|bash_task_read|bash_skills|loop|pdr) printf '%s' "--agent-flavor $1" ;;
    *) echo "ERROR: unknown arm: $1 (expected baseline|bash|bash_task|bash_task_read|bash_skills|loop|pdr)" >&2; exit 2 ;;
  esac
}

echo "=== SWE-bench Pro arm comparison ==="
echo "Arms:       ${ARMS}"
echo "Instances:  ${#INSTANCE_IDS[@]} (${INSTANCE_IDS[*]:0:3}${INSTANCE_IDS[3]:+ ...})"
echo "Max turns:  ${MAX_TURNS}"
echo "Parallel:   ${PARALLEL}"
echo "Timestamp:  ${TS}"
echo ""

echo "Preparing wheelhouse and Linux uv once before launching..."
swebench_ensure_linux_uv
WHEELHOUSE="$WHEELHOUSE" "${PYTHON[@]}" - <<'PY'
import os
from pathlib import Path
from evals.swebench.harness import prepare_wheelhouse
prepare_wheelhouse(Path(os.environ["WHEELHOUSE"]))
PY

# Count this shell's still-running background jobs (the in-flight container runs).
running_jobs() { jobs -pr | wc -l | tr -d ' '; }

RUN_IDS=()
FAIL=0
for arm in $ARMS; do
  run_id="arms-${TS}-${arm}"
  RUN_IDS+=("$run_id")
  read -r -a flags <<< "$(arm_flags "$arm")"
  echo ">>> Arm '${arm}'  run-id=${run_id}"
  for instance_id in "${INSTANCE_IDS[@]}"; do
    # Skip-if-done: a finished instance already wrote out/result.json under this
    # run-id, so a resumed run (same --ts) doesn't redo completed work.
    if [ -f "${RUN_ROOT}/${run_id}/${instance_id}/out/result.json" ]; then
      echo "    skip (done): ${arm}: ${instance_id}"
      continue
    fi
    # Rolling window (bash 3.2-safe: poll `jobs`, no `wait -n`): keep at most
    # PARALLEL container runs in flight; launch a new one the moment a slot frees
    # — so a slow instance no longer stalls the others (the old batch-wait did).
    while [ "$(running_jobs)" -ge "$PARALLEL" ]; do sleep 2; done
    log="${RUN_ROOT}/${run_id}/${instance_id}.log"
    mkdir -p "$(dirname "$log")"
    echo "    starting ${arm}: ${instance_id}"
    "${PYTHON[@]}" runs/run_swebench_suite.py "$instance_id" \
      --instance-json "$(instance_json_path "$instance_id")" \
      --dataset-name "$DATASET" \
      --provider openai \
      --dotenv .env \
      --max-turns "$MAX_TURNS" \
      --run-id "$run_id" \
      --run-root "$RUN_ROOT" \
      --wheelhouse "$WHEELHOUSE" \
      --uv-binary "$SWEBENCH_UV_BIN" \
      --network-mode host \
      --force \
      ${flags[@]+"${flags[@]}"} \
      --prepare-wheelhouse > "$log" 2>&1 &
  done
  wait || FAIL=$((FAIL + 1))

  pred_out="${RUN_ROOT}/${run_id}_predictions.jsonl"
  "${PYTHON[@]}" evals/swebench/evaluate_predictions.py --collect-predictions \
    --run-root "$RUN_ROOT" --run-id "$run_id" \
    --dataset-name "$DATASET" --model-name "${MODEL_NAME}-${arm}" \
    --predictions "$pred_out"
  echo "    predictions: ${pred_out}"
  echo ""
done

echo "=== per-arm patch + cost summary ==="
"${PYTHON[@]}" runs/summarize_swebench_arms.py --run-root "$RUN_ROOT" "${RUN_IDS[@]}"

cat <<EOF

Next: grade resolved-rate with the official SWE-bench Pro harness, per arm, e.g.
  uv run --extra swebench python evals/swebench/evaluate_predictions.py --pro --run-official \\
    --instances ${INSTANCE_DIR}/instance_<id>.jsonl \\
    --predictions ${RUN_ROOT}/arms-${TS}-pdr_predictions.jsonl
(see evals/swebench/README.md for the one-time scaleapi/SWE-bench_Pro-os setup),
then re-run runs/summarize_swebench_arms.py to fold resolved counts in.
EOF

if [ "$FAIL" -gt 0 ]; then echo "Failed runs: $FAIL" >&2; exit 1; fi
