#!/usr/bin/env bash
# Run Simple Agent Lab on one SWE-Marathon task through Harbor.
#
# Usage:
#   bash runs/swe-marathon/run_swe_marathon.sh [TASK_NAME] [extra harbor args...]
#   SAL_MAX_TURNS=1 bash runs/swe-marathon/run_swe_marathon.sh rust-java-lsp --no-delete
#
# Clean-machine inputs:
#   SWE_MARATHON_TASKS_DIR   Existing SWE-Marathon checkout or tasks/ dir.
#   SWE_MARATHON_REPO_URL    Git URL to clone when TASKS_DIR is unset.
#   SWE_MARATHON_REF         Optional branch/tag/commit to checkout.
#   HARBOR_SOURCE            uvx install source when `harbor` is not on PATH.
#   SIMPLE_AGENT_LAB_SOURCE  Local source/wheel/git URL installed in container.
#   SIMPLE_AGENT_LAB_WHEELHOUSE  Optional local wheelhouse mounted for offline pip.

set -euo pipefail

cd "$(dirname "$0")/../.."

DEFAULT_TASK="find-network-alignments"
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

TASK_NAME="${1:-${SWE_MARATHON_TASK:-$DEFAULT_TASK}}"
if [ "$#" -gt 0 ]; then
  shift
fi

SWE_MARATHON_REPO_URL="${SWE_MARATHON_REPO_URL:-https://github.com/abundant-ai/swe-marathon}"
SWE_MARATHON_REF="${SWE_MARATHON_REF:-}"
HARBOR_SOURCE="${HARBOR_SOURCE:-git+https://github.com/RishiDesai/harbor.git}"
DEPS_DIR="${SWE_MARATHON_DEPS_DIR:-evals/out/swe-marathon/deps}"
JOBS_DIR="${SWE_MARATHON_JOBS_DIR:-evals/out/swe-marathon/jobs}"
SAL_CONTAINER_SOURCE="/opt/simple-agent-lab/source"
SAL_CONTAINER_WHEELHOUSE="/opt/simple-agent-lab/wheelhouse"
SAL_SOURCE="${SIMPLE_AGENT_LAB_SOURCE:-$PWD}"
SAL_WHEELHOUSE="${SIMPLE_AGENT_LAB_WHEELHOUSE:-}"
SAL_MAX_TURNS="${SAL_MAX_TURNS:-2000}"
SAL_AGENT_FLAVOR="${SAL_AGENT_FLAVOR:-bash_task}"

usage() {
  sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

load_dotenv() {
  if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
  fi
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

resolve_harbor_cmd() {
  if [ -n "${HARBOR_CMD:-}" ]; then
    # shellcheck disable=SC2206
    HARBOR=($HARBOR_CMD)
  elif command -v harbor >/dev/null 2>&1; then
    HARBOR=(harbor)
  else
    require_command uv
    HARBOR=(uvx --from "$HARBOR_SOURCE" harbor)
  fi
}

ensure_tasks_checkout() {
  if [ -n "${SWE_MARATHON_TASKS_DIR:-}" ]; then
    TASKS_ROOT="$SWE_MARATHON_TASKS_DIR"
    return
  fi

  require_command git
  mkdir -p "$DEPS_DIR"
  TASKS_ROOT="$DEPS_DIR/swe-marathon"
  if [ ! -d "$TASKS_ROOT/.git" ]; then
    git clone "$SWE_MARATHON_REPO_URL" "$TASKS_ROOT"
  fi
  if [ -n "$SWE_MARATHON_REF" ]; then
    git -C "$TASKS_ROOT" fetch --all --tags
    git -C "$TASKS_ROOT" checkout "$SWE_MARATHON_REF"
  fi
}

resolve_task_path() {
  if [ -f "$TASKS_ROOT/task.toml" ]; then
    TASK_PATH="$TASKS_ROOT"
  elif [ -f "$TASKS_ROOT/$TASK_NAME/task.toml" ]; then
    TASK_PATH="$TASKS_ROOT/$TASK_NAME"
  elif [ -f "$TASKS_ROOT/tasks/$TASK_NAME/task.toml" ]; then
    TASK_PATH="$TASKS_ROOT/tasks/$TASK_NAME"
  else
    die "could not find task '$TASK_NAME' under $TASKS_ROOT"
  fi
}

mounts_json_for_pairs() {
  python3 - "$@" <<'PY'
import json
import sys

args = sys.argv[1:]
if len(args) % 2:
    raise SystemExit("mount arguments must be source/target pairs")
mounts = []
for index in range(0, len(args), 2):
    source, target = args[index:index + 2]
    mounts.append({
        "type": "bind",
        "source": source,
        "target": target,
        "read_only": True,
        "bind": {"create_host_path": False},
    })
print(json.dumps(mounts))
PY
}

add_env_if_present() {
  local key="$1"
  if [ -n "${!key:-}" ]; then
    AGENT_ENV+=(--ae "$key=${!key}")
  fi
}

load_dotenv
require_command docker
docker info >/dev/null 2>&1 || die "Docker is not running or not reachable."
require_command python3
resolve_harbor_cmd
ensure_tasks_checkout
resolve_task_path

HARBOR_MODEL="${HARBOR_MODEL:-}"
if [ -z "$HARBOR_MODEL" ]; then
  [ -n "${OPENAI_MODEL:-}" ] || die "set HARBOR_MODEL or OPENAI_MODEL."
  HARBOR_MODEL="openai/$OPENAI_MODEL"
fi
if [ -z "${OPENAI_AUTH_TOKEN:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
  die "set OPENAI_AUTH_TOKEN or OPENAI_API_KEY for the model gateway."
fi

mkdir -p "$JOBS_DIR"

AGENT_ENV=(
  --ae "SAL_MAX_TURNS=$SAL_MAX_TURNS"
  --ae "SAL_AGENT_FLAVOR=$SAL_AGENT_FLAVOR"
)
add_env_if_present OPENAI_AUTH_TOKEN
add_env_if_present OPENAI_API_KEY
add_env_if_present OPENAI_BASE_URL
add_env_if_present API_KIND
add_env_if_present REASONING_EFFORT
add_env_if_present OPENAI_REASONING_EFFORT
add_env_if_present OPENAI_SESSION_ID
add_env_if_present OPENAI_LOG_ID
add_env_if_present SAL_WORKDIR

MOUNT_ARGS=()
MOUNT_PAIRS=()
if [ -e "$SAL_SOURCE" ]; then
  MOUNT_PAIRS+=("$SAL_SOURCE" "$SAL_CONTAINER_SOURCE")
  AGENT_ENV+=(--ae "SIMPLE_AGENT_LAB_SOURCE=$SAL_CONTAINER_SOURCE")
else
  AGENT_ENV+=(--ae "SIMPLE_AGENT_LAB_SOURCE=$SAL_SOURCE")
fi
if [ -n "$SAL_WHEELHOUSE" ]; then
  [ -d "$SAL_WHEELHOUSE" ] || die "SIMPLE_AGENT_LAB_WHEELHOUSE is not a directory: $SAL_WHEELHOUSE"
  MOUNT_PAIRS+=("$SAL_WHEELHOUSE" "$SAL_CONTAINER_WHEELHOUSE")
  if [ -n "${SIMPLE_AGENT_LAB_PIP_ARGS:-}" ]; then
    AGENT_ENV+=(--ae "SIMPLE_AGENT_LAB_PIP_ARGS=--no-index --find-links $SAL_CONTAINER_WHEELHOUSE $SIMPLE_AGENT_LAB_PIP_ARGS")
  else
    AGENT_ENV+=(--ae "SIMPLE_AGENT_LAB_PIP_ARGS=--no-index --find-links $SAL_CONTAINER_WHEELHOUSE")
  fi
else
  add_env_if_present SIMPLE_AGENT_LAB_PIP_ARGS
fi
if [ "${#MOUNT_PAIRS[@]}" -gt 0 ]; then
  MOUNT_ARGS+=(--mounts-json "$(mounts_json_for_pairs "${MOUNT_PAIRS[@]}")")
fi

echo "Task: $TASK_PATH"
echo "Agent flavor: $SAL_AGENT_FLAVOR"
echo "Model: $HARBOR_MODEL"
echo "Jobs: $JOBS_DIR"

PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  "${HARBOR[@]}" run \
    -p "$TASK_PATH" \
    --agent-import-path adapters.harbor_agent:SimpleAgentLab \
    --model "$HARBOR_MODEL" \
    --jobs-dir "$JOBS_DIR" \
    --override-gpus 0 \
    "${AGENT_ENV[@]}" \
    "${MOUNT_ARGS[@]}" \
    "$@"
