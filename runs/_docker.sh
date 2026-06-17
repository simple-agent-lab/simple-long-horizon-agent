# Shared: resolve DOCKER_HOST and, when needed, auto-start a Docker daemon.
#
# Source this file, then call `docker_ensure_running` before SWE-bench --execute
# work. It first resolves DOCKER_HOST from the active context or a known socket;
# if the daemon is still unreachable it tries to start Colima (preferred) or
# Docker Desktop on macOS, or the docker service via systemctl on Linux, polling
# until the daemon answers.
#
# Knobs (env overrides):
#   SAL_DOCKER_AUTOSTART=0   Skip the auto-start attempt (resolve + check only).
#   DOCKER_START_TIMEOUT=120 Seconds to wait for the daemon to come up.
#   COLIMA_CPU=12 COLIMA_MEMORY=32 COLIMA_DISK=100 COLIMA_ARCH=aarch64  Colima VM sizing.
#
# Sizing only applies when Colima is started here. Colima cannot resize a live
# VM: if one is already running with less, stop it and re-run, or just run
#   colima stop && colima start --cpu 12 --memory 32 --disk 100 --arch aarch64 \
#     --vm-type vz --vz-rosetta

COLIMA_CPU="${COLIMA_CPU:-12}"
COLIMA_MEMORY="${COLIMA_MEMORY:-32}"
COLIMA_DISK="${COLIMA_DISK:-100}"
COLIMA_ARCH="${COLIMA_ARCH:-aarch64}"

docker_resolve_host() {
  if [ -n "${DOCKER_HOST:-}" ]; then
    return 0
  fi
  local active
  active="$(docker context inspect --format '{{.Endpoints.docker.Host}}' 2>/dev/null || true)"
  if [ -n "$active" ] && [ "$active" != "<no value>" ]; then
    export DOCKER_HOST="$active"
    return 0
  fi
  local sock
  for sock in "$HOME/.docker/run/docker.sock" "$HOME/.colima/default/docker.sock"; do
    if [ -S "$sock" ]; then
      export DOCKER_HOST="unix://$sock"
      return 0
    fi
  done
}

docker_daemon_ready() {
  command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1
}

docker_wait_ready() {
  local timeout="${1:-120}"
  local waited=0
  while [ "$waited" -lt "$timeout" ]; do
    if docker_daemon_ready; then
      return 0
    fi
    sleep 3
    waited=$((waited + 3))
  done
  docker_daemon_ready
}

# Resolve DOCKER_HOST and ensure the daemon is reachable, starting it if needed.
# Returns non-zero if the daemon could not be reached.
docker_ensure_running() {
  docker_resolve_host
  if docker_daemon_ready; then
    return 0
  fi

  if [ "${SAL_DOCKER_AUTOSTART:-1}" != "1" ]; then
    return 1
  fi

  if command -v colima >/dev/null 2>&1; then
    echo "==> Docker daemon unreachable; starting Colima (${COLIMA_CPU} CPU / ${COLIMA_MEMORY} GiB)..." >&2
    colima start \
      --cpu "$COLIMA_CPU" \
      --memory "$COLIMA_MEMORY" \
      --disk "$COLIMA_DISK" \
      --arch "$COLIMA_ARCH" \
      --vm-type vz --vz-rosetta >&2 || true
  elif [ "$(uname -s)" = "Darwin" ] && [ -d "/Applications/Docker.app" ]; then
    echo "==> Docker daemon unreachable; starting Docker Desktop..." >&2
    open -a Docker >&2 || true
  elif command -v systemctl >/dev/null 2>&1; then
    echo "==> Docker daemon unreachable; starting docker service..." >&2
    sudo systemctl start docker >&2 || true
  else
    return 1
  fi

  echo "==> Waiting for Docker daemon..." >&2
  if docker_wait_ready "${DOCKER_START_TIMEOUT:-120}"; then
    docker_resolve_host
    echo "==> Docker daemon is ready." >&2
    return 0
  fi

  return 1
}
