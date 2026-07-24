# Shared Python command for this uv-managed repository.
# Source this file. Sets the PYTHON array; call modules with "${PYTHON[@]}" -m.

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required; install it from https://docs.astral.sh/uv/" >&2
  exit 1
fi

PYTHON=(uv run python)
