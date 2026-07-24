# Shared: pick `uv run python` when uv is available, else fall back to python3.
# Source this file. Sets the PYTHON array; call it as "${PYTHON[@]}" script.py.

if command -v uv >/dev/null 2>&1; then
  PYTHON=(uv run python)
else
  PYTHON=(python3)
fi
