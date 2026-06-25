"""Suite-agnostic container bootstrap: prepare Python, install the wheel, run.

Every containerized suite needs the same preamble before its in-container
runner can start: find a Python (prefer a copied ``uv`` binary, then ``uv`` on
PATH, then a venv from system ``python3``, with an Alpine/musl branch), install
the ``simple-agent-lab`` wheel (from a mounted wheelhouse when offline), then
exec the runner. Only the final runner invocation differs per suite, so that is
passed in as `runner_argv`.

Suites contribute only `runner_argv` (and, via `LaunchSpec.shell`, the argv
prefix this script is handed to); everything else here is suite-agnostic.
"""

from __future__ import annotations

import shlex

UV_CONTAINER_PATH = "/tmp/uv"
AGENT_VENV = "/tmp/agent-venv"

_PYTHON_SETUP = f"""\
set -eu
(set -o pipefail) 2>/dev/null && set -o pipefail || true
UV_BIN=""
if [ -f {UV_CONTAINER_PATH} ]; then
  chmod +x {UV_CONTAINER_PATH} 2>/dev/null || true
  if {UV_CONTAINER_PATH} --version >/dev/null 2>&1; then UV_BIN={UV_CONTAINER_PATH}; fi
fi
if [ -z "$UV_BIN" ] && command -v uv >/dev/null 2>&1; then
  PATH_UV="$(command -v uv)"
  if "$PATH_UV" --version >/dev/null 2>&1; then UV_BIN="$PATH_UV"; fi
fi
_IS_MUSL=0
if ldd /bin/sh 2>/dev/null | grep -q musl; then _IS_MUSL=1; fi
if [ -n "$UV_BIN" ]; then
  "$UV_BIN" venv --managed-python --python 3.11 {AGENT_VENV} || {{
    echo "ERROR: uv could not provision Python 3.11 for the agent runtime." >&2
    echo "The mounted wheelhouse targets CPython 3.11; refusing to fall back to the SWE-bench repo Python." >&2
    exit 1
  }}
  AGENT_PYTHON={AGENT_VENV}/bin/python
elif [ "$_IS_MUSL" = 1 ]; then
  command -v python3 >/dev/null 2>&1 || {{ echo "ERROR: Alpine has no Python" >&2; exit 1; }}
  python3 -m venv {AGENT_VENV}
  AGENT_PYTHON={AGENT_VENV}/bin/python3
else
  if command -v python3 >/dev/null 2>&1; then
    if python3 -m venv {AGENT_VENV} >/dev/null 2>&1; then AGENT_PYTHON={AGENT_VENV}/bin/python3
    else AGENT_PYTHON=python3; fi
  else echo "ERROR: container has no uv and no python3" >&2; exit 1; fi
fi
"$AGENT_PYTHON" --version"""


def _package_name(package_extras: tuple[str, ...]) -> str:
    extra_text = f"[{','.join(package_extras)}]" if package_extras else ""
    return shlex.quote(f"simple-agent-lab{extra_text}")


def _install_line(wheelhouse_mount: str | None, package_extras: tuple[str, ...]) -> str:
    pkg = _package_name(package_extras)
    if wheelhouse_mount:
        find_links = shlex.quote(wheelhouse_mount)
        return (
            f'if [ -n "$UV_BIN" ]; then "$UV_BIN" pip install --python '
            f'"$AGENT_PYTHON" --no-index --find-links {find_links} {pkg}; '
            f'else "$AGENT_PYTHON" -m pip install --no-index '
            f"--find-links {find_links} {pkg}; fi"
        )
    return (
        f'if [ -n "$UV_BIN" ]; then "$UV_BIN" pip install --python '
        f'"$AGENT_PYTHON" {pkg}; else "$AGENT_PYTHON" -m pip install {pkg}; fi'
    )


def _pythonpath_line(extra_pythonpath: tuple[str, ...]) -> str:
    if not extra_pythonpath:
        return ""
    joined = ":".join(extra_pythonpath)
    return f'export PYTHONPATH={shlex.quote(joined)}"${{PYTHONPATH:+:$PYTHONPATH}}"'


def bootstrap_script(
    *,
    runner_argv: tuple[str, ...],
    install: bool = True,
    wheelhouse_mount: str | None = None,
    package_extras: tuple[str, ...] = (),
    extra_pythonpath: tuple[str, ...] = (),
) -> str:
    """Return the shell script run as the container's main process.

    `runner_argv` is the suite's in-container runner invocation as an argv
    tuple (path + flags); it is exec'd with the prepared Python. Everything
    before it — Python discovery and the wheel install — is suite-agnostic.
    """

    parts = [_PYTHON_SETUP]
    if install:
        parts.append(_install_line(wheelhouse_mount, package_extras))
    if line := _pythonpath_line(extra_pythonpath):
        parts.append(line)
    quoted = " ".join(shlex.quote(part) for part in runner_argv)
    parts.append(f'"$AGENT_PYTHON" {quoted}')
    return "\n".join(parts)
