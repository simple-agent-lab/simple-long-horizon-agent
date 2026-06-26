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
# Keep the control-plane venv OUT of /tmp: agents routinely create scratch dirs
# under /tmp and clean up with a broad `rm -rf ... /tmp`, which would delete this
# venv (incl. certifi's CA bundle) and crash the next in-container model call.
AGENT_VENV = "/opt/agent-venv"

def _python_setup(wheelhouse_mount: str | None) -> str:
    """Render the Python-discovery preamble, offline-3.11 aware.

    When a wheelhouse is mounted, prefer a pre-provisioned uv-managed CPython
    3.11 under ``<wheelhouse>/uv-python`` (see ``harness.prepare_wheelhouse``):
    pointing ``UV_PYTHON_INSTALL_DIR`` there lets ``uv venv --python 3.11``
    resolve OFFLINE, so each container does not download a ~29MB standalone
    Python over the (often slow or locked-down) container network — a download
    that, when it failed, used to fall back silently to the image's <3.10 system
    Python and then break 60 lines later in pip resolution. ``UV_PYTHON_DOWNLOADS
    =never`` makes that fallback fail fast instead of retrying for an hour.
    """

    offline_python = 'OFFLINE_PYTHON=""'
    if wheelhouse_mount:
        uv_py_dir = f"{wheelhouse_mount.rstrip('/')}/uv-python"
        offline_python = f"""
for _candidate_python in "{uv_py_dir}"/cpython-3.11.*-linux-x86_64-gnu/bin/python3.11; do
  if [ -x "$_candidate_python" ]; then OFFLINE_PYTHON="$_candidate_python"; break; fi
done
if [ -n "$UV_BIN" ] && [ -d "{uv_py_dir}" ]; then
  export UV_PYTHON_INSTALL_DIR="{uv_py_dir}"
  export UV_PYTHON_DOWNLOADS=never
fi"""
    return f"""\
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
mkdir -p "$(dirname {AGENT_VENV})" 2>/dev/null || true{offline_python}
if [ -n "$UV_BIN" ]; then
  "$UV_BIN" venv --python 3.11 {AGENT_VENV} || "$UV_BIN" venv --python python3 {AGENT_VENV}
  AGENT_PYTHON={AGENT_VENV}/bin/python
elif [ "$_IS_MUSL" != 1 ] && [ -x "$OFFLINE_PYTHON" ]; then
  "$OFFLINE_PYTHON" -m venv {AGENT_VENV}
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
# The agent wheels target CPython 3.11 and simple-agent-lab requires >=3.10. If
# 3.11 provisioning failed and we fell back to an older system python, fail
# loudly HERE rather than 60 lines later with an opaque pip resolution error.
if ! "$AGENT_PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
  echo "ERROR: agent venv Python ($("$AGENT_PYTHON" -V 2>&1)) is < 3.10 -- CPython 3.11 provisioning failed (offline <wheelhouse>/uv-python missing and in-container download blocked)." >&2
  exit 1
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


def bootstrap_script(
    *,
    runner_argv: tuple[str, ...],
    install: bool = True,
    wheelhouse_mount: str | None = None,
    package_extras: tuple[str, ...] = (),
) -> str:
    """Return the shell script run as the container's main process.

    `runner_argv` is the suite's in-container runner invocation as an argv
    tuple (path + flags); it is exec'd with the prepared Python. Everything
    before it — Python discovery and the wheel install — is suite-agnostic.
    """

    parts = [_python_setup(wheelhouse_mount)]
    if install:
        parts.append(_install_line(wheelhouse_mount, package_extras))
    quoted = " ".join(shlex.quote(part) for part in runner_argv)
    parts.append(f'"$AGENT_PYTHON" {quoted}')
    return "\n".join(parts)
