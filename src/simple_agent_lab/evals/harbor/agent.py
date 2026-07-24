"""Harbor installed-agent adapter for Simple Agent Lab."""

from __future__ import annotations

import importlib
import json
import shlex
import tarfile
from pathlib import Path
from typing import Any

from . import AGENT_IMPORT_PATH, DEFAULT_API_KIND, DEFAULT_MAX_TURNS
from .runner import DEFAULT_SUMMARY_PATH, DEFAULT_TRACE_PATH

BaseInstalledAgent: Any = object
BaseEnvironment: Any = object
AgentContext: Any = object

try:  # Harbor is optional for SAL's core package and test suite.
    _base_mod = importlib.import_module("harbor.agents.installed.base")
    _env_mod = importlib.import_module("harbor.environments.base")
    _context_mod = importlib.import_module("harbor.models.agent.context")
    BaseInstalledAgent = _base_mod.BaseInstalledAgent
    with_prompt_template = _base_mod.with_prompt_template
    BaseEnvironment = _env_mod.BaseEnvironment
    AgentContext = _context_mod.AgentContext
except ImportError as exc:  # pragma: no cover - exercised by construction only
    _HARBOR_IMPORT_ERROR: ImportError | None = exc

    def with_prompt_template(fn: Any) -> Any:  # type: ignore[no-redef]
        return fn

else:
    _HARBOR_IMPORT_ERROR = None


REMOTE_AGENT_DIR = "/logs/agent"
REMOTE_INSTRUCTION_PATH = f"{REMOTE_AGENT_DIR}/sal-instruction.txt"
REMOTE_OUTPUT_PATH = f"{REMOTE_AGENT_DIR}/simple-agent-lab.txt"
REMOTE_SOURCE_ARCHIVE_PATH = "/tmp/simple-agent-lab-src.tar.gz"
REMOTE_SOURCE_DIR = "/tmp/simple-agent-lab-src"
REMOTE_RUNTIME_REQUIREMENTS_PATH = "/tmp/simple-agent-lab-runtime-requirements.txt"
DEFAULT_VENV_PATH = "/opt/simple-agent-lab-venv"
DEFAULT_SAL_PACKAGE = "simple-agent-lab"
_SOURCE_ROOT_FILES = ("pyproject.toml", "README.md", "LICENSE")
_PATH_SETUP = (
    'export PATH="$HOME/.local/bin:$PATH"; '
    '[ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env" || true'
)
_SAL_IMPORT_CHECK = (
    "import simple_agent_lab; import simple_agent_lab.evals.harbor.runner"
)
_DEFAULT_INSTALL_TIMEOUT_SEC = 3000
_SETUP_ENV_PREFIX = "SAL_HARBOR_SETUP_"


def _quote_parts(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def find_sal_source_root(start: Path | None = None) -> Path | None:
    """Return the local SAL checkout root when this module is imported from one."""

    current = (start or Path(__file__)).resolve()
    for path in (current, *current.parents):
        if (path / "pyproject.toml").is_file() and (
            path / "src/simple_agent_lab"
        ).is_dir():
            return path
    return None


def _add_tree(tar: tarfile.TarFile, source: Path, arcname: str) -> None:
    for path in sorted(source.rglob("*")):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        tar.add(
            path,
            arcname=str(Path(arcname) / path.relative_to(source)),
            recursive=False,
        )


def build_sal_source_archive(source_root: str | Path, archive_path: str | Path) -> Path:
    """Create the minimal source archive Harbor uploads into the task container."""

    root = Path(source_root).resolve()
    if (
        not (root / "pyproject.toml").is_file()
        or not (root / "src/simple_agent_lab").is_dir()
    ):
        raise ValueError(f"{root} is not a Simple Agent Lab source checkout")

    archive = Path(archive_path)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        for name in _SOURCE_ROOT_FILES:
            path = root / name
            if path.exists():
                tar.add(path, arcname=name)
        _add_tree(tar, root / "src", "src")
    return archive


def build_sal_installed_check_command(*, venv_path: str = DEFAULT_VENV_PATH) -> str:
    """Build the command that checks whether the in-container SAL venv works."""

    return f"{shlex.quote(venv_path)}/bin/python -c {shlex.quote(_SAL_IMPORT_CHECK)}"


def build_sal_system_dependencies_command() -> str:
    """Build a Codex/ClaudeCode-style system dependency bootstrap command."""

    return """
set -euo pipefail

_sal_python_version_ok() {
  "$1" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
}

if command -v python3 >/dev/null 2>&1 && _sal_python_version_ok python3; then
  tmp_dir="$(mktemp -d)"
  if python3 -m venv "$tmp_dir/venv" >/dev/null 2>&1 && "$tmp_dir/venv/bin/python" -m pip --version >/dev/null 2>&1; then
    rm -rf "$tmp_dir"
    exit 0
  fi
  rm -rf "$tmp_dir"
fi

if command -v python >/dev/null 2>&1 && _sal_python_version_ok python; then
  tmp_dir="$(mktemp -d)"
  if python -m venv "$tmp_dir/venv" >/dev/null 2>&1 && "$tmp_dir/venv/bin/python" -m pip --version >/dev/null 2>&1; then
    rm -rf "$tmp_dir"
    exit 0
  fi
  rm -rf "$tmp_dir"
fi

if ldd --version 2>&1 | grep -qi musl || [ -f /etc/alpine-release ]; then
  apk add --no-cache python3 py3-pip py3-virtualenv curl ca-certificates bash
elif command -v apt-get >/dev/null 2>&1; then
  apt-get update && apt-get install -y --no-install-recommends python3 python3-venv curl ca-certificates
elif command -v yum >/dev/null 2>&1; then
  yum install -y python3 python3-pip curl ca-certificates
elif command -v dnf >/dev/null 2>&1; then
  dnf install -y python3 python3-pip curl ca-certificates
else
  echo "Warning: No known package manager found; trying existing Python, uv, or curl" >&2
fi
""".strip()


def build_sal_venv_command(
    *,
    venv_path: str = DEFAULT_VENV_PATH,
    python_version: str = "3.12",
) -> str:
    """Build the command that creates the SAL runtime venv."""

    quoted_venv = shlex.quote(venv_path)
    quoted_python_version = shlex.quote(python_version)
    return f"""
set -euo pipefail
{_PATH_SETUP}

_sal_python_version_ok() {{
  "$1" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
}}

if command -v python3 >/dev/null 2>&1 && _sal_python_version_ok python3; then
  rm -rf {quoted_venv}
  if python3 -m venv {quoted_venv}; then
    {quoted_venv}/bin/python -m pip --version
    exit 0
  fi
fi

if command -v python >/dev/null 2>&1 && _sal_python_version_ok python; then
  rm -rf {quoted_venv}
  if python -m venv {quoted_venv}; then
    {quoted_venv}/bin/python -m pip --version
    exit 0
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    {_PATH_SETUP}
  fi
fi

command -v uv >/dev/null 2>&1 || {{
  echo "Simple Agent Lab setup needs Python >=3.10 with venv, or uv plus curl for fallback" >&2
  exit 1
}}

uv python install {quoted_python_version}
uv venv {quoted_venv} --python {quoted_python_version} --clear --seed
{quoted_venv}/bin/python -m pip --version
""".strip()


def build_sal_package_install_command(
    *,
    venv_path: str = DEFAULT_VENV_PATH,
    install_target: str,
    local_source: bool = False,
) -> str:
    """Build the command that installs SAL into the prepared runtime venv."""

    quoted_venv = shlex.quote(venv_path)
    quoted_target = shlex.quote(install_target)
    quoted_requirements = shlex.quote(REMOTE_RUNTIME_REQUIREMENTS_PATH)
    python = f"{quoted_venv}/bin/python"
    if local_source:
        return f"""
set -euo pipefail
{_PATH_SETUP}
SAL_SOURCE_TARGET={quoted_target}
SAL_RUNTIME_REQUIREMENTS={quoted_requirements}
export SAL_SOURCE_TARGET SAL_RUNTIME_REQUIREMENTS
{python} - <<'PY'
import os
import sysconfig
from pathlib import Path

source = Path(os.environ["SAL_SOURCE_TARGET"])  # env-ok: shell-to-Python install handoff
pyproject = source / "pyproject.toml"
requirements = Path(os.environ["SAL_RUNTIME_REQUIREMENTS"])  # env-ok: shell-to-Python install handoff

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None

if tomllib is not None:
    deps = tomllib.loads(pyproject.read_text(encoding="utf-8")).get("project", {{}}).get("dependencies", [])
else:
    deps = []
    in_project = False
    in_dependencies = False
    for raw_line in pyproject.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            in_project = line == "[project]"
            in_dependencies = False
            continue
        if not in_project:
            continue
        if not in_dependencies and line.startswith("dependencies"):
            _key, _sep, value = line.partition("=")
            value = value.strip()
            if value == "[":
                in_dependencies = True
                continue
        if in_dependencies:
            if line.startswith("]"):
                break
            item = line.split("#", 1)[0].rstrip(",").strip().strip("'\\\"")
            if item:
                deps.append(item)

requirements.write_text("\\n".join(deps) + ("\\n" if deps else ""), encoding="utf-8")

site_packages = Path(sysconfig.get_paths()["purelib"])
site_packages.mkdir(parents=True, exist_ok=True)
(site_packages / "simple-agent-lab-source.pth").write_text(
    str(source / "src") + "\\n",
    encoding="utf-8",
)
PY
if [ -s {quoted_requirements} ]; then
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python {python} -r {quoted_requirements}
  else
    {python} -m pip install --disable-pip-version-check --no-input -r {quoted_requirements}
  fi
fi
{python} -c {shlex.quote(_SAL_IMPORT_CHECK)}
""".strip()
    return f"""
set -euo pipefail
{_PATH_SETUP}
if command -v uv >/dev/null 2>&1; then
  uv pip install --python {python} {quoted_target}
else
  {python} -m pip install --disable-pip-version-check --no-input {quoted_target}
fi
{python} -c {shlex.quote(_SAL_IMPORT_CHECK)}
""".strip()


def build_sal_runner_command(
    *,
    instruction_path: str,
    cwd: str,
    max_turns: int,
    provider: str,
    api_kind: str,
    agent_flavor: str,
    trace_path: str,
    summary_path: str,
    python: str = f"{DEFAULT_VENV_PATH}/bin/python",
    trace_id: str | None = None,
) -> str:
    """Build the in-container command that starts the SAL runner."""

    parts = [
        python,
        "-m",
        "simple_agent_lab.evals.harbor.runner",
        "--instruction-file",
        instruction_path,
        "--cwd",
        cwd,
        "--max-turns",
        str(max_turns),
        "--provider",
        provider,
        "--api-kind",
        api_kind,
        "--agent-flavor",
        agent_flavor,
        "--trace-path",
        trace_path,
        "--summary-path",
        summary_path,
    ]
    if trace_id:
        parts.extend(["--trace-id", trace_id])
    return _quote_parts(parts)


if _HARBOR_IMPORT_ERROR is not None:

    class SimpleAgentLabHarborAgent:  # pragma: no cover - construction guard
        """Placeholder that keeps module importable without Harbor installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise ImportError(
                "Harbor is required to construct SimpleAgentLabHarborAgent. "
                "Install the optional Harbor extra in a Python 3.12+ environment."
            ) from _HARBOR_IMPORT_ERROR

else:

    class SimpleAgentLabHarborAgent(BaseInstalledAgent):
        """Run Simple Agent Lab inside Harbor-managed task environments."""

        def __init__(
            self,
            max_turns: int | str = DEFAULT_MAX_TURNS,
            agent_flavor: str = "bash_task_read",
            provider: str = "openai",
            api_kind: str = DEFAULT_API_KIND,
            python_version: str = "3.12",
            sal_package: str = DEFAULT_SAL_PACKAGE,
            sal_source: str | None = None,
            venv_path: str = DEFAULT_VENV_PATH,
            cwd: str | None = None,
            install_timeout_sec: int | str = _DEFAULT_INSTALL_TIMEOUT_SEC,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self._max_turns = int(max_turns)
            self._agent_flavor = agent_flavor
            self._provider = provider
            self._api_kind = api_kind
            self._python_version = str(python_version)
            self._sal_package = sal_package
            self._sal_source = sal_source
            self._venv_path = venv_path
            self._cwd = cwd
            self._install_timeout_sec = int(install_timeout_sec)

        @staticmethod
        def name() -> str:
            return "simple-agent-lab"

        async def _installed_sal_available(self, environment: BaseEnvironment) -> bool:
            result = await environment.exec(
                command=build_sal_installed_check_command(venv_path=self._venv_path),
                timeout_sec=30,
            )
            return result.return_code == 0

        def _setup_env(self) -> dict[str, str]:
            setup_env: dict[str, str] = {}
            for key, value in self.extra_env.items():
                if key.startswith(_SETUP_ENV_PREFIX):
                    target_key = key.removeprefix(_SETUP_ENV_PREFIX)
                    if target_key:
                        setup_env[target_key] = value
            return setup_env

        async def install(self, environment: BaseEnvironment) -> None:
            if await self._installed_sal_available(environment):
                self.logger.debug(
                    "Simple Agent Lab is already available in the requested venv"
                )
                return

            agent_user = environment.default_user or "root"
            quoted_venv = shlex.quote(self._venv_path)
            quoted_user = shlex.quote(str(agent_user))
            source_root = (
                Path(self._sal_source).expanduser().resolve()
                if self._sal_source
                else find_sal_source_root()
            )
            setup_env = self._setup_env()
            await self.exec_as_root(
                environment,
                build_sal_system_dependencies_command(),
                env={"DEBIAN_FRONTEND": "noninteractive", **setup_env},
                timeout_sec=self._install_timeout_sec,
            )
            await self.exec_as_root(
                environment,
                f"mkdir -p {quoted_venv} && chown {quoted_user}:{quoted_user} {quoted_venv}",
            )
            await self.exec_as_agent(
                environment,
                build_sal_venv_command(
                    venv_path=self._venv_path,
                    python_version=self._python_version,
                ),
                env=setup_env or None,
                timeout_sec=self._install_timeout_sec,
            )
            if source_root is not None:
                local_archive = build_sal_source_archive(
                    source_root,
                    self.logs_dir / "setup" / "simple-agent-lab-src.tar.gz",
                )
                await environment.upload_file(
                    source_path=local_archive,
                    target_path=REMOTE_SOURCE_ARCHIVE_PATH,
                )
                await self.exec_as_agent(
                    environment,
                    (
                        f"rm -rf {shlex.quote(REMOTE_SOURCE_DIR)} && "
                        f"mkdir -p {shlex.quote(REMOTE_SOURCE_DIR)} && "
                        f"{quoted_venv}/bin/python -m tarfile -e "
                        f"{shlex.quote(REMOTE_SOURCE_ARCHIVE_PATH)} "
                        f"{shlex.quote(REMOTE_SOURCE_DIR)}"
                    ),
                    timeout_sec=60,
                )
                install_target = REMOTE_SOURCE_DIR
            else:
                install_target = self._sal_package
            await self.exec_as_agent(
                environment,
                build_sal_package_install_command(
                    venv_path=self._venv_path,
                    install_target=install_target,
                    local_source=source_root is not None,
                ),
                env=setup_env or None,
                timeout_sec=self._install_timeout_sec,
            )

        def _runner_env(self) -> dict[str, str]:
            env = {
                key: value
                for key, value in self.extra_env.items()
                if not key.startswith(_SETUP_ENV_PREFIX)
            }
            if self.model_name and "OPENAI_MODEL" not in env:
                env["OPENAI_MODEL"] = self.model_name
            env.setdefault("API_KIND", self._api_kind)
            return env

        def _resolve_cwd(self, environment: BaseEnvironment) -> str:
            if self._cwd is not None and self._cwd != "":
                return self._cwd
            task_env_config = getattr(environment, "task_env_config", None)
            workdir = getattr(task_env_config, "workdir", None)
            return str(workdir or ".")

        def _resolve_exec_cwd(self, environment: BaseEnvironment) -> str | None:
            cwd = self._resolve_cwd(environment)
            if Path(cwd).is_absolute():
                return cwd
            return None

        def _runner_command(self, environment: BaseEnvironment) -> str:
            cwd = self._resolve_cwd(environment)
            command = build_sal_runner_command(
                instruction_path=REMOTE_INSTRUCTION_PATH,
                cwd=cwd,
                max_turns=self._max_turns,
                provider=self._provider,
                api_kind=self._api_kind,
                agent_flavor=self._agent_flavor,
                trace_path=DEFAULT_TRACE_PATH,
                summary_path=DEFAULT_SUMMARY_PATH,
                python=f"{self._venv_path}/bin/python",
                trace_id=str(
                    self.context_id or self.session_id or "harbor.simple-agent-lab"
                ),
            )
            return f"{command} 2>&1 | tee {shlex.quote(REMOTE_OUTPUT_PATH)}"

        @with_prompt_template
        async def run(
            self,
            instruction: str,
            environment: BaseEnvironment,
            context: AgentContext,
        ) -> None:
            await self.exec_as_agent(environment, f"mkdir -p {REMOTE_AGENT_DIR}")
            local_instruction = self.logs_dir / "sal-instruction.txt"
            local_instruction.parent.mkdir(parents=True, exist_ok=True)
            local_instruction.write_text(instruction, encoding="utf-8")
            await environment.upload_file(
                source_path=local_instruction,
                target_path=REMOTE_INSTRUCTION_PATH,
            )

            await self.exec_as_agent(
                environment,
                self._runner_command(environment),
                env=self._runner_env(),
                cwd=self._resolve_exec_cwd(environment),
            )
            self.populate_context_post_run(context)

        def populate_context_post_run(self, context: AgentContext) -> None:
            summary_path = self.logs_dir / Path(DEFAULT_SUMMARY_PATH).name
            if not summary_path.exists():
                return
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return
            metadata = dict(context.metadata or {})
            metadata["simple_agent_lab"] = summary
            context.metadata = metadata


__all__ = [
    "AGENT_IMPORT_PATH",
    "SimpleAgentLabHarborAgent",
    "build_sal_installed_check_command",
    "build_sal_package_install_command",
    "build_sal_system_dependencies_command",
    "build_sal_venv_command",
    "build_sal_source_archive",
    "build_sal_runner_command",
    "find_sal_source_root",
]
