"""Editable Python-agent package used by self-evolving recipes.

A package is a mapping of relative path -> file text. ``default_agent_package()``
reproduces the default bash agent, so an unedited package is behavior-neutral.
``load_agent_package`` materializes the files, imports ``agent_program``, and
returns its ``build_agent`` callable (or ``None`` if the code is invalid).

This helper is benchmark-neutral: suites decide how to stage the package into a
run, while the package itself is just the editable Python agent program behind
``python_agent_surface``.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Mapping

ENTRY_MODULE_FILENAME = "agent_program.py"
ENTRY_IMPORT_NAME = "sal_evolved_agent_program"

_DEFAULT_AGENT_PROGRAM = '''\
"""Evolvable agent program. Edit build_agent (and add files) to change behavior.

build_agent receives the provider, the working directory, and the suite's
default system prompt, and returns an Agent. It may use anything in the
installed simple_agent_lab wheel (tools, agents.starter, core). Keep the
function present and returning an Agent; the harness keeps the task framing,
artifact extraction, and scoring outside this file.
"""

from __future__ import annotations

from pathlib import Path

from simple_agent_lab.agents.starter import make_bash_agent
from simple_agent_lab.core import Agent
from simple_agent_lab.llm import Provider


def build_agent(*, provider: Provider, cwd: Path, base_system_prompt: str) -> Agent:
    return make_bash_agent(
        provider=provider,
        cwd=cwd,
        name="evolving_agent",
        system_prompt=base_system_prompt,
    )
'''


def default_agent_package() -> dict[str, str]:
    """Return the behavior-neutral default agent package files."""

    return {ENTRY_MODULE_FILENAME: _DEFAULT_AGENT_PROGRAM}


@dataclass(frozen=True)
class AgentPackageLoad:
    builder: Callable[..., object] | None
    loaded: bool
    error: str = ""


def load_agent_package(
    files: Mapping[str, str], *, root: Path
) -> Callable[..., object] | None:
    """Materialize ``files`` under ``root``, import the entry, return build_agent.

    Returns ``None`` (never raises) when files are missing, do not parse/import,
    or lack a callable ``build_agent``. The caller can then fall back to its
    built-in default agent.
    """

    return load_agent_package_result(files, root=root).builder


def load_agent_package_result(
    files: Mapping[str, str], *, root: Path
) -> AgentPackageLoad:
    """Load an agent package and preserve load diagnostics."""

    try:
        for rel, text in files.items():
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8")
        module_path = root / ENTRY_MODULE_FILENAME
        if not module_path.is_file():
            return AgentPackageLoad(None, False, f"missing {ENTRY_MODULE_FILENAME}")
        spec = importlib.util.spec_from_file_location(ENTRY_IMPORT_NAME, module_path)
        if spec is None or spec.loader is None:
            return AgentPackageLoad(None, False, "could not create import spec")
        module = importlib.util.module_from_spec(spec)
        with _package_import_context(root, files):
            spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - diagnostic boundary for untrusted code.
        return AgentPackageLoad(None, False, f"{type(exc).__name__}: {exc}")
    builder = getattr(module, "build_agent", None)
    if not callable(builder):
        return AgentPackageLoad(None, False, "missing callable build_agent")
    return AgentPackageLoad(_wrap_builder(builder, root, files), True)


def _wrap_builder(
    builder: Callable[..., object], root: Path, files: Mapping[str, str]
) -> Callable[..., object]:
    def call(*args: object, **kwargs: object) -> object:
        with _package_import_context(root, files):
            return builder(*args, **kwargs)

    return call


@contextmanager
def _package_import_context(root: Path, files: Mapping[str, str]) -> Iterator[None]:
    import_roots = _top_level_import_roots(files)
    previous_modules = _snapshot_modules(import_roots)
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(root))
        _remove_modules(import_roots)
        yield
    finally:
        sys.path[:] = old_path
        _remove_modules(import_roots)
        sys.modules.update(previous_modules)


def _snapshot_modules(import_roots: tuple[str, ...]) -> dict[str, ModuleType]:
    return {
        name: module
        for name, module in sys.modules.items()
        if _belongs_to_roots(name, import_roots)
    }


def _remove_modules(import_roots: tuple[str, ...]) -> None:
    for name in tuple(sys.modules):
        if _belongs_to_roots(name, import_roots):
            sys.modules.pop(name, None)


def _belongs_to_roots(module_name: str, import_roots: tuple[str, ...]) -> bool:
    return any(
        module_name == root or module_name.startswith(root + ".")
        for root in import_roots
    )


def _top_level_import_roots(files: Mapping[str, str]) -> tuple[str, ...]:
    roots: set[str] = set()
    for rel in files:
        path = Path(rel)
        if path.suffix != ".py" or not path.parts:
            continue
        root = path.stem if len(path.parts) == 1 else path.parts[0]
        if root != Path(ENTRY_MODULE_FILENAME).stem and root.isidentifier():
            roots.add(root)
    return tuple(sorted(roots))
