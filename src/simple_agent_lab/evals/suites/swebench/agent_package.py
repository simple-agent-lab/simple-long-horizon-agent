"""Evolvable SWE-bench agent program: the package the meta-agent edits.

A package is a mapping of relative path -> file text. ``default_agent_package()``
reproduces today's bash agent, so an unedited package is behavior-neutral.
``load_agent_package`` materializes the files, imports ``agent_program``, and
returns its ``build_agent`` callable (or ``None`` if the code is invalid). The
evolving container module uses it ahead of the baked-in agent; an invalid
package falls back, so a broken meta-agent edit can never break the harness.

Stdlib-only at module top so it imports inside any SWE-bench image; the entry
module may import the installed wheel (``simple_agent_lab``), which is present in
the eval image.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable, Mapping

ENTRY_MODULE_FILENAME = "agent_program.py"
ENTRY_IMPORT_NAME = "sal_evolved_agent_program"

_DEFAULT_AGENT_PROGRAM = '''\
"""Evolvable agent program. Edit build_agent (and add files) to change behavior.

build_agent receives the provider, the working directory, and the suite's
default system prompt, and returns an Agent. It may use anything in the
installed simple_agent_lab wheel (tools, llm_agent, agents.starter). Keep the
function present and returning an Agent; the harness keeps the task framing,
git-diff extraction, and scoring outside this file.
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
        name="swebench_agent",
        system_prompt=base_system_prompt,
    )
'''


def default_agent_package() -> dict[str, str]:
    """Return the behavior-neutral default agent package files."""

    return {ENTRY_MODULE_FILENAME: _DEFAULT_AGENT_PROGRAM}


def load_agent_package(
    files: Mapping[str, str], *, root: Path
) -> Callable[..., object] | None:
    """Materialize ``files`` under ``root``, import the entry, return build_agent.

    Returns ``None`` (never raises) when files are missing, do not parse/import,
    or lack a callable ``build_agent`` — the caller then falls back to the
    baked-in agent.
    """

    try:
        for rel, text in files.items():
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8")
        module_path = root / ENTRY_MODULE_FILENAME
        if not module_path.is_file():
            return None
        spec = importlib.util.spec_from_file_location(ENTRY_IMPORT_NAME, module_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        return None
    builder = getattr(module, "build_agent", None)
    return builder if callable(builder) else None
