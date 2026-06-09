"""ProgramBench as a `Suite` (ADR 0017) — the reverse-engineering host half.

This maps ProgramBench (facebookresearch/programbench) onto one `Suite` whose
`launch_spec` carries the per-instance launch values as **data**. Two values
encode the deal in ADR 0022:

- `network_mode="host"` keeps the container online (so the in-container agent
  can reach the model API and bootstrap can install the wheel), while
- `cap_add=("SYS_ADMIN",)` lets the container half wrap each agent bash command
  in ``unshare --net`` — so the agent's *commands* have no network even though
  the container does. That restores ProgramBench's offline anti-cheat without
  the ``--network none`` that would also cut off the agent's own model calls.

The container half (``build_task`` / ``build_agent`` / ``prepare`` /
``extract_result``) ships in the wheel at
``simple_agent_lab.evals.suites.programbench.container``. Scoring is the official
``programbench eval`` CLI, run on the host by ``evaluate_submissions.py`` (no
in-environment ``evaluate`` hook, so ``eval_inputs`` returns ``None``).
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from simple_agent_lab.evals.protocols import LaunchSpec  # noqa: E402

from . import harness  # noqa: E402


class ProgrambenchSuite:
    """`Suite` for ProgramBench reverse-engineering instances."""

    name = "programbench"
    # Container half ships in the wheel; the generic runner imports it by
    # dotted path with zero file copying.
    container_module = "simple_agent_lab.evals.suites.programbench.container"

    def __init__(
        self,
        *,
        image_tag: str = harness.DEFAULT_IMAGE_TAG,
        platform: str = "",
        network_mode: str = "host",
        cap_add: Sequence[str] = ("SYS_ADMIN",),
    ) -> None:
        self.image_tag = image_tag
        self.platform = platform
        # Online container (model API + wheel install); agent *commands* are
        # isolated per-command via unshare --net in the container half.
        self.network_mode = network_mode
        # SYS_ADMIN is what `unshare --net` needs to build the per-command
        # network namespace; drop it and the container half falls back to
        # un-isolated commands (and says so in the result).
        self.cap_add = tuple(cap_add)

    def launch_spec(self, instance: Mapping[str, Any]) -> LaunchSpec:
        image = harness.image_for_instance(dict(instance), image_tag=self.image_tag)
        return LaunchSpec(
            image=image,
            workdir=harness.DEFAULT_WORKDIR,
            shell=("bash", "-lc"),
            entrypoint=None,
            platform=self.platform or None,
            network_mode=self.network_mode or None,
            cap_add=self.cap_add,
        )

    def task_input(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        return harness.sanitized_instance(dict(instance))

    def eval_inputs(self, instance: Mapping[str, Any]) -> dict[str, Any] | None:
        # ProgramBench scores with the official `programbench eval` CLI on the
        # host (compile → restore ./executable → per-branch pytest), not in the
        # run environment, so no gold is staged and the container half exposes
        # no `evaluate` hook (ADR 0022 / ADR 0020).
        del instance
        return None
