"""ProgramBench as a `Suite` — the reverse-engineering host half.

This maps ProgramBench (facebookresearch/programbench) onto one `Suite` whose
`launch_spec` carries the per-instance launch values as **data**. Two values
encode the benchmark boundary:

- `network_mode="host"` keeps the container online (so the in-container agent
  can reach the model API and bootstrap can install the wheel), while
- `cap_add=("SYS_ADMIN",)` lets the container half wrap each agent bash command
  in ``unshare --net`` — so the agent's *commands* have no network even though
  the container does. That restores ProgramBench's offline anti-cheat without
  the ``--network none`` that would also cut off the agent's own model calls.

The container half (``build_task`` / ``build_agent`` / ``prepare`` /
``extract_result``) ships in the wheel at
``simple_agent_lab.evals.suites.programbench.container``. Scoring is the official
official ProgramBench evaluator, run on the host by ``evaluate_submissions.py`` (no
in-environment ``evaluate`` hook, so ``eval_inputs`` returns ``None``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from simple_agent_lab.evals.protocols import LaunchSpec

from . import harness


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
        security_opt: tuple[str, ...] = ("seccomp=unconfined",),
        cpus: int | None = 20,
        mem_limit: str | None = "60g",
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
        # Default seccomp=unconfined so older daemons (whose default profile
        # predates `clone3`) don't kill the agent's threads — same default as
        # SwebenchSuite. Pass seccomp=default to restore the daemon's profile.
        self.security_opt = tuple(security_opt)
        self.cpus = cpus
        self.mem_limit = mem_limit

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
            security_opt=self.security_opt,
            nano_cpus=self.cpus * 1_000_000_000 if self.cpus else None,
            mem_limit=self.mem_limit,
            memswap_limit=self.mem_limit,
        )

    def task_input(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        return harness.sanitized_instance(dict(instance))

    def eval_inputs(self, instance: Mapping[str, Any]) -> dict[str, Any] | None:
        # ProgramBench scores with the official evaluator on the
        # host (compile → restore ./executable → per-branch pytest), not in the
        # run environment, so no gold is staged and the container half exposes
        # no `evaluate` hook.
        del instance
        return None
