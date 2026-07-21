"""SWE-bench as a `Suite` (ADR generic-containerized-eval-framework) — the reference driver case.

This is the *host half*: it maps SWE-bench Verified, SWE-bench Multilingual, and
SWE-bench Pro onto one `Suite` whose `launch_spec` carries the per-suite
differences as **data**
(image, workdir, shell, entrypoint) instead of the runner branching on
``is_swebench_pro_instance(...)``. It delegates to the shared image/launch
helpers in `harness` so behavior is consistent across the run entry and scoring.

The *container half* (``build_task`` / ``prepare`` / ``extract_result``) ships
in the wheel at ``simple_agent_lab.evals.suites.swebench.container`` and is
driven by the generic in-container runner — so the container needs no copied
files. This suite is the reference for the "one Suite + two functions"
integration shape (ADR generic-containerized-eval-framework).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from simple_agent_lab.evals.protocols import LaunchSpec

from . import harness


class SwebenchSuite:
    """`Suite` for the SWE-bench family (Verified + Multilingual + Pro)."""

    # Container half ships in the wheel; the generic runner imports it by
    # dotted path with zero file copying.
    container_module = "simple_agent_lab.evals.suites.swebench.container"

    def __init__(
        self,
        *,
        dataset_name: str = harness.DEFAULT_DATASET,
        namespace: str = "swebench",
        instance_image_tag: str = "latest",
        env_image_tag: str = "latest",
        dockerhub_username: str = harness.DEFAULT_PRO_DOCKERHUB_USERNAME,
        platform: str = "",
        network_mode: str = "",
        security_opt: tuple[str, ...] = ("seccomp=unconfined",),
        in_env_scoring: bool = False,
        mem_limit: str | None = "16g",
    ) -> None:
        self.dataset_name = dataset_name
        self.name = harness.suite_for_instance(
            dataset_name=self.dataset_name,
            instance_id="",
        )
        self.namespace = namespace
        self.instance_image_tag = instance_image_tag
        self.env_image_tag = env_image_tag
        self.dockerhub_username = dockerhub_username
        self.platform = platform
        self.network_mode = network_mode
        # Default seccomp=unconfined: SWE-bench eval containers are disposable
        # sandboxes, and old daemons (≤18.09) block clone3 under their default
        # profile, which glibc ≥2.34 needs for threads. Override via --security-opt.
        self.security_opt = tuple(security_opt)
        # Where scoring runs (ADR collapse-scorer-seam-into-run-primitive). Default: score separately with the
        # official harness (`evaluate_predictions.py`) — re-scorable, parity by
        # definition. Set `in_env_scoring=True` to also stage the official eval
        # script as `eval_inputs`, which turns on the container-half ``evaluate``
        # hook so the run scores itself in place (grade host-side with
        # `evaluate_predictions.reuse_eval_row`).
        self.in_env_scoring = in_env_scoring
        # Per-container memory guardrail (docker --memory): caps physical RAM so
        # one runaway build/test step can't exhaust the host or starve sibling
        # runs in a parallel batch. memswap_limit is left unset so Docker keeps
        # its default swap headroom — a brief overshoot degrades instead of
        # OOM-killing the instance. "16g" matches the mini-SWE-agent Pro
        # baseline envelope; pass None to disable.
        self.mem_limit = mem_limit

    def launch_spec(self, instance: Mapping[str, Any]) -> LaunchSpec:
        record = dict(instance)
        image = harness.docker_image_for_instance(
            record,
            dataset_name=self.dataset_name,
            namespace=self.namespace,
            instance_image_tag=self.instance_image_tag,
            env_image_tag=self.env_image_tag,
            dockerhub_username=self.dockerhub_username,
        )
        workdir = harness.resolve_workdir("", record, dataset_name=self.dataset_name)
        # Pro images set /bin/bash as ENTRYPOINT and use /bin/sh; Verified
        # images run bash -lc with the image default entrypoint.
        shell = tuple(
            harness.docker_run_command("", record, dataset_name=self.dataset_name)
        )
        shell = shell[:-1] if shell and shell[-1] == "" else shell
        entrypoint = harness.container_entrypoint_override(
            record, dataset_name=self.dataset_name
        ).get("entrypoint")
        return LaunchSpec(
            image=image,
            workdir=workdir,
            shell=shell,
            entrypoint=entrypoint,
            platform=self.platform or None,
            network_mode=self.network_mode or None,
            cap_add=self._cap_add(record),
            security_opt=self.security_opt,
            mem_limit=self.mem_limit,
        )

    def _cap_add(self, record: dict[str, Any]) -> tuple[str, ...]:
        """Capabilities the SWE-bench test spec requests (e.g. SYS_PTRACE).

        These come from the instance's test spec ``run_args``; Pro images carry
        none. Without this the container starts under-privileged and
        capability-dependent test steps fail.
        """

        if harness.is_swebench_pro_instance(record, dataset_name=self.dataset_name):
            return ()
        spec = harness._make_swebench_test_spec(
            record,
            namespace=self.namespace,
            instance_image_tag=self.instance_image_tag,
            env_image_tag=self.env_image_tag,
        )
        run_args = spec.docker_specs.get("run_args", {}) or {}
        return tuple(run_args.get("cap_add", []) or [])

    def task_input(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        return harness.sanitized_instance(dict(instance))

    def eval_inputs(self, instance: Mapping[str, Any]) -> dict[str, Any] | None:
        """Gold scoring inputs for in-environment scoring (agent never sees them).

        Staging these is the toggle for the container-half ``evaluate`` hook: the
        host generates the official eval script from ``make_test_spec`` and stages
        it (under EVAL_KEY) so the hook runs the *official* script for parity.
        Returns ``None`` unless ``in_env_scoring`` is set, and only for
        Verified-style instances (Pro scores via the separate harness). Without
        it, score separately with `evaluate_predictions.py`.
        """

        if not self.in_env_scoring:
            return None
        record = dict(instance)
        if harness.is_swebench_pro_instance(record, dataset_name=self.dataset_name):
            return None
        spec = harness._make_swebench_test_spec(
            record,
            namespace=self.namespace,
            instance_image_tag=self.instance_image_tag,
            env_image_tag=self.env_image_tag,
        )
        return {
            "instance_id": str(record.get("instance_id") or ""),
            "eval_script": spec.eval_script,
            "fail_to_pass": record.get("FAIL_TO_PASS") or record.get("fail_to_pass"),
            "pass_to_pass": record.get("PASS_TO_PASS") or record.get("pass_to_pass"),
        }
