"""SWE-bench as a `Suite` (ADR 0017) — the reference driver case.

This is the *host half*: it maps SWE-bench Verified and SWE-bench Pro onto one
`Suite` whose `container_plan` carries the per-suite differences as **data**
(image, workdir, shell, entrypoint) instead of the runner branching on
``is_swebench_pro_instance(...)``. It delegates to the existing, battle-tested
image/launch helpers in `containerized_agent` so behavior is unchanged and
`evaluate_predictions.py` (scoring) is untouched.

The *container half* (``build_task`` / ``prepare`` / ``extract_result``) ships
in the wheel at ``simple_agent_lab.evals.suites.swebench.container`` and is
driven by the generic in-container runner — so the container needs no copied
files. This suite is the reference for the "one Suite + two functions"
integration shape (ADR 0017).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from simple_agent_lab.evals.protocols import ContainerPlan

from . import containerized_agent as ca
from . import in_container_runner as icr


class SwebenchSuite:
    """`Suite` for the SWE-bench family (Verified + Pro)."""

    name = "swebench"
    # Container half ships in the wheel; the generic runner imports it by
    # dotted path with zero file copying.
    container_module = "simple_agent_lab.evals.suites.swebench.container"

    def __init__(
        self,
        *,
        dataset_name: str = ca.DEFAULT_DATASET,
        namespace: str = "swebench",
        instance_image_tag: str = "latest",
        env_image_tag: str = "latest",
        dockerhub_username: str = ca.DEFAULT_PRO_DOCKERHUB_USERNAME,
        platform: str = "",
        network_mode: str = "",
    ) -> None:
        self.dataset_name = dataset_name
        self.namespace = namespace
        self.instance_image_tag = instance_image_tag
        self.env_image_tag = env_image_tag
        self.dockerhub_username = dockerhub_username
        self.platform = platform
        self.network_mode = network_mode

    def container_plan(self, instance: Mapping[str, Any]) -> ContainerPlan:
        record = dict(instance)
        image = ca.docker_image_for_instance(
            record,
            dataset_name=self.dataset_name,
            namespace=self.namespace,
            instance_image_tag=self.instance_image_tag,
            env_image_tag=self.env_image_tag,
            dockerhub_username=self.dockerhub_username,
        )
        workdir = ca.resolve_workdir("", record, dataset_name=self.dataset_name)
        # Pro images set /bin/bash as ENTRYPOINT and use /bin/sh; Verified
        # images run bash -lc with the image default entrypoint.
        shell = tuple(ca.docker_run_command("", record, dataset_name=self.dataset_name))
        shell = shell[:-1] if shell and shell[-1] == "" else shell
        entrypoint = ca.container_entrypoint_override(
            record, dataset_name=self.dataset_name
        ).get("entrypoint")
        return ContainerPlan(
            image=image,
            workdir=workdir,
            shell=shell,
            entrypoint=entrypoint,
            platform=self.platform or None,
            network_mode=self.network_mode or None,
        )

    def sanitize_instance(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        return ca.sanitized_instance(dict(instance))

    def prediction_record(
        self,
        instance: Mapping[str, Any],
        *,
        model_name: str,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        return icr.prediction_record(
            str(instance["instance_id"]),
            model_name,
            str(result.get("model_patch", "")),
            dataset_name=self.dataset_name,
        )
