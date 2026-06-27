"""Host-side SWE-bench helpers shared by the suite, the runner, and scoring.

These are the reusable, Docker-free building blocks the SWE-bench integration
needs on the host: image/launch resolution (from the official ``make_test_spec``),
SWE-bench Pro detection and Docker Hub image naming, instance loading, dotenv +
provider environment, the offline wheelhouse build, and prediction shaping for
the official harness.

`SwebenchSuite` (`suite.py`) consumes the image/launch helpers; the run entry
(`runs/run_swebench_suite.py`) consumes instance loading + env + wheelhouse prep;
`evaluate_predictions.py` consumes the test spec + prediction shaping. The agent
loop itself lives in the wheel (`simple_agent_lab.evals.in_container` + the
SWE-bench container half), so nothing here launches or talks to a container.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# The OpenAI env-var names and the `.env` loader are owned by
# `simple_agent_lab.llm.env` (single source of truth). This host-side harness
# only forwards these names into the container; the container half reads them
# via the same module. See ADR consolidate-provider-env.
from simple_agent_lab.agent_flavors import (  # noqa: E402
    AGENT_FLAVORS,
    DEFAULT_AGENT_FLAVOR as _DEFAULT_AGENT_FLAVOR,
)
from simple_agent_lab.llm.env import (  # noqa: E402
    API_KIND_ENV,
    OPENAI_AUTH_ENV,
    OPENAI_BASE_URL_ENV,
    OPENAI_LOG_ID_ENV,
    OPENAI_MODEL_ENV,
    OPENAI_REASONING_EFFORT_ENV,
    OPENAI_SESSION_ID_ENV,
    REASONING_EFFORT_ENV,
)

# Re-exported so the run entry (`runs/run_swebench_suite.py`) keeps calling
# `harness.load_dotenv`; the implementation is owned by `llm.env`.
from simple_agent_lab.llm.env import load_dotenv as load_dotenv  # noqa: E402,F401

DEFAULT_DATASET = "princeton-nlp/SWE-bench_Verified"
DEFAULT_MULTILINGUAL_DATASET = "SWE-bench/SWE-bench_Multilingual"
DEFAULT_SPLIT = "test"
DEFAULT_WHEELHOUSE_MOUNT = "/agent/wheelhouse"
DEFAULT_WORKDIR = "/testbed"
DEFAULT_PRO_WORKDIR = "/app"
DEFAULT_PRO_DOCKERHUB_USERNAME = "jefzda"
DEFAULT_RUN_ROOT = ROOT / "evals/out/swebench"
DEFAULT_WHEELHOUSE = ROOT / "evals/out/swebench/wheelhouse/cp311-manylinux"
DEFAULT_MULTILINGUAL_RUN_ROOT = ROOT / "evals/out/swebench_multilingual"
DEFAULT_MULTILINGUAL_WHEELHOUSE = (
    ROOT / "evals/out/swebench_multilingual/wheelhouse/cp311-manylinux"
)
DEFAULT_PRO_RUN_ROOT = ROOT / "evals/out/swebench_pro"
DEFAULT_PRO_WHEELHOUSE = ROOT / "evals/out/swebench_pro/wheelhouse/cp311-manylinux"
DEFAULT_UV_BINARY = shutil.which("uv") or ""
# This suite intentionally accepts only the OpenAI-protocol adapters (not the
# broader set in `llm.env.API_KIND_CHOICES`), so it is declared locally. The
# reasoning-effort names are imported above from `llm.env`; forwarding them keeps
# the in-container agent from silently running at the endpoint's default depth.
API_KIND_CHOICES = ("openai-chat", "openai-responses")
# The single agent selector (`--agent-flavor` / AGENT_FLAVOR). The vocabulary is
# owned by `simple_agent_lab.agent_flavors` so host and container choices cannot
# drift.
AGENT_FLAVOR_CHOICES = AGENT_FLAVORS
DEFAULT_AGENT_FLAVOR = _DEFAULT_AGENT_FLAVOR
SWE_BENCH_PRO_DATASET_MARKER = "swe-bench_pro"
SWE_BENCH_MULTILINGUAL_DATASET_MARKER = "swe-bench_multilingual"
OPENAI_PASSTHROUGH_ENVS = (
    OPENAI_MODEL_ENV,
    OPENAI_AUTH_ENV,
    OPENAI_BASE_URL_ENV,
    OPENAI_SESSION_ID_ENV,
    OPENAI_LOG_ID_ENV,
    API_KIND_ENV,
    REASONING_EFFORT_ENV,
    OPENAI_REASONING_EFFORT_ENV,
)
PRIVATE_INSTANCE_FIELDS = {
    "patch",
    "test_patch",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "fail_to_pass",
    "pass_to_pass",
    "selected_test_files_to_run",
}


# --------------------------------------------------------------------------- #
# Instance shaping + SWE-bench Pro detection
# --------------------------------------------------------------------------- #
def sanitized_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Drop gold/private fields so the agent never sees the solution."""

    return {
        str(key): value
        for key, value in instance.items()
        if str(key) not in PRIVATE_INSTANCE_FIELDS
    }


def is_swebench_pro_instance(
    instance: dict[str, Any],
    *,
    dataset_name: str = "",
) -> bool:
    """Return True for SWE-bench Pro records and dataset names."""

    dataset = dataset_name.casefold()
    instance_id = str(instance.get("instance_id") or "")
    return (
        "swe-bench_pro" in dataset
        or "dockerhub_tag" in instance
        or instance_id.startswith("instance_")
    )


def is_swebench_pro(*, dataset_name: str = "", instance_id: str = "") -> bool:
    """Pro check by dataset name / instance id (prediction shaping helper)."""

    dataset = dataset_name.casefold()
    return SWE_BENCH_PRO_DATASET_MARKER in dataset or instance_id.startswith(
        "instance_"
    )


def is_swebench_multilingual(*, dataset_name: str = "") -> bool:
    """Multilingual check by dataset name."""

    return SWE_BENCH_MULTILINGUAL_DATASET_MARKER in dataset_name.casefold()


def suite_for_instance(*, dataset_name: str, instance_id: str) -> str:
    if is_swebench_pro(dataset_name=dataset_name, instance_id=instance_id):
        return "swebench_pro"
    if is_swebench_multilingual(dataset_name=dataset_name):
        return "swebench_multilingual"
    return "swebench"


# --------------------------------------------------------------------------- #
# Image + launch resolution (delegates to the official make_test_spec)
# --------------------------------------------------------------------------- #
def resolve_workdir(
    workdir: str,
    instance: dict[str, Any],
    *,
    dataset_name: str,
) -> str:
    """Return an explicit workdir or the suite default for the instance."""

    if workdir:
        return workdir
    if is_swebench_pro_instance(instance, dataset_name=dataset_name):
        return DEFAULT_PRO_WORKDIR
    return DEFAULT_WORKDIR


def docker_image_for_instance(
    instance: dict[str, Any],
    *,
    dataset_name: str,
    namespace: str,
    instance_image_tag: str,
    env_image_tag: str,
    dockerhub_username: str = DEFAULT_PRO_DOCKERHUB_USERNAME,
) -> str:
    """Return the Docker image key for a SWE-bench-family instance."""

    if is_swebench_pro_instance(instance, dataset_name=dataset_name):
        return _pro_dockerhub_image(instance, dockerhub_username=dockerhub_username)

    spec = _make_swebench_test_spec(
        instance,
        namespace=namespace,
        instance_image_tag=instance_image_tag,
        env_image_tag=env_image_tag,
    )
    return str(spec.instance_image_key)


def docker_run_command(
    command: str,
    instance: dict[str, Any],
    *,
    dataset_name: str,
) -> list[str]:
    """Return the Docker command shape for the suite's image entrypoint."""

    if is_swebench_pro_instance(instance, dataset_name=dataset_name):
        return ["/bin/sh", "-lc", command]
    return ["bash", "-lc", command]


def container_entrypoint_override(
    instance: dict[str, Any],
    *,
    dataset_name: str,
) -> dict[str, str]:
    if is_swebench_pro_instance(instance, dataset_name=dataset_name):
        return {"entrypoint": ""}
    return {}


def _pro_dockerhub_image(
    instance: dict[str, Any],
    *,
    dockerhub_username: str,
) -> str:
    tag = str(instance.get("dockerhub_tag") or "").strip()
    if not tag:
        tag = _pro_dockerhub_tag_from_instance(instance)
    return f"{dockerhub_username}/sweap-images:{tag}"


def _make_swebench_test_spec(
    instance: dict[str, Any],
    *,
    namespace: str,
    instance_image_tag: str,
    env_image_tag: str,
) -> Any:
    module = importlib.import_module("swebench.harness.test_spec.test_spec")
    return module.make_test_spec(
        instance,
        namespace=namespace,
        instance_image_tag=instance_image_tag,
        env_image_tag=env_image_tag,
    )


def _pro_dockerhub_tag_from_instance(instance: dict[str, Any]) -> str:
    instance_id = str(instance.get("instance_id") or "")
    repo = str(instance.get("repo") or "")
    if "/" not in repo:
        raise SystemExit(
            "SWE-bench Pro instances without dockerhub_tag must include repo "
            f"as 'owner/name'; got {repo!r}."
        )
    repo_base, repo_name = repo.casefold().split("/", 1)
    suffix = instance_id.removeprefix("instance_")

    if (
        instance_id
        == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan"
    ):
        repo_name = "element-web"
    elif repo_base == "element-hq" and repo_name == "element-web":
        repo_name = "element"
        if suffix.endswith("-vnan"):
            suffix = suffix[:-5]
    elif suffix.endswith("-vnan"):
        suffix = suffix[:-5]

    tag = f"{repo_base}.{repo_name}-{suffix}"
    return tag[:128]


# --------------------------------------------------------------------------- #
# Instance loading
# --------------------------------------------------------------------------- #
def load_instance(path: str | Path, instance_id: str | None) -> dict[str, Any]:
    """Load one instance record from JSON or JSONL."""

    records = _load_instance_records(Path(path))
    if not records:
        raise SystemExit(f"No instance records found in {path}")
    if instance_id is None:
        return dict(records[0])
    for record in records:
        if str(record.get("instance_id")) == instance_id:
            return dict(record)
    raise SystemExit(f"Instance {instance_id!r} not found in {path}")


def _load_instance_records(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("[") or raw.startswith("{"):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            if raw.startswith("["):
                raise SystemExit(f"Expected valid JSON list in {path}")
        else:
            return _records_from_json(parsed, path)
    records: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            records.append(dict(json.loads(line)))
    return records


def _records_from_json(parsed: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(parsed, list):
        return [dict(item) for item in parsed]
    if isinstance(parsed, dict):
        if "instances" in parsed:
            instances = parsed["instances"]
            if not isinstance(instances, list):
                raise SystemExit(f"Expected instances to be a JSON list in {path}")
            return [dict(item) for item in instances]
        return [dict(parsed)]
    raise SystemExit(f"Expected JSON object, JSON list, or JSONL records in {path}")


# --------------------------------------------------------------------------- #
# Provider environment + dotenv
# --------------------------------------------------------------------------- #
def _container_environment(provider: str) -> dict[str, str]:
    env: dict[str, str] = {}
    if provider != "openai":
        return env
    for name in OPENAI_PASSTHROUGH_ENVS:
        value = os.environ.get(name)
        if value:
            env[name] = value
    for name in ("NO_PROXY", "no_proxy"):
        value = os.environ.get(name)
        if value:
            env[name] = value
    missing = [name for name in (OPENAI_MODEL_ENV, OPENAI_AUTH_ENV) if name not in env]
    if missing:
        raise SystemExit(
            "Missing required env vars for --provider openai: " + ", ".join(missing)
        )
    return env


def resolve_api_kind(value: str | None) -> str:
    """Return the requested adapter API kind, defaulting through API_KIND."""

    api_kind = (value or os.environ.get(API_KIND_ENV) or "openai-chat").strip()
    if api_kind not in API_KIND_CHOICES:
        raise SystemExit(
            f"Unsupported API_KIND {api_kind!r}; expected one of: "
            + ", ".join(API_KIND_CHOICES)
        )
    return api_kind


# --------------------------------------------------------------------------- #
# Prediction shaping (official harness input record)
# --------------------------------------------------------------------------- #
def prediction_record(
    instance_id: str,
    model_name: str,
    patch: str,
    *,
    dataset_name: str = DEFAULT_DATASET,
) -> dict[str, str]:
    if is_swebench_pro(dataset_name=dataset_name, instance_id=instance_id):
        return {
            "instance_id": instance_id,
            "prefix": model_name,
            "patch": patch,
        }
    return {
        "instance_id": instance_id,
        "model_name_or_path": model_name,
        "model_patch": patch,
    }


# --------------------------------------------------------------------------- #
# Offline wheelhouse (provider wheels + the current project wheel)
# --------------------------------------------------------------------------- #
def prepare_wheelhouse_for_run(
    wheelhouse: Path | None,
    *,
    prepare_all: bool,
    extras: tuple[str, ...] = (),
) -> None:
    """Prepare mounted wheels, keeping local project code fresh for each run."""

    if wheelhouse is None:
        return
    if prepare_all:
        if extras:
            prepare_wheelhouse(wheelhouse, extras=extras)
        else:
            prepare_wheelhouse(wheelhouse)
    else:
        prepare_project_wheel(wheelhouse)


def prepare_project_wheel(
    path: Path,
    *,
    runner: Callable[[list[str]], None] | None = None,
) -> None:
    """Build the current repo wheel so containers do not use stale wheelhouse cache."""

    run = runner or _run_checked
    path.mkdir(parents=True, exist_ok=True)
    run(["uv", "build", "--wheel", "--out-dir", str(path)])


def prepare_wheelhouse(
    path: Path,
    *,
    runner: Callable[[list[str]], None] | None = None,
    extras: tuple[str, ...] = (),
) -> None:
    """Download provider wheels for the container's CPython 3.11 runtime."""

    run = runner or _run_checked
    prepare_project_wheel(path, runner=run)

    uv = shutil.which("uv")
    if uv is not None:
        requirements_file = _export_locked_requirements(
            uv, path, run=run, extras=extras
        )
        requirement_args = ["-r", str(requirements_file)]
    else:
        requirement_args = _project_runtime_dependencies()

    platforms = (
        ("manylinux2014_x86_64",)
        if extras
        else ("manylinux2014_x86_64", "musllinux_1_1_x86_64")
    )
    for platform in platforms:
        pip_args = [
            "download",
            "--only-binary=:all:",
            "--dest",
            str(path),
            "--python-version",
            "311",
            "--implementation",
            "cp",
            "--abi",
            "cp311",
            "--platform",
            platform,
            *requirement_args,
        ]
        command = (
            [uv, "run", "--with", "pip", "python", "-m", "pip", *pip_args]
            if uv is not None
            else [sys.executable, "-m", "pip", *pip_args]
        )
        run(command)

    _provision_uv_python(path, uv=uv, run=run)


_CONTAINER_CPYTHON = "cpython-3.11-linux-x86_64-gnu"


def _provision_uv_python(
    path: Path,
    *,
    uv: str | None,
    run: Callable[[list[str]], None],
) -> None:
    """Pre-install the container's CPython 3.11 into ``<path>/uv-python`` (offline).

    Every eval container needs CPython 3.11 (the wheels' target), but most images
    ship an older Python — so the bootstrap would otherwise have uv download a
    ~29MB standalone build *inside each container*, over the container network.
    On slow/locked-down networks that download is the dominant cost and, when it
    fails, silently degrades to the image's <3.10 Python (the SWE-bench Pro
    failures we saw). Installing it once here, into a directory mounted alongside
    the wheelhouse, lets every container resolve ``uv venv --python 3.11`` with
    ``UV_PYTHON_INSTALL_DIR`` offline.

    The interpreter is requested for the FIXED container target — Linux x86_64
    glibc — regardless of the host's own platform, exactly like the cp311
    ``manylinux2014_x86_64`` wheels above (``uv`` cross-downloads it). So a
    wheelhouse prepared on macOS still carries the Linux 3.11 the containers run.
    musl images don't use it; the bootstrap's musl branch uses their own python3.
    """

    if uv is None:
        return
    run(
        [
            uv,
            "python",
            "install",
            "--install-dir",
            str(path / "uv-python"),
            _CONTAINER_CPYTHON,
        ]
    )


def _export_locked_requirements(
    uv: str,
    destination: Path,
    *,
    run: Callable[[list[str]], None],
    extras: tuple[str, ...] = (),
) -> Path:
    """Export the project's locked runtime closure from ``uv.lock``."""

    requirements = destination / "requirements.lock.txt"
    extra_args = [arg for extra in extras for arg in ("--extra", extra)]
    run(
        [
            uv,
            "export",
            "--frozen",
            "--no-dev",
            "--no-emit-project",
            *extra_args,
            "--no-hashes",
            "--no-annotate",
            "--no-header",
            "--format",
            "requirements.txt",
            "--output-file",
            str(requirements),
        ]
    )
    return requirements


def _project_runtime_dependencies() -> list[str]:
    """Runtime dependencies the container must install, read from ``pyproject.toml``."""

    pyproject = ROOT / "pyproject.toml"
    if pyproject.exists():
        try:
            import tomllib
        except ModuleNotFoundError:
            tomllib = None  # type: ignore[assignment]
        if tomllib is not None:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            dependencies = data.get("project", {}).get("dependencies", [])
            if dependencies:
                return [str(dependency) for dependency in dependencies]

    from importlib.metadata import requires

    return [
        requirement
        for requirement in (requires("simple-agent-lab") or [])
        if "extra =="
        not in (requirement.split(";", 1)[1] if ";" in requirement else "")
    ]


def _run_checked(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _get_container(client: Any, name: str) -> Any | None:
    matches = client.containers.list(all=True, filters={"name": name})
    return next((container for container in matches if container.name == name), None)
