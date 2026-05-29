"""Host-side orchestration for running Simple Agent Lab inside SWE-bench containers.

The host starts a SWE-bench instance container with a wheelhouse and run
directory mounted, copies in the eval runner, installs the `simple-agent-lab`
wheel, then runs the agent loop inside that container using the normal local
bash tool against `/testbed``.

Live traces: host ``evals/out/.../<run>/out/`` is bind-mounted at
``/agent/run``; the in-container runner writes
``/agent/run/out/trajectory.jsonl`` for the host trace viewer. See
``docs/agent-native/docker-live-trace.md``.
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import os
import shutil
import shlex
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_DATASET = "princeton-nlp/SWE-bench_Verified"
DEFAULT_SPLIT = "test"
DEFAULT_RUN_MOUNT = "/agent/run"
DEFAULT_WHEELHOUSE_MOUNT = "/agent/wheelhouse"
DEFAULT_RUNNER_PATH = "/agent/evals/swebench/in_container_runner.py"
DEFAULT_WORKDIR = "/testbed"
DEFAULT_PRO_WORKDIR = "/app"
DEFAULT_PRO_DOCKERHUB_USERNAME = "jefzda"
DEFAULT_RUN_ROOT = ROOT / "evals/out/swebench"
DEFAULT_WHEELHOUSE = ROOT / "evals/out/swebench/wheelhouse/cp311-manylinux"
DEFAULT_PRO_RUN_ROOT = ROOT / "evals/out/swebench_pro"
DEFAULT_PRO_WHEELHOUSE = ROOT / "evals/out/swebench_pro/wheelhouse/cp311-manylinux"
DEFAULT_UV_BINARY = shutil.which("uv") or ""
UV_CONTAINER_PATH = "/tmp/uv"
DEFAULT_LOCAL_RUNNER = ROOT / "evals/swebench/in_container_runner.py"
OPENAI_MODEL_ENV = "OPENAI_MODEL"
OPENAI_AUTH_ENV = "OPENAI_AUTH_TOKEN"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
OPENAI_SESSION_ID_ENV = "OPENAI_SESSION_ID"
OPENAI_LOG_ID_ENV = "OPENAI_LOG_ID"
API_KIND_ENV = "API_KIND"
API_KIND_CHOICES = ("openai-chat", "openai-responses")
AGENT_FLAVOR_CHOICES = ("bash", "bash_task")
DEFAULT_AGENT_FLAVOR = "bash"
OPENAI_PASSTHROUGH_ENVS = (
    OPENAI_MODEL_ENV,
    OPENAI_AUTH_ENV,
    OPENAI_BASE_URL_ENV,
    OPENAI_SESSION_ID_ENV,
    OPENAI_LOG_ID_ENV,
    API_KIND_ENV,
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


@dataclass(frozen=True)
class RunPaths:
    root: Path
    input_dir: Path
    output_dir: Path
    instance_json: Path
    trajectory_jsonl: Path
    prediction_jsonl: Path


def container_name(instance_id: str, run_id: str) -> str:
    """Return a stable Docker-safe name for the agent container."""

    safe_run_id = _safe_docker_part(run_id)
    safe_instance_id = _safe_docker_part(instance_id)
    return f"sweb.agent.{safe_instance_id}.{safe_run_id}"


def prepare_run_directory(
    *,
    run_root: Path,
    instance: dict[str, Any],
    run_id: str,
) -> RunPaths:
    """Create input/output dirs and write the model-visible instance record."""

    instance_id = str(instance["instance_id"])
    run_root = run_root.resolve()
    root = run_root / _safe_docker_part(run_id) / _safe_docker_part(instance_id)
    input_dir = root / "input"
    output_dir = root / "out"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    instance_json = input_dir / "instance.json"
    instance_json.write_text(
        json.dumps(sanitized_instance(instance), ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return RunPaths(
        root=root,
        input_dir=input_dir,
        output_dir=output_dir,
        instance_json=instance_json,
        trajectory_jsonl=output_dir / "trajectory.jsonl",
        prediction_jsonl=output_dir / "prediction.jsonl",
    )


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


def sanitized_instance(instance: dict[str, Any]) -> dict[str, Any]:
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


def build_runner_command(
    *,
    run_mount: str,
    instance_id: str,
    dataset_name: str,
    split: str,
    model_name: str,
    provider: str,
    max_turns: int,
    api_kind: str = "openai-chat",
    workdir: str = DEFAULT_WORKDIR,
    install: bool = True,
    wheelhouse_mount: str | None = None,
    runner_path: str = DEFAULT_RUNNER_PATH,
    agent_flavor: str = DEFAULT_AGENT_FLAVOR,
) -> str:
    """Build the shell command that runs as the container's main process.

    Python setup strategy:
    - Prefer a host-copied uv binary, then uv from the container PATH.
    - Use uv to create an isolated Python 3.11 environment when available.
    - Fall back to system Python venv only when uv is unavailable.
    """

    parts = [
        "set -eu",
        "(set -o pipefail) 2>/dev/null && set -o pipefail || true",
        'UV_BIN=""',
        f"if [ -f {shlex.quote(UV_CONTAINER_PATH)} ]; then",
        f"  chmod +x {shlex.quote(UV_CONTAINER_PATH)} 2>/dev/null || true",
        f"  if {shlex.quote(UV_CONTAINER_PATH)} --version >/dev/null 2>&1; then",
        f"    UV_BIN={shlex.quote(UV_CONTAINER_PATH)}",
        "  fi",
        "fi",
        'if [ -z "$UV_BIN" ] && command -v uv >/dev/null 2>&1; then',
        '  PATH_UV="$(command -v uv)"',
        '  if "$PATH_UV" --version >/dev/null 2>&1; then',
        '    UV_BIN="$PATH_UV"',
        "  fi",
        "fi",
        "# Detect musl vs glibc",
        "_IS_MUSL=0",
        "if ldd /bin/sh 2>/dev/null | grep -q musl; then _IS_MUSL=1; fi",
        "",
        'if [ -n "$UV_BIN" ]; then',
        '  "$UV_BIN" venv --python 3.11 /tmp/agent-venv || '
        '"$UV_BIN" venv --python python3 /tmp/agent-venv',
        "  AGENT_PYTHON=/tmp/agent-venv/bin/python",
        'elif [ "$_IS_MUSL" = 1 ]; then',
        "  # Alpine: use system Python with venv (never touches system site-packages)",
        "  if ! command -v python3 >/dev/null 2>&1; then",
        '    echo "ERROR: Alpine container has no Python" >&2; exit 1',
        "  fi",
        "  python3 -m venv /tmp/agent-venv",
        "  AGENT_PYTHON=/tmp/agent-venv/bin/python3",
        "else",
        "  # glibc fallback: use system Python when uv is unavailable",
        "  if command -v python3 >/dev/null 2>&1; then",
        "    if python3 -m venv /tmp/agent-venv >/dev/null 2>&1; then",
        "      AGENT_PYTHON=/tmp/agent-venv/bin/python3",
        "    else",
        "      AGENT_PYTHON=python3",
        "    fi",
        "  else",
        '    echo "ERROR: container has no uv and no python3" >&2; exit 1',
        "  fi",
        "fi",
        '"$AGENT_PYTHON" --version',
    ]
    if install:
        if wheelhouse_mount:
            parts.append(
                'if [ -n "$UV_BIN" ]; then '
                '"$UV_BIN" pip install --python "$AGENT_PYTHON" --no-index --find-links '
                + shlex.quote(wheelhouse_mount)
                + " "
                + shlex.quote("simple-agent-lab")
                + '; else "$AGENT_PYTHON" -m pip install --no-index --find-links '
                + shlex.quote(wheelhouse_mount)
                + " "
                + shlex.quote("simple-agent-lab")
                + "; fi"
            )
        else:
            parts.append(
                'if [ -n "$UV_BIN" ]; then '
                '"$UV_BIN" pip install --python "$AGENT_PYTHON" '
                + shlex.quote("simple-agent-lab")
                + '; else "$AGENT_PYTHON" -m pip install '
                + shlex.quote("simple-agent-lab")
                + "; fi"
            )
    parts.append(
        '"$AGENT_PYTHON" '
        + shlex.quote(runner_path)
        + " --instance-json "
        + shlex.quote(f"{run_mount}/input/instance.json")
        + " --instance-id "
        + shlex.quote(instance_id)
        + " --dataset-name "
        + shlex.quote(dataset_name)
        + " --split "
        + shlex.quote(split)
        + " --model-name "
        + shlex.quote(model_name)
        + " --provider "
        + shlex.quote(provider)
        + " --api-kind "
        + shlex.quote(api_kind)
        + " --workdir "
        + shlex.quote(workdir)
        + " --max-turns "
        + shlex.quote(str(max_turns))
        + " --agent-flavor "
        + shlex.quote(agent_flavor)
        + " --traces "
        + shlex.quote(f"{run_mount}/out/trajectory.jsonl")
        + " --predictions "
        + shlex.quote(f"{run_mount}/out/prediction.jsonl")
    )
    return "\n".join(parts)


def run_containerized_agent(args: argparse.Namespace) -> RunPaths:
    """Start one SWE-bench container and run the agent as its main process."""

    import docker
    import docker.errors

    instance = load_instance(args.instance_json, args.instance_id)
    pro_instance = is_swebench_pro_instance(instance, dataset_name=args.dataset_name)
    run_root = Path(args.run_root)
    if pro_instance and str(args.run_root) == str(DEFAULT_RUN_ROOT):
        run_root = DEFAULT_PRO_RUN_ROOT
    paths = prepare_run_directory(
        run_root=run_root,
        instance=instance,
        run_id=args.run_id,
    )
    workdir = resolve_workdir(args.workdir, instance, dataset_name=args.dataset_name)

    if args.provider == "openai":
        load_dotenv(args.dotenv)
    environment = _container_environment(args.provider)
    wheelhouse_path = args.wheelhouse
    if pro_instance and wheelhouse_path == str(DEFAULT_WHEELHOUSE):
        wheelhouse_path = str(DEFAULT_PRO_WHEELHOUSE)
    wheelhouse = Path(wheelhouse_path).resolve() if wheelhouse_path else None
    prepare_wheelhouse_for_run(wheelhouse, prepare_all=args.prepare_wheelhouse)

    image_key = docker_image_for_instance(
        instance,
        dataset_name=args.dataset_name,
        namespace=args.namespace,
        instance_image_tag=args.instance_image_tag,
        env_image_tag=args.env_image_tag,
        dockerhub_username=args.dockerhub_username,
    )
    platform: str | None = None
    run_args: dict[str, Any] = {}
    if pro_instance:
        platform = args.docker_platform or None
    else:
        spec = _make_swebench_test_spec(
            instance,
            namespace=args.namespace,
            instance_image_tag=args.instance_image_tag,
            env_image_tag=args.env_image_tag,
        )
        platform = spec.platform
        run_args = spec.docker_specs.get("run_args", {})

    client = docker.from_env()
    _ensure_image_available(
        client.images,
        image_key=image_key,
        platform=platform or None,
        pull_policy=args.pull,
        image_not_found_error=docker.errors.ImageNotFound,
        docker_exception=docker.errors.DockerException,
    )

    name = container_name(str(instance["instance_id"]), args.run_id)
    if args.force and (existing := _get_container(client, name)) is not None:
        existing.remove(force=True)
    elif _get_container(client, name) is not None:
        raise SystemExit(
            f"Container {name!r} already exists. Pass --force to remove it first."
        )

    command = build_runner_command(
        run_mount=args.run_mount,
        instance_id=str(instance["instance_id"]),
        dataset_name=args.dataset_name,
        split=args.split,
        model_name=args.model_name,
        provider=args.provider,
        max_turns=args.max_turns,
        api_kind=resolve_api_kind(args.api_kind),
        workdir=workdir,
        install=not args.skip_install,
        wheelhouse_mount=args.wheelhouse_mount if wheelhouse is not None else None,
        runner_path=args.container_runner_path,
        agent_flavor=args.agent_flavor,
    )
    volumes = {
        str(paths.root): {"bind": args.run_mount, "mode": "rw"},
    }
    if wheelhouse is not None:
        volumes[str(wheelhouse)] = {"bind": args.wheelhouse_mount, "mode": "ro"}
    create_options = container_create_options(args.network_mode)
    create_kwargs: dict[str, Any] = {
        "image": image_key,
        "name": name,
        "user": "root",
        "detach": True,
        # Pro images often define /bin/bash as ENTRYPOINT; clear it so the
        # command list is executed as the container main process.
        **container_entrypoint_override(instance, dataset_name=args.dataset_name),
        "command": docker_run_command(
            command,
            instance,
            dataset_name=args.dataset_name,
        ),
        "cap_add": run_args.get("cap_add", []),
        "environment": environment,
        "volumes": volumes,
        **create_options,
    }
    if platform:
        create_kwargs["platform"] = platform
    container = client.containers.create(
        **create_kwargs,
    )
    try:
        copy_file_to_container(
            container,
            source_path=Path(args.runner_path),
            target_path=args.container_runner_path,
        )
        copy_runner_support_files(
            container,
            runner_path=Path(args.runner_path),
            container_runner_path=args.container_runner_path,
        )
        uv_path = Path(args.uv_binary) if args.uv_binary else None
        if uv_path is not None and uv_path.exists():
            copy_file_to_container(
                container,
                source_path=uv_path,
                target_path=UV_CONTAINER_PATH,
            )
        container.start()
        result = container.wait()
        logs = container.logs(stdout=True, stderr=True).decode(errors="replace")
        if logs:
            print(logs, end="" if logs.endswith("\n") else "\n")
        status_code = int(result.get("StatusCode", 1))
        if status_code != 0:
            raise SystemExit(f"Containerized agent exited with {status_code}")
    finally:
        if not args.keep_container:
            container.remove(force=True)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Simple Agent Lab inside a SWE-bench container."
    )
    parser.add_argument("--instance-json", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--model-name", default="simple-agent-lab-containerized")
    parser.add_argument("--provider", choices=["fake", "openai"], default="openai")
    parser.add_argument(
        "--api-kind",
        choices=API_KIND_CHOICES,
        default=None,
        help=(
            "Adapter API kind to use when --provider openai. Defaults to "
            "API_KIND from the environment, then openai-chat."
        ),
    )
    parser.add_argument("--dotenv", default=str(ROOT / ".env"))
    parser.add_argument("--max-turns", type=int, default=75)
    parser.add_argument(
        "--agent-flavor",
        choices=AGENT_FLAVOR_CHOICES,
        default=DEFAULT_AGENT_FLAVOR,
        help=(
            "Agent preset: 'bash' (parent has only the bash tool, current "
            "baseline) or 'bash_task' (parent has bash + task with an "
            "explorer sub-agent for context-isolating heavy reads)."
        ),
    )
    parser.add_argument("--run-id", default="containerized-agent")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--run-mount", default=DEFAULT_RUN_MOUNT)
    parser.add_argument("--wheelhouse", default=str(DEFAULT_WHEELHOUSE))
    parser.add_argument("--wheelhouse-mount", default=DEFAULT_WHEELHOUSE_MOUNT)
    parser.add_argument("--prepare-wheelhouse", action="store_true")
    parser.add_argument("--runner-path", default=str(DEFAULT_LOCAL_RUNNER))
    parser.add_argument("--container-runner-path", default=DEFAULT_RUNNER_PATH)
    parser.add_argument("--uv-binary", default=DEFAULT_UV_BINARY)
    parser.add_argument(
        "--workdir",
        default="",
        help="Container repository workdir. Defaults to /testbed, or /app for SWE-bench Pro.",
    )
    parser.add_argument("--namespace", default="swebench")
    parser.add_argument("--instance-image-tag", default="latest")
    parser.add_argument("--env-image-tag", default="latest")
    parser.add_argument(
        "--dockerhub-username",
        default=DEFAULT_PRO_DOCKERHUB_USERNAME,
        help="Docker Hub username for SWE-bench Pro images.",
    )
    parser.add_argument(
        "--docker-platform",
        default="",
        help="Optional Docker platform override for direct Docker Hub images.",
    )
    parser.add_argument(
        "--network-mode",
        default="",
        help="Optional Docker network mode for the agent container, e.g. 'host'.",
    )
    parser.add_argument(
        "--pull",
        choices=["always", "missing", "never"],
        default="missing",
        help="Docker image pull policy: 'always' pulls every time, 'missing' pulls only if not local, 'never' never pulls (default: missing).",
    )
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--keep-container", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    paths = run_containerized_agent(args)
    print(f"trajectory={paths.trajectory_jsonl}")
    print(f"prediction={paths.prediction_jsonl}")


def _safe_docker_part(value: str) -> str:
    return "".join(char if char.isalnum() or char in "_.-" else "_" for char in value)


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


def container_create_options(network_mode: str | None) -> dict[str, str]:
    if not network_mode:
        return {}
    return {"network_mode": network_mode}


def copy_file_to_container(
    container: Any, *, source_path: Path, target_path: str
) -> None:
    """Copy one host file into a container using Docker's tar archive API."""

    data = source_path.read_bytes()
    archive_path = target_path.lstrip("/")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        parent = Path(archive_path).parent
        parts = parent.parts
        current = Path()
        for part in parts:
            current = current / part
            info = tarfile.TarInfo(current.as_posix())
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            archive.addfile(info)
        info = tarfile.TarInfo(archive_path)
        info.size = len(data)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(data))
    buffer.seek(0)
    container.put_archive("/", buffer.getvalue())


def copy_runner_support_files(
    container: Any,
    *,
    runner_path: Path,
    container_runner_path: str,
) -> None:
    """Copy small eval-side modules imported by the in-container runner."""

    target_dir = str(Path(container_runner_path).parent).replace("\\", "/")
    for filename in ("patch_extract.py",):
        source = runner_path.with_name(filename)
        if source.exists():
            copy_file_to_container(
                container,
                source_path=source,
                target_path=f"{target_dir}/{filename}",
            )


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


def load_dotenv(path: str | Path) -> None:
    """Load simple KEY=VALUE dotenv lines without overriding the environment."""

    dotenv = Path(path)
    if not dotenv.exists():
        return
    for raw_line in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, separator, value = line.partition("=")
        if separator and key.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip().strip("'\"")


def resolve_api_kind(value: str | None) -> str:
    """Return the requested adapter API kind, defaulting through API_KIND."""

    api_kind = (value or os.environ.get(API_KIND_ENV) or "openai-chat").strip()
    if api_kind not in API_KIND_CHOICES:
        raise SystemExit(
            f"Unsupported API_KIND {api_kind!r}; expected one of: "
            + ", ".join(API_KIND_CHOICES)
        )
    return api_kind


def _ensure_image_available(
    images: Any,
    *,
    image_key: str,
    platform: str | None,
    pull_policy: str,
    image_not_found_error: type[BaseException],
    docker_exception: type[BaseException],
) -> None:
    """Apply Docker image pull policy before creating the eval container."""

    try:
        images.get(image_key)
    except image_not_found_error:
        if pull_policy == "never":
            raise SystemExit(
                f"Missing SWE-bench image {image_key} and --pull=never was set."
            ) from None
        _pull_image(
            images,
            image_key=image_key,
            platform=platform,
            docker_exception=docker_exception,
        )
        return

    if pull_policy == "always":
        _pull_image(
            images,
            image_key=image_key,
            platform=platform,
            docker_exception=docker_exception,
        )


def _pull_image(
    images: Any,
    *,
    image_key: str,
    platform: str | None,
    docker_exception: type[BaseException],
) -> None:
    try:
        images.pull(image_key, platform=platform)
    except docker_exception as exc:
        raise SystemExit(f"Failed to pull image {image_key}: {exc}") from exc


def prepare_wheelhouse(
    path: Path,
    *,
    runner: Callable[[list[str]], None] | None = None,
) -> None:
    """Download provider wheels for the container's CPython 3.11 runtime."""

    run = runner or _run_checked
    prepare_project_wheel(path, runner=run)
    for platform in ("manylinux2014_x86_64", "musllinux_1_1_x86_64"):
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
            "anthropic>=0.39.0",
            "openai>=1.50.0",
        ]
        uv = shutil.which("uv")
        command = (
            [uv, "run", "--with", "pip", "python", "-m", "pip", *pip_args]
            if uv is not None
            else [sys.executable, "-m", "pip", *pip_args]
        )
        run(command)


def prepare_wheelhouse_for_run(wheelhouse: Path | None, *, prepare_all: bool) -> None:
    """Prepare mounted wheels, keeping local project code fresh for each run."""

    if wheelhouse is None:
        return
    if prepare_all:
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


def _run_checked(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _get_container(client: Any, name: str) -> Any | None:
    matches = client.containers.list(all=True, filters={"name": name})
    return next((container for container in matches if container.name == name), None)


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


if __name__ == "__main__":
    main()
