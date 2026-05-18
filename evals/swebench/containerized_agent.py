"""Host-side orchestration for running Simple Agent Lab inside SWE-bench containers.

The host starts a SWE-bench instance container with a wheelhouse and run
directory mounted, copies in the eval runner, installs the `simple-agent-lab`
wheel, then runs the agent loop inside that container using the normal local
bash tool against `/testbed`.
"""

from __future__ import annotations

import argparse
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
DEFAULT_RUN_ROOT = ROOT / "evals/out/swebench_container_runs"
DEFAULT_WHEELHOUSE = ROOT / "evals/out/wheelhouse/cp311-manylinux"
DEFAULT_LOCAL_RUNNER = ROOT / "evals/swebench/in_container_runner.py"
OPENAI_MODEL_ENV = "OPENAI_MODEL"
OPENAI_AUTH_ENV = "OPENAI_AUTH_TOKEN"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
OPENAI_PASSTHROUGH_ENVS = (
    OPENAI_MODEL_ENV,
    OPENAI_AUTH_ENV,
    OPENAI_BASE_URL_ENV,
)
PRIVATE_INSTANCE_FIELDS = {
    "patch",
    "test_patch",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "fail_to_pass",
    "pass_to_pass",
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


def build_runner_command(
    *,
    run_mount: str,
    instance_id: str,
    dataset_name: str,
    split: str,
    model_name: str,
    provider: str,
    max_turns: int,
    workdir: str = DEFAULT_WORKDIR,
    install: bool = True,
    wheelhouse_mount: str | None = None,
    runner_path: str = DEFAULT_RUNNER_PATH,
) -> str:
    """Build the shell command that runs as the container's main process."""

    parts = [
        "set -euo pipefail",
        "AGENT_PYTHON=/opt/miniconda3/bin/python3",
        'if [ ! -x "$AGENT_PYTHON" ]; then AGENT_PYTHON=python3; fi',
        '"$AGENT_PYTHON" --version',
    ]
    if install:
        if wheelhouse_mount:
            parts.append(
                '"$AGENT_PYTHON" -m pip install --no-index --find-links '
                + shlex.quote(wheelhouse_mount)
                + " "
                + shlex.quote("simple-agent-lab[openai]")
            )
        else:
            parts.append('"$AGENT_PYTHON" -m pip install simple-agent-lab[openai]')
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
        + " --workdir "
        + shlex.quote(workdir)
        + " --max-turns "
        + shlex.quote(str(max_turns))
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
    from swebench.harness.test_spec.test_spec import make_test_spec

    instance = load_instance(args.instance_json, args.instance_id)
    paths = prepare_run_directory(
        run_root=Path(args.run_root),
        instance=instance,
        run_id=args.run_id,
    )

    if args.provider == "openai":
        load_dotenv(args.dotenv)
    environment = _container_environment(args.provider)
    wheelhouse = Path(args.wheelhouse) if args.wheelhouse else None
    if args.prepare_wheelhouse and wheelhouse is not None:
        prepare_wheelhouse(wheelhouse)

    spec = make_test_spec(
        instance,
        namespace=args.namespace,
        instance_image_tag=args.instance_image_tag,
        env_image_tag=args.env_image_tag,
    )
    client = docker.from_env()
    try:
        client.images.get(spec.instance_image_key)
    except docker.errors.ImageNotFound as exc:
        raise SystemExit(
            f"Missing SWE-bench image {spec.instance_image_key}. "
            "Build or pull the image before running the containerized agent."
        ) from exc

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
        workdir=args.workdir,
        install=not args.skip_install,
        wheelhouse_mount=args.wheelhouse_mount if wheelhouse is not None else None,
        runner_path=args.container_runner_path,
    )
    run_args = spec.docker_specs.get("run_args", {})
    volumes = {
        str(paths.root): {"bind": args.run_mount, "mode": "rw"},
    }
    if wheelhouse is not None:
        volumes[str(wheelhouse)] = {"bind": args.wheelhouse_mount, "mode": "ro"}
    create_options = container_create_options(args.network_mode)
    container = client.containers.create(
        image=spec.instance_image_key,
        name=name,
        user="root",
        detach=True,
        command=["bash", "-lc", command],
        platform=spec.platform,
        cap_add=run_args.get("cap_add", []),
        environment=environment,
        volumes=volumes,
        **create_options,
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
    parser.add_argument("--dotenv", default=str(ROOT / ".env"))
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--run-id", default="containerized-agent")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--run-mount", default=DEFAULT_RUN_MOUNT)
    parser.add_argument("--wheelhouse", default=str(DEFAULT_WHEELHOUSE))
    parser.add_argument("--wheelhouse-mount", default=DEFAULT_WHEELHOUSE_MOUNT)
    parser.add_argument("--prepare-wheelhouse", action="store_true")
    parser.add_argument("--runner-path", default=str(DEFAULT_LOCAL_RUNNER))
    parser.add_argument("--container-runner-path", default=DEFAULT_RUNNER_PATH)
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR)
    parser.add_argument("--namespace", default="swebench")
    parser.add_argument("--instance-image-tag", default="latest")
    parser.add_argument("--env-image-tag", default="latest")
    parser.add_argument(
        "--network-mode",
        default="",
        help="Optional Docker network mode for the agent container, e.g. 'host'.",
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
    if raw.startswith("["):
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise SystemExit(f"Expected a JSON list in {path}")
        return [dict(item) for item in parsed]
    records: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            records.append(dict(json.loads(line)))
    return records


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


def prepare_wheelhouse(
    path: Path,
    *,
    runner: Callable[[list[str]], None] | None = None,
) -> None:
    """Download provider wheels for the container's CPython 3.11 runtime."""

    run = runner or _run_checked
    path.mkdir(parents=True, exist_ok=True)
    build_command = ["uv", "build", "--wheel", "--out-dir", str(path)]
    run(build_command)
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
        "manylinux2014_x86_64",
        "openai>=1.50.0",
    ]
    uv = shutil.which("uv")
    command = (
        [uv, "run", "--with", "pip", "python", "-m", "pip", *pip_args]
        if uv is not None
        else [sys.executable, "-m", "pip", *pip_args]
    )
    run(command)


def _run_checked(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _get_container(client: Any, name: str) -> Any | None:
    matches = client.containers.list(all=True, filters={"name": name})
    return next((container for container in matches if container.name == name), None)


if __name__ == "__main__":
    main()
