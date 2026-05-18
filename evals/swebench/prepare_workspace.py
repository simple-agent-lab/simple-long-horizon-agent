"""Prepare a SWE-bench worktree for an agent run.

Run from the repo root:

    PYTHONPATH=src python3 evals/swebench/prepare_workspace.py \
      --instance-json path/to/instances.jsonl \
      --instance-id sympy__sympy-20590

This is suite harness code, not core runtime code. It prepares the repository
workspace the agent may inspect and edit, and writes an agent-visible task file
without gold patch or test patch fields.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import shutil
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE_ROOT = ROOT / "evals/out/swebench_workspaces"
PRIVATE_INSTANCE_FIELDS = {
    "patch",
    "test_patch",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "fail_to_pass",
    "pass_to_pass",
}


def load_instance(path: str | None, instance_id: str | None) -> dict[str, Any]:
    if path is None:
        return {
            "instance_id": instance_id or "sympy__sympy-20590",
            "repo": "sympy/sympy",
            "problem_statement": (
                "Smoke setup placeholder. Provide --instance-json for a real "
                "SWE-bench instance."
            ),
        }

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
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    data = json.loads(text)
    if isinstance(data, list):
        return [dict(item) for item in data]
    if isinstance(data, dict):
        if "instances" in data and isinstance(data["instances"], list):
            return [dict(item) for item in data["instances"]]
        return [dict(data)]
    raise SystemExit(f"Unsupported instance record shape in {path}")


def repo_url_for_instance(instance: dict[str, Any], explicit: str | None) -> str:
    if explicit:
        return explicit
    repo = str(instance.get("repo") or "").strip()
    if not repo:
        raise SystemExit("Instance has no repo. Pass --repo-url explicitly.")
    if repo.startswith(("https://", "ssh://", "git@")):
        return repo
    return f"https://github.com/{repo}.git"


def workspace_dir(root: Path, instance_id: str) -> Path:
    safe = instance_id.replace("/", "__").replace(":", "_")
    return root / safe


def run_git(args: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)


def prepare_repo(
    *,
    repo_dir: Path,
    repo_url: str,
    base_commit: str,
    force: bool,
    skip_clone: bool,
) -> None:
    if force and repo_dir.exists():
        shutil.rmtree(repo_dir)
    if skip_clone:
        repo_dir.mkdir(parents=True, exist_ok=True)
        if not (repo_dir / ".git").exists():
            run_git(["init", "-q"], cwd=repo_dir)
        return

    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        run_git(["clone", "--no-tags", repo_url, str(repo_dir)])

    if base_commit:
        run_git(["fetch", "--depth", "1", "origin", base_commit], cwd=repo_dir)
        run_git(["checkout", "--force", base_commit], cwd=repo_dir)
    else:
        run_git(["checkout", "--force"], cwd=repo_dir)
    run_git(["clean", "-fdx"], cwd=repo_dir)


def agent_task(instance: dict[str, Any]) -> dict[str, Any]:
    return {
        "suite": "swebench",
        "instance_id": str(instance.get("instance_id") or ""),
        "repo": instance.get("repo"),
        "base_commit": instance.get("base_commit"),
        "problem_statement": (
            instance.get("problem_statement")
            or instance.get("problem")
            or instance.get("description")
            or ""
        ),
    }


def sanitized_instance(instance: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in instance.items()
        if str(key) not in PRIVATE_INSTANCE_FIELDS
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-json", help="SWE-bench instance JSON or JSONL.")
    parser.add_argument("--instance-id", default="sympy__sympy-20590")
    parser.add_argument("--repo-url", help="Override clone URL.")
    parser.add_argument(
        "--workspace-root",
        default=str(DEFAULT_WORKSPACE_ROOT),
        help="Directory where per-instance workspaces are created.",
    )
    parser.add_argument("--force", action="store_true", help="Recreate this workspace.")
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Create metadata only. Useful for local smoke tests without network.",
    )
    args = parser.parse_args()

    instance = load_instance(args.instance_json, args.instance_id)
    instance_id = str(instance.get("instance_id") or args.instance_id)
    root = Path(args.workspace_root).resolve()
    workdir = workspace_dir(root, instance_id)
    repo_dir = workdir / "repo"
    base_commit = str(instance.get("base_commit") or "")

    prepare_repo(
        repo_dir=repo_dir,
        repo_url=repo_url_for_instance(instance, args.repo_url),
        base_commit=base_commit,
        force=args.force,
        skip_clone=args.skip_clone,
    )
    write_json(workdir / "task.json", agent_task(instance))
    write_json(workdir / "instance.sanitized.json", sanitized_instance(instance))

    print(f"workspace={workdir}")
    print(f"repo={repo_dir}")
    print(f"task={workdir / 'task.json'}")
    if args.skip_clone:
        print("clone=skipped")
    elif base_commit:
        print(f"base_commit={base_commit}")


if __name__ == "__main__":
    main()
