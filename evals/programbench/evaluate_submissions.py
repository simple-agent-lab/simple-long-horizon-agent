"""Score ProgramBench runs with the official ``programbench eval`` harness.

ProgramBench's authoritative scorer is the official ``programbench`` CLI
(compile the submission → restore ``./executable`` with a sha256 check → run each
test branch's pytest → JUnit XML → pass rate). It is not a framework seam (ADR
0020 / ADR 0022): a run writes its product to ``out/result.json`` and scoring is
this standalone follow-up CLI.

The one ProgramBench-specific shaping step is that our container half returns the
submission tarball base64-encoded inside ``result.json`` (a container half can
only return bytes through that file). This script:

1. **collects** — walks ``<run_root>/<run_id>/<id>/out/result.json``, decodes
   ``submission_tar_b64`` back into the ``<eval_dir>/<id>/submission.tar.gz``
   layout the official harness expects, then
2. **evaluates** — runs ``programbench eval <eval_dir> --image-tag task`` (needs
   Docker + the ``programbench`` package + access to the HF test blobs; if that
   dataset needs auth, set ``HF_TOKEN`` in the environment or ``.env``), and
3. **summarizes** — runs ``programbench info`` for the authoritative scores and
   writes a small machine-readable ``scores.json`` manifest beside the results.

Usage::

    uv run python evals/programbench/evaluate_submissions.py \
        --run-root evals/out/programbench --run-id my-run [--instance-ids a b] \
        [--workers 4] [--branch-workers 2] [--docker-cpus 20] [--docker-memory 60g] [--force] [--collect-only]
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from evals.programbench import harness  # noqa: E402

SUBMISSION_KEY = "submission_tar_b64"
DEFAULT_DOCKER_MEMORY = "60g"


def _configure_programbench_docker(
    *, cpus: int | None, memory: str | None
) -> list[str]:
    """Patch programbench's docker run args before starting eval containers.

    The official ``programbench eval`` CLI exposes ``--docker-cpus`` but not
    memory. Containers are created via ``programbench.constants.DOCKER_RUN_ARGS``,
    which we extend with ``--memory`` / ``--memory-swap`` here so each eval
    container matches the inference launch spec (60g, swap disabled).
    """

    import programbench.constants as pb_constants

    if cpus is not None:
        pb_constants.DOCKER_CPUS = cpus
    run_args: list[str] = []
    if memory:
        run_args.extend(["--memory", memory, "--memory-swap", memory])
    pb_constants.DOCKER_RUN_ARGS = run_args
    return run_args


# --------------------------------------------------------------------------- #
# 1. Collect: rebuild submission.tar.gz from each run's result.json
# --------------------------------------------------------------------------- #
def collect_submissions(
    *,
    run_root: Path,
    run_id: str,
    eval_dir: Path,
    instance_ids: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Decode each run's ``result.json`` into ``<eval_dir>/<id>/submission.tar.gz``.

    Returns ``(collected, missing)``: ids with a submission written, and ids
    whose ``result.json`` carried no submission (e.g. a crashed run).
    """

    run_dir = Path(run_root) / run_id
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")
    wanted = set(instance_ids) if instance_ids else None
    eval_dir = Path(eval_dir)
    collected: list[str] = []
    missing: list[str] = []
    for result_path in sorted(run_dir.glob("*/out/result.json")):
        instance_id = result_path.parent.parent.name
        if wanted is not None and instance_id not in wanted:
            continue
        result = json.loads(result_path.read_text(encoding="utf-8") or "{}")
        encoded = result.get(SUBMISSION_KEY)
        if not encoded:
            missing.append(instance_id)
            continue
        dest = eval_dir / instance_id
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "submission.tar.gz").write_bytes(base64.b64decode(encoded))
        collected.append(instance_id)
    return collected, missing


# --------------------------------------------------------------------------- #
# 2. Evaluate + 3. summarize: shell out to the official CLI
# --------------------------------------------------------------------------- #
def run_official_eval(
    eval_dir: Path,
    *,
    programbench_bin: str = "programbench",
    image_tag: str = harness.DEFAULT_SCORE_IMAGE_TAG,
    workers: int = 1,
    branch_workers: int = 1,
    docker_cpus: int | None = None,
    docker_memory: str | None = DEFAULT_DOCKER_MEMORY,
    branch_retries: int | None = None,
    force: bool = False,
    filter_spec: str = "",
    slice_spec: str = "",
    summarize_only: bool = False,
) -> int:
    """Run ``programbench eval`` on the rebuilt submissions directory."""

    run_args = _configure_programbench_docker(cpus=docker_cpus, memory=docker_memory)
    cmd_parts = [
        programbench_bin,
        "eval",
        str(eval_dir),
        "--image-tag",
        image_tag,
        "-w",
        str(workers),
        "-b",
        str(branch_workers),
    ]
    if docker_cpus is not None:
        cmd_parts += ["--docker-cpus", str(docker_cpus)]
    if docker_memory:
        cmd_parts += ["--docker-memory", docker_memory, "(via DOCKER_RUN_ARGS)"]
    if branch_retries is not None:
        cmd_parts += ["--branch-retries", str(branch_retries)]
    if force:
        cmd_parts.append("--force")
    if filter_spec:
        cmd_parts += ["--filter", filter_spec]
    if slice_spec:
        cmd_parts += ["--slice", slice_spec]
    if summarize_only:
        cmd_parts.append("--summarize-only")
    print("==> " + " ".join(cmd_parts))
    if run_args:
        print(f"    docker run args: {' '.join(run_args)}")

    from programbench.eval.eval_batch import run_eval_batch

    run_eval_batch(
        sources=[str(eval_dir)],
        force=force,
        workers=workers,
        branch_workers=branch_workers,
        docker_cpus=docker_cpus if docker_cpus is not None else 20,
        filter_spec=filter_spec,
        slice_spec=slice_spec,
        summarize_only=summarize_only,
        image_tag=image_tag,
        branch_retries=branch_retries if branch_retries is not None else 1,
    )
    return 0


def run_official_info(eval_dir: Path, *, programbench_bin: str = "programbench") -> int:
    """Print the authoritative per-instance scores via ``programbench info``."""

    cmd = [programbench_bin, "info", str(eval_dir)]
    print("==> " + " ".join(cmd))
    return subprocess.run(cmd).returncode


def collect_eval_manifest(eval_dir: Path, instance_ids: list[str]) -> dict[str, Any]:
    """Read each ``<id>/<id>.eval.json`` into a small machine-readable manifest.

    The authoritative score display is ``programbench info`` (it applies the
    tests.json ignored-branch/test filtering); this manifest is just a stable,
    parseable index of which instances produced an eval result and any
    top-level ``error_code``.
    """

    eval_dir = Path(eval_dir)
    manifest: dict[str, Any] = {}
    for instance_id in instance_ids:
        eval_json = eval_dir / instance_id / f"{instance_id}.eval.json"
        entry: dict[str, Any] = {
            "eval_json": str(eval_json),
            "evaluated": eval_json.exists(),
        }
        if eval_json.exists():
            data = json.loads(eval_json.read_text(encoding="utf-8") or "{}")
            entry["error_code"] = data.get("error_code")
            entry["solution_branch"] = data.get("solution_branch")
            entry["test_branches"] = data.get("test_branches")
            entry["warnings"] = data.get("warnings")
        manifest[instance_id] = entry
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=str(harness.DEFAULT_RUN_ROOT))
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--eval-dir",
        default=None,
        help="Where to rebuild submissions + write eval.json "
        "(default: <run_root>/<run_id>_eval).",
    )
    parser.add_argument("--instance-ids", nargs="*", default=None)
    parser.add_argument("--programbench-bin", default="programbench")
    parser.add_argument("--image-tag", default=harness.DEFAULT_SCORE_IMAGE_TAG)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--branch-workers", type=int, default=1)
    parser.add_argument("--docker-cpus", type=int, default=20)
    parser.add_argument(
        "--docker-memory",
        default=DEFAULT_DOCKER_MEMORY,
        help="Memory limit per eval container (docker --memory / --memory-swap). "
        "Passed via programbench.constants.DOCKER_RUN_ARGS; default matches inference (60g).",
    )
    parser.add_argument("--branch-retries", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--filter", dest="filter_spec", default="")
    parser.add_argument("--slice", dest="slice_spec", default="")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Skip evaluation; just read existing eval.json results.",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only rebuild submission.tar.gz files; do not run the harness.",
    )
    parser.add_argument(
        "--no-info",
        action="store_true",
        help="Skip the `programbench info` summary at the end.",
    )
    parser.add_argument(
        "--dotenv",
        default=str(ROOT / ".env"),
        help="Load KEY=VALUE lines (e.g. HF_TOKEN for gated HF test blobs) into "
        "the environment without overriding it; pass '' to skip.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    # Load .env (e.g. HF_TOKEN for the official scorer's HF test-blob download)
    # without overriding the environment, mirroring the run entry point. The
    # subprocess `programbench eval` inherits it, so huggingface-hub picks it up.
    if args.dotenv:
        harness.load_dotenv(args.dotenv)
    run_root = Path(args.run_root)
    eval_dir = (
        Path(args.eval_dir) if args.eval_dir else run_root / f"{args.run_id}_eval"
    )

    collected, missing = collect_submissions(
        run_root=run_root,
        run_id=args.run_id,
        eval_dir=eval_dir,
        instance_ids=args.instance_ids,
    )
    print(f"==> rebuilt {len(collected)} submission(s) under {eval_dir}")
    if missing:
        print(f"    no submission for: {', '.join(sorted(missing))}")
    if not collected:
        raise SystemExit("No submissions to evaluate.")

    if not args.collect_only and not args.summarize_only:
        code = run_official_eval(
            eval_dir,
            programbench_bin=args.programbench_bin,
            image_tag=args.image_tag,
            workers=args.workers,
            branch_workers=args.branch_workers,
            docker_cpus=args.docker_cpus,
            docker_memory=args.docker_memory or None,
            branch_retries=args.branch_retries,
            force=args.force,
            filter_spec=args.filter_spec,
            slice_spec=args.slice_spec,
        )
        if code != 0:
            raise SystemExit(code)

    if not args.collect_only:
        manifest = collect_eval_manifest(eval_dir, collected)
        scores_path = eval_dir / "scores.json"
        scores_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"==> wrote score manifest: {scores_path}")
        if not args.no_info:
            run_official_info(eval_dir, programbench_bin=args.programbench_bin)


if __name__ == "__main__":
    main()
