"""Run or normalize SWE-bench evaluation results.

Run from the repo root:

    PYTHONPATH=src python3 evals/swebench/evaluate_predictions.py --allow-missing-reports

By default this script normalizes existing official harness output into the
project-owned EvalResult JSONL shape. Pass --run-official to invoke
`python -m swebench.harness.run_evaluation` first.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import importlib.util
import json
import os
import stat
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_agent_lab.trajectory import json_safe, read_jsonl, write_jsonl  # noqa: E402


DEFAULT_DATASET = "princeton-nlp/SWE-bench_Lite"
DEFAULT_SPLIT = "test"
DEFAULT_RUN_ID = "simple-agent-lab-swebench"
DEFAULT_OFFICIAL_OUTPUT_DIR = ROOT / "evals/out/swebench_official"
EVAL_SCHEMA = "simple-agent-lab.evaluation.v1"


@dataclass(frozen=True)
class EvalResult:
    trace_id: str
    scorer: str
    passed: bool
    score: float
    metrics: dict[str, Any]
    reason: str = ""
    meta: dict[str, Any] | None = None


def eval_result_record(result: EvalResult) -> dict[str, Any]:
    return {
        "schema": EVAL_SCHEMA,
        "type": "eval_result",
        **json_safe(result),
    }


def load_predictions(path: str | Path) -> list[dict[str, Any]]:
    return [dict(record) for record in read_jsonl(path)]


def ensure_docker_host_env() -> None:
    """Probe known Docker socket locations when DOCKER_HOST is unset.

    The Python `docker` SDK that ships with SWE-bench falls back to
    `/var/run/docker.sock`, which doesn't exist on Docker Desktop (macOS) or
    Colima setups. Mirror the launcher script so `--run-official` works in
    headless / CI sessions without an explicit export.
    """
    if os.environ.get("DOCKER_HOST"):
        return
    home = Path(os.environ.get("HOME") or "~").expanduser()
    candidates = (
        home / ".docker/run/docker.sock",
        home / ".colima/default/docker.sock",
    )
    for sock in candidates:
        try:
            if stat.S_ISSOCK(sock.stat().st_mode):
                os.environ["DOCKER_HOST"] = f"unix://{sock}"
                return
        except FileNotFoundError:
            continue


def run_official_harness(args: argparse.Namespace) -> None:
    if importlib.util.find_spec("swebench") is None:
        raise SystemExit(
            "SWE-bench is not installed in this Python environment. "
            "Install it before using --run-official."
        )

    ensure_docker_host_env()

    run_dir = official_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    report_dir = official_report_dir(args)
    report_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        args.dataset_name,
        "--split",
        args.split,
        "--predictions_path",
        prediction_path_for_harness(args.predictions),
        "--max_workers",
        str(args.max_workers),
        "--run_id",
        args.run_id,
        "--report_dir",
        str(report_dir),
    ]
    if args.instance_ids:
        command.extend(["--instance_ids", *args.instance_ids])
    if args.cache_level:
        command.extend(["--cache_level", args.cache_level])
    if args.clean:
        command.extend(["--clean", "True"])
    if args.timeout is not None:
        command.extend(["--timeout", str(args.timeout)])
    subprocess.run(command, cwd=run_dir, check=True)


def load_official_results(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if args.results_json:
        results.update(results_from_summary(Path(args.results_json)))
    if args.instance_results_jsonl:
        results.update(results_from_instance_jsonl(Path(args.instance_results_jsonl)))
    report_dir = official_report_dir(args)
    run_dir = official_run_dir(args)
    results.update(results_from_summary_files(run_dir))
    results.update(results_from_instance_result_files(run_dir))
    results.update(results_from_report_dir(run_dir))
    results.update(results_from_summary_files(report_dir))
    results.update(results_from_instance_result_files(report_dir))
    results.update(results_from_report_dir(report_dir))
    return results


def official_run_dir(args: argparse.Namespace) -> Path:
    return resolve_repo_path(args.official_output_dir) / safe_path_part(args.run_id)


def official_report_dir(args: argparse.Namespace) -> Path:
    if args.report_dir:
        return resolve_repo_path(args.report_dir)
    return official_run_dir(args) / "reports"


def prediction_path_for_harness(path: str | Path) -> str:
    if str(path) == "gold":
        return "gold"
    return str(resolve_repo_path(path))


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def safe_path_part(value: str) -> str:
    return "".join(char if char.isalnum() or char in "_.-" else "_" for char in value)


def results_from_summary(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    resolved = set(data.get("resolved_ids") or [])
    unresolved = set(data.get("unresolved_ids") or [])
    empty = set(data.get("empty_patch_ids") or [])
    errors = set(data.get("error_ids") or [])
    incomplete = set(data.get("incomplete_ids") or [])
    submitted = set(data.get("submitted_ids") or [])

    out: dict[str, dict[str, Any]] = {}
    for instance_id in sorted(
        submitted | resolved | unresolved | empty | errors | incomplete
    ):
        status = "submitted"
        if instance_id in resolved:
            status = "resolved"
        elif instance_id in unresolved:
            status = "unresolved"
        elif instance_id in empty:
            status = "empty_patch"
        elif instance_id in errors:
            status = "error"
        elif instance_id in incomplete:
            status = "incomplete"
        out[instance_id] = {
            "resolved": instance_id in resolved,
            "status": status,
            "summary_report": data,
            "report_source": str(path),
        }
    return out


def results_from_summary_files(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for candidate in path.rglob("*.json"):
        if candidate.name == "report.json":
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or "submitted_ids" not in data:
            continue
        out.update(results_from_summary(candidate))
    return out


def results_from_instance_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        instance_id = str(record.get("instance_id") or "")
        if not instance_id:
            continue
        out[instance_id] = {
            **record,
            "resolved": bool(record.get("resolved") or record.get("passed")),
            "report_source": str(path),
        }
    return out


def results_from_instance_result_files(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for candidate in path.rglob("instance_results.jsonl"):
        out.update(results_from_instance_jsonl(candidate))
    return out


def results_from_report_dir(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for report_path in path.rglob("report.json"):
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for instance_id, result in data.items():
            if not isinstance(result, dict):
                continue
            out[str(instance_id)] = {
                **result,
                "resolved": bool(result.get("resolved")),
                "report_source": str(report_path),
            }
    return out


def eval_results_for_predictions(
    predictions: list[dict[str, Any]],
    official_results: dict[str, dict[str, Any]],
    *,
    allow_missing_reports: bool,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    missing: list[str] = []
    for prediction in predictions:
        instance_id = str(prediction.get("instance_id") or "")
        if not instance_id:
            raise SystemExit(f"Prediction is missing instance_id: {prediction}")
        official = official_results.get(instance_id)
        if official is None:
            if not allow_missing_reports:
                missing.append(instance_id)
                continue
            official = {
                "resolved": False,
                "status": "missing_report",
                "report_source": None,
            }
        results.append(eval_result_from_official(prediction, official))

    if missing:
        raise SystemExit(
            "Missing official SWE-bench reports for: "
            + ", ".join(missing)
            + ". Pass --allow-missing-reports for adapter smoke tests."
        )
    return results


def eval_result_from_official(
    prediction: dict[str, Any],
    official: dict[str, Any],
) -> EvalResult:
    instance_id = str(prediction["instance_id"])
    resolved = bool(official.get("resolved"))
    status = str(official.get("status") or ("resolved" if resolved else "unresolved"))
    reason = (
        "resolved by official SWE-bench harness"
        if resolved
        else f"SWE-bench status: {status}"
    )
    return EvalResult(
        trace_id=f"swebench.{instance_id}",
        scorer="swebench.official_harness.v1",
        passed=resolved,
        score=1.0 if resolved else 0.0,
        reason=reason,
        metrics={
            "instance_id": instance_id,
            "model_name_or_path": prediction.get("model_name_or_path"),
            "patch_chars": len(str(prediction.get("model_patch") or "")),
            "resolved": resolved,
            "status": status,
            "patch_exists": official.get("patch_exists"),
            "patch_successfully_applied": official.get("patch_successfully_applied"),
            "tests_status": official.get("tests_status"),
        },
        meta={
            "suite": "swebench",
            "report_source": official.get("report_source"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default=str(ROOT / "evals/out/swebench_predictions.jsonl"),
        help="SWE-bench prediction JSONL input.",
    )
    parser.add_argument(
        "--jsonl",
        default=str(ROOT / "evals/out/swebench_eval_results.jsonl"),
        help="Eval-result JSONL output.",
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--instance-ids", nargs="*", default=[])
    parser.add_argument("--cache-level", default="")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--timeout", type=int)
    parser.add_argument(
        "--official-output-dir",
        default=str(DEFAULT_OFFICIAL_OUTPUT_DIR),
        help=(
            "Root directory for official SWE-bench harness cwd, summary JSON, "
            "logs, and default reports. Each run uses <official-output-dir>/<run-id>/."
        ),
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help=(
            "Directory containing official SWE-bench reports. Defaults to "
            "<official-output-dir>/<run-id>/reports."
        ),
    )
    parser.add_argument("--results-json", help="Official summary results JSON.")
    parser.add_argument(
        "--instance-results-jsonl", help="Official instance results JSONL."
    )
    parser.add_argument(
        "--run-official",
        action="store_true",
        help="Invoke the official SWE-bench Docker harness before normalizing.",
    )
    parser.add_argument(
        "--allow-missing-reports",
        action="store_true",
        help="Write missing-report eval rows for local adapter smoke tests.",
    )
    args = parser.parse_args()

    if args.run_official:
        run_official_harness(args)

    predictions = load_predictions(args.predictions)
    official_results = load_official_results(args)
    results = eval_results_for_predictions(
        predictions,
        official_results,
        allow_missing_reports=args.allow_missing_reports,
    )
    write_jsonl(args.jsonl, [eval_result_record(result) for result in results])

    print(f"wrote {len(results)} SWE-bench eval results to {args.jsonl}")
    for result in results:
        status = "pass" if result.passed else "fail"
        print(
            f"{result.trace_id}: {status} score={result.score} reason={result.reason}"
        )


if __name__ == "__main__":
    main()
