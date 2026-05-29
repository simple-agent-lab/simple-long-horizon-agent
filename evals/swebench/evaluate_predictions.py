"""Run or normalize SWE-bench evaluation results.

Supports both SWE-bench Verified and SWE-bench Pro. Pass ``--pro`` to
switch to Pro mode.

By default this script normalizes existing official harness output into the
project-owned EvalResult JSONL shape. Pass ``--run-official`` to invoke
the official harness first:

- Verified: ``python -m swebench.harness.run_evaluation``
- Pro: ``swe_bench_pro_eval.py`` from scaleapi/SWE-bench_Pro-os
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
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_agent_lab.trajectory import json_safe, read_jsonl, write_jsonl  # noqa: E402


DEFAULT_DATASET = "princeton-nlp/SWE-bench_Verified"
DEFAULT_SPLIT = "test"
DEFAULT_RUN_ID = "simple-agent-lab-swebench"
DEFAULT_PREDICTIONS = ROOT / "evals/out/swebench_predictions.jsonl"
DEFAULT_EVAL_RESULTS = ROOT / "evals/out/swebench_eval_results.jsonl"
DEFAULT_OFFICIAL_OUTPUT_DIR = ROOT / "evals/out/swebench_official"
DEFAULT_PRO_PREDICTIONS = ROOT / "evals/out/swebench_pro/swebench_pro_predictions.jsonl"
DEFAULT_PRO_EVAL_RESULTS = (
    ROOT / "evals/out/swebench_pro/swebench_pro_eval_results.jsonl"
)
DEFAULT_PRO_OFFICIAL_OUTPUT_DIR = ROOT / "evals/out/swebench_pro_official"
DEFAULT_PRO_EVAL_SCRIPT = Path("/tmp/SWE-bench_Pro-os/swe_bench_pro_eval.py")
DEFAULT_PRO_SCRIPTS_DIR = Path("/tmp/SWE-bench_Pro-os/run_scripts")
DEFAULT_DOCKERHUB_USERNAME = "jefzda"
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
    prediction_path = Path(path)
    first = _first_non_whitespace(prediction_path)
    if not first:
        return []
    if first in "[{":
        try:
            with prediction_path.open(encoding="utf-8") as f:
                parsed = json.load(f)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(parsed, list):
                return [dict(record) for record in parsed]
            if isinstance(parsed, dict):
                if "predictions" in parsed and isinstance(parsed["predictions"], list):
                    return [dict(record) for record in parsed["predictions"]]
                return [dict(parsed)]
    return [dict(record) for record in read_jsonl(prediction_path)]


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


def run_official_pro_harness(args: argparse.Namespace) -> None:
    """Run the official SWE-bench Pro evaluation harness as a subprocess."""
    if not args.instances:
        raise SystemExit("--instances is required when using --pro --run-official")
    pro_eval_script = Path(args.pro_eval_script)
    if not pro_eval_script.exists():
        raise SystemExit(
            f"Official Pro evaluator not found: {pro_eval_script}\n"
            "Clone https://github.com/scaleapi/SWE-bench_Pro-os and pass "
            "--pro-eval-script /path/to/SWE-bench_Pro-os/swe_bench_pro_eval.py"
        )

    predictions_path = Path(args.predictions)
    instances_path = Path(args.instances)
    output_dir = official_report_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    instance_ids = args.instance_ids or None

    # Convert predictions to JSON array format expected by official script
    preds = load_predictions(predictions_path)
    patches: list[dict[str, str]] = []
    for pred in preds:
        iid = str(pred.get("instance_id") or "")
        if instance_ids and iid not in instance_ids:
            continue
        patches.append(
            {
                "instance_id": iid,
                "patch": str(pred.get("patch") or pred.get("model_patch") or ""),
                "prefix": str(
                    pred.get("prefix") or pred.get("model_name_or_path") or ""
                ),
            }
        )

    patches_path = output_dir / "patches.json"
    patches_path.parent.mkdir(parents=True, exist_ok=True)
    patches_path.write_text(json.dumps(patches, ensure_ascii=False), encoding="utf-8")

    # Convert instances to JSONL format expected by official script.
    instances_data = _load_instance_records(instances_path)
    if instance_ids:
        instances_data = [
            i for i in instances_data if i.get("instance_id") in instance_ids
        ]

    # Ensure list fields are string representations (official script uses eval())
    for record in instances_data:
        for fld in ("fail_to_pass", "pass_to_pass", "selected_test_files_to_run"):
            value = record.get(fld)
            if isinstance(value, list):
                record[fld] = json.dumps(value)

    instances_jsonl_path = output_dir / "instances_for_official.jsonl"
    with open(instances_jsonl_path, "w", encoding="utf-8") as fh:
        for record in instances_data:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    command = [
        sys.executable,
        str(pro_eval_script),
        "--raw_sample_path",
        str(instances_jsonl_path),
        "--patch_path",
        str(patches_path),
        "--output_dir",
        str(output_dir / "official"),
        "--dockerhub_username",
        args.dockerhub_username,
        "--scripts_dir",
        args.scripts_dir,
        "--use_local_docker",
        "--redo",
    ]

    print(f"Running official Pro evaluator: {pro_eval_script.name} ...")
    result = subprocess.run(command, cwd=pro_eval_script.parent, check=False)
    returncode = int(getattr(result, "returncode", 0) or 0)
    if returncode != 0:
        raise SystemExit(
            f"Official Pro evaluator exited with {returncode}; see logs in "
            f"{output_dir / 'official'}"
        )

    results_path = output_dir / "official" / "eval_results.json"
    if not results_path.exists():
        raise SystemExit(
            f"Official Pro evaluator did not produce {results_path}\n"
            f"Exit code: {returncode}"
        )


def load_official_results(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if args.results_json:
        results.update(results_from_summary(Path(args.results_json)))
    if args.instance_results_jsonl:
        results.update(results_from_instance_jsonl(Path(args.instance_results_jsonl)))
    run_dir = official_run_dir(args)
    report_dir = official_report_dir(args)
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
    if _is_pro_eval_results_map(data):
        return {
            str(instance_id): {
                "resolved": bool(resolved),
                "status": "resolved" if resolved else "unresolved",
                "summary_report": data,
                "report_source": str(path),
            }
            for instance_id, resolved in data.items()
        }
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
        if not isinstance(data, dict):
            continue
        if "submitted_ids" not in data and not _is_pro_eval_results_map(data):
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
    pro_prediction = "patch" in prediction or "prefix" in prediction
    suite = "swebench_pro" if pro_prediction else "swebench"
    scorer = (
        "swebench_pro.official_harness.v1"
        if pro_prediction
        else "swebench.official_harness.v1"
    )
    patch = str(prediction.get("model_patch") or prediction.get("patch") or "")
    model_name = prediction.get("model_name_or_path") or prediction.get("prefix")
    resolved = bool(official.get("resolved"))
    status = str(official.get("status") or ("resolved" if resolved else "unresolved"))
    reason = (
        f"resolved by official {suite} harness"
        if resolved
        else f"{suite} status: {status}"
    )
    return EvalResult(
        trace_id=f"swebench.{instance_id}",
        scorer=scorer,
        passed=resolved,
        score=1.0 if resolved else 0.0,
        reason=reason,
        metrics={
            "instance_id": instance_id,
            "model_name_or_path": model_name,
            "patch_chars": len(patch),
            "resolved": resolved,
            "status": status,
            "patch_exists": official.get("patch_exists"),
            "patch_successfully_applied": official.get("patch_successfully_applied"),
            "tests_status": official.get("tests_status"),
        },
        meta={
            "suite": suite,
            "report_source": official.get("report_source"),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or normalize SWE-bench (Verified / Pro) evaluation results."
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="SWE-bench prediction JSONL input.",
    )
    parser.add_argument(
        "--jsonl",
        default=None,
        help="Eval-result JSONL output.",
    )
    # Verified harness args
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--cache-level", default="")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--timeout", type=int)
    # Pro mode
    parser.add_argument(
        "--pro",
        action="store_true",
        help="Evaluate SWE-bench Pro predictions instead of Verified.",
    )
    parser.add_argument(
        "--instances",
        help="Pro instance metadata JSON/JSONL (required for --pro --run-official).",
    )
    parser.add_argument(
        "--scripts-dir",
        default=str(DEFAULT_PRO_SCRIPTS_DIR),
        help="Path to SWE-bench_Pro-os/run_scripts directory.",
    )
    parser.add_argument(
        "--pro-eval-script",
        default=str(DEFAULT_PRO_EVAL_SCRIPT),
        help="Path to official swe_bench_pro_eval.py.",
    )
    parser.add_argument(
        "--dockerhub-username",
        default=DEFAULT_DOCKERHUB_USERNAME,
        help="Docker Hub username for Pro images.",
    )
    # Result loading args
    parser.add_argument(
        "--report-dir",
        default=None,
        help=(
            "Directory containing official SWE-bench reports. Defaults to "
            "<official-output-dir>/<run-id>/reports."
        ),
    )
    parser.add_argument(
        "--official-output-dir",
        default=None,
        help=(
            "Root directory for official SWE-bench harness cwd, summary JSON, "
            "logs, and default reports. Each run uses <official-output-dir>/<run-id>/."
        ),
    )
    parser.add_argument("--results-json", help="Official summary results JSON.")
    parser.add_argument(
        "--instance-results-jsonl", help="Official instance results JSONL."
    )
    parser.add_argument("--instance-ids", nargs="*", default=[])
    # Execution modes
    parser.add_argument(
        "--run-official",
        action="store_true",
        help="Invoke the official harness before normalizing.",
    )
    parser.add_argument(
        "--allow-missing-reports",
        action="store_true",
        help="Write missing-report eval rows for local adapter smoke tests.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.pro:
        args.predictions = args.predictions or str(DEFAULT_PRO_PREDICTIONS)
        args.jsonl = args.jsonl or str(DEFAULT_PRO_EVAL_RESULTS)
        args.official_output_dir = args.official_output_dir or str(
            DEFAULT_PRO_OFFICIAL_OUTPUT_DIR
        )
    else:
        args.predictions = args.predictions or str(DEFAULT_PREDICTIONS)
        args.jsonl = args.jsonl or str(DEFAULT_EVAL_RESULTS)
        args.official_output_dir = args.official_output_dir or str(
            DEFAULT_OFFICIAL_OUTPUT_DIR
        )
    return args


def main() -> None:
    args = parse_args()

    # Run official harness if requested
    if args.run_official:
        if args.pro:
            run_official_pro_harness(args)
        else:
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


def _is_pro_eval_results_map(data: Any) -> bool:
    return (
        bool(data)
        and isinstance(data, dict)
        and all(
            isinstance(key, str) and isinstance(value, bool)
            for key, value in data.items()
        )
    )


def _load_instance_records(path: Path) -> list[dict[str, Any]]:
    first = _first_non_whitespace(path)
    if not first:
        return []
    if first in "[{":
        try:
            with path.open(encoding="utf-8") as f:
                parsed = json.load(f)
        except json.JSONDecodeError:
            if first == "[":
                raise SystemExit(f"Expected valid JSON list in {path}")
        else:
            return _records_from_json(parsed, path)
    return [dict(record) for record in read_jsonl(path)]


def _first_non_whitespace(path: Path) -> str:
    with path.open(encoding="utf-8") as f:
        while chunk := f.read(4096):
            stripped = chunk.lstrip()
            if stripped:
                return stripped[0]
    return ""


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


if __name__ == "__main__":
    main()
