"""Run or normalize SWE-bench evaluation results.

Supports SWE-bench Verified, SWE-bench Multilingual, and SWE-bench Pro. Pass
``--multilingual`` or ``--pro`` to switch away from Verified mode.

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
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_agent_lab.trace import json_safe, read_jsonl, write_jsonl  # noqa: E402
from simple_agent_lab.evals.runner import canonical_run_id  # noqa: E402


DEFAULT_DATASET = "princeton-nlp/SWE-bench_Verified"
DEFAULT_MULTILINGUAL_DATASET = "SWE-bench/SWE-bench_Multilingual"
DEFAULT_SPLIT = "test"
DEFAULT_RUN_ID = "simple-agent-lab-swebench"
DEFAULT_PREDICTIONS = ROOT / "evals/out/swebench_predictions.jsonl"
DEFAULT_EVAL_RESULTS = ROOT / "evals/out/swebench_eval_results.jsonl"
DEFAULT_OFFICIAL_OUTPUT_DIR = ROOT / "evals/out/swebench_official"
DEFAULT_MULTILINGUAL_PREDICTIONS = (
    ROOT / "evals/out/swebench_multilingual/swebench_multilingual_predictions.jsonl"
)
DEFAULT_MULTILINGUAL_EVAL_RESULTS = (
    ROOT / "evals/out/swebench_multilingual/swebench_multilingual_eval_results.jsonl"
)
DEFAULT_MULTILINGUAL_OFFICIAL_OUTPUT_DIR = (
    ROOT / "evals/out/swebench_multilingual_official"
)
DEFAULT_PRO_PREDICTIONS = ROOT / "evals/out/swebench_pro/swebench_pro_predictions.jsonl"
DEFAULT_PRO_EVAL_RESULTS = (
    ROOT / "evals/out/swebench_pro/swebench_pro_eval_results.jsonl"
)
DEFAULT_PRO_OFFICIAL_OUTPUT_DIR = ROOT / "evals/out/swebench_pro_official"
_PRO_REPO_DIR = ROOT / "evals/out/swebench_pro/official_harness"
_PRO_REPO_URL = "https://github.com/scaleapi/SWE-bench_Pro-os.git"
DEFAULT_PRO_EVAL_SCRIPT = _PRO_REPO_DIR / "swe_bench_pro_eval.py"
DEFAULT_PRO_SCRIPTS_DIR = _PRO_REPO_DIR / "run_scripts"
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
    # The official harness reads the predictions file as one JSON object per
    # line, but our predictions are written pretty-printed (write_jsonl, indent=2
    # — readable in the trace viewer). Re-emit a compact one-per-line copy for
    # the subprocess so it doesn't choke on multi-line records. ``gold`` is a
    # sentinel the harness resolves itself, so pass it through untouched.
    if str(args.predictions) == "gold":
        predictions_arg = "gold"
    else:
        compact_predictions = run_dir / "predictions_compact.jsonl"
        with compact_predictions.open("w", encoding="utf-8") as fh:
            for record in load_predictions(args.predictions):
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        predictions_arg = str(compact_predictions)
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        args.dataset_name,
        "--split",
        args.split,
        "--predictions_path",
        predictions_arg,
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


_PRO_DOCKER_TIMEOUT_S = 600


def _ensure_pro_repo(target_dir: Path) -> None:
    """Clone SWE-bench_Pro-os when the evaluator is not already available."""
    eval_script = target_dir / "swe_bench_pro_eval.py"
    if eval_script.exists():
        return
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {_PRO_REPO_URL} → {target_dir} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", _PRO_REPO_URL, str(target_dir)],
        check=True,
    )
    if not eval_script.exists():
        raise SystemExit(
            f"Clone succeeded but {eval_script} not found — "
            "the upstream repo layout may have changed."
        )


def _patch_pro_evaluator(eval_script: Path) -> None:
    """Install the local timeout and fail-fast patch-application contract."""

    text = eval_script.read_text(encoding="utf-8")
    patched = text.replace(
        "docker.from_env()",
        f"docker.from_env(timeout={_PRO_DOCKER_TIMEOUT_S})",
    )
    marker = "PATCH_APPLY_STATUS=/workspace/patch_apply_status.json"
    if marker not in patched:
        old = """# apply patch
cd /app
git reset --hard {base_commit}
git checkout {base_commit}
git apply -v /workspace/patch.diff
{before_repo_set_cmd}
# run test and save stdout and stderr to separate files"""
        new = """# apply patch
cd /app
PATCH_APPLY_STATUS=/workspace/patch_apply_status.json
PATCH_APPLY_STDERR=/workspace/patch_apply.stderr
if ! git reset --hard {base_commit}; then
  printf '%s\\n' '{{"success": false, "stage": "reset"}}' > "$PATCH_APPLY_STATUS"
  exit 80
fi
if ! git checkout {base_commit}; then
  printf '%s\\n' '{{"success": false, "stage": "checkout"}}' > "$PATCH_APPLY_STATUS"
  exit 81
fi
if ! git apply --check /workspace/patch.diff 2> "$PATCH_APPLY_STDERR"; then
  printf '%s\\n' '{{"success": false, "stage": "check"}}' > "$PATCH_APPLY_STATUS"
  exit 82
fi
if ! git apply -v /workspace/patch.diff 2> "$PATCH_APPLY_STDERR"; then
  printf '%s\\n' '{{"success": false, "stage": "apply"}}' > "$PATCH_APPLY_STATUS"
  exit 83
fi
printf '%s\\n' '{{"success": true, "stage": "applied"}}' > "$PATCH_APPLY_STATUS"
{before_repo_set_cmd}
# run test and save stdout and stderr to separate files"""
        if old not in patched:
            raise RuntimeError(
                f"Official Pro evaluator entryscript shape changed: {eval_script}"
            )
        patched = patched.replace(old, new, 1)
    local_image_marker = "Using locally cached Docker image:"
    if local_image_marker not in patched:
        old_pull = """        try:
            if docker_platform:
                client.images.pull(dockerhub_image_uri, platform=docker_platform)
            else:
                client.images.pull(dockerhub_image_uri)
        except Exception as pull_err:
            # If pull fails, fall back to a local image if present; otherwise, fail this run
            try:
                client.images.get(dockerhub_image_uri)
                print(f"Using locally available image: {dockerhub_image_uri}")
            except Exception:
                print(f"Failed to pull or find image locally for {uid}: {pull_err}")
                return None"""
        local_first = """        try:
            client.images.get(dockerhub_image_uri)
            print(f"Using locally cached Docker image: {dockerhub_image_uri}")
        except Exception:
            try:
                if docker_platform:
                    client.images.pull(
                        dockerhub_image_uri, platform=docker_platform
                    )
                else:
                    client.images.pull(dockerhub_image_uri)
            except Exception as pull_err:
                print(f"Failed to pull image for {uid}: {pull_err}")
                return None"""
        if old_pull not in patched:
            raise RuntimeError(
                f"Official Pro evaluator image-pull shape changed: {eval_script}"
            )
        patched = patched.replace(old_pull, local_first, 1)
    if patched != text:
        eval_script.write_text(patched, encoding="utf-8")
        print(
            f"Patched {eval_script.name}: timeout + strict apply + local-first images"
        )


def run_official_pro_harness(args: argparse.Namespace) -> None:
    """Run the official SWE-bench Pro evaluation harness as a subprocess."""
    if not args.instances:
        raise SystemExit("--instances is required when using --pro --run-official")
    pro_eval_script = Path(args.pro_eval_script)
    if not pro_eval_script.exists():
        if pro_eval_script == DEFAULT_PRO_EVAL_SCRIPT:
            _ensure_pro_repo(_PRO_REPO_DIR)
        else:
            raise SystemExit(
                f"Official Pro evaluator not found: {pro_eval_script}\n"
                "Pass --pro-eval-script /path/to/swe_bench_pro_eval.py"
            )
    _patch_pro_evaluator(pro_eval_script)

    predictions_path = Path(args.predictions)
    instances_path = Path(args.instances)
    output_dir = official_report_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    official_output_dir = output_dir / "official"
    if official_output_dir.exists():
        # Upstream --redo rewrites inputs but leaves workspace/output.json
        # behind. Remove the prior evaluator products so a failed parser cannot
        # be mistaken for this patch's result.
        shutil.rmtree(official_output_dir)
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
        str(official_output_dir),
        "--dockerhub_username",
        args.dockerhub_username,
        "--scripts_dir",
        args.scripts_dir,
        "--num_workers",
        str(getattr(args, "max_workers", 1)),
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
    if getattr(args, "pro", False):
        merge_pro_apply_statuses(results, report_dir / "official")
    return results


def merge_pro_apply_statuses(
    results: dict[str, dict[str, Any]], official_dir: Path
) -> None:
    """Attach strict Pro patch-application status and force failures unresolved."""

    for status_path in official_dir.glob(
        "instance_*/workspace/patch_apply_status.json"
    ):
        instance_id = status_path.parents[1].name
        status = json.loads(status_path.read_text(encoding="utf-8"))
        success = bool(status.get("success"))
        record = results.setdefault(
            instance_id,
            {
                "resolved": False,
                "status": "missing_official_result",
                "report_source": str(status_path),
            },
        )
        record["patch_exists"] = True
        record["patch_successfully_applied"] = success
        record["patch_apply_stage"] = str(status.get("stage") or "")
        stderr_path = status_path.with_name("patch_apply.stderr")
        if not success and stderr_path.is_file():
            record["patch_apply_stderr"] = stderr_path.read_text(
                encoding="utf-8", errors="replace"
            )
        if not success:
            record["resolved"] = False
            record["status"] = "patch_apply_failed"


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
    suite: str | None = None,
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
        results.append(eval_result_from_official(prediction, official, suite=suite))

    if missing:
        raise SystemExit(
            "Missing official SWE-bench reports for: "
            + ", ".join(missing)
            + ". Pass --allow-missing-reports for adapter smoke tests."
        )
    return results


def parity_mismatches(
    separate_rows: list[dict[str, Any]],
    reuse_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare official "separate" rows to "reuse" rows; list resolved disagreements.

    The hard parity requirement (ADR scorer-seam-and-scoring-topology): the in-environment "reuse" verdict
    must match the official harness. Empty list == parity holds for this sample.
    Keyed on ``metrics.instance_id`` (falls back to ``trace_id``).
    """

    def by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            metrics = row.get("metrics") or {}
            key = str(metrics.get("instance_id") or row.get("trace_id") or "")
            out[key] = row
        return out

    sep, rse = by_id(separate_rows), by_id(reuse_rows)
    mismatches: list[dict[str, Any]] = []
    for instance_id in sorted(set(sep) | set(rse)):
        a, b = sep.get(instance_id), rse.get(instance_id)
        if a is None or b is None or a.get("passed") != b.get("passed"):
            mismatches.append(
                {
                    "instance_id": instance_id,
                    "separate_passed": None if a is None else a.get("passed"),
                    "reuse_passed": None if b is None else b.get("passed"),
                }
            )
    return mismatches


def eval_rows_from_official(
    predictions: list[dict[str, Any]],
    official_results: dict[str, dict[str, Any]],
    *,
    suite: str | None = None,
    allow_missing_reports: bool = True,
) -> list[dict[str, Any]]:
    """Importable core: predictions + official harness results -> eval-result rows.

    The pure normalize step shared by the official CLI path (`--run-official`)
    and the in-environment "reuse" path (`reuse_eval_row`). Routing both through
    this one mapping is what makes their rows byte-identical and the reuse
    verdict trustable against the official harness (parity gate; ADR collapse-scorer-seam-into-run-primitive).
    """

    results = eval_results_for_predictions(
        predictions,
        official_results,
        suite=suite,
        allow_missing_reports=allow_missing_reports,
    )
    return [eval_result_record(result) for result in results]


def eval_result_from_official(
    prediction: dict[str, Any],
    official: dict[str, Any],
    *,
    suite: str | None = None,
) -> EvalResult:
    instance_id = str(prediction["instance_id"])
    pro_prediction = "patch" in prediction or "prefix" in prediction
    suite = suite or ("swebench_pro" if pro_prediction else "swebench")
    scorer = f"{suite}.official_harness.v1"
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
            "patch_apply_stage": official.get("patch_apply_stage"),
            "tests_status": official.get("tests_status"),
        },
        meta={
            "suite": suite,
            "report_source": official.get("report_source"),
            "patch_apply_stderr": official.get("patch_apply_stderr"),
        },
    )


def reuse_eval_row(
    instance: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    dataset_name: str = DEFAULT_DATASET,
    model_name: str = "simple-agent-lab",
    namespace: str = "swebench",
    instance_image_tag: str = "latest",
    env_image_tag: str = "latest",
) -> dict[str, Any]:
    """Grade one in-environment ("reuse") run into an eval-result row.

    SWE-bench's in-environment scoring (the container-half ``evaluate`` hook) runs
    the official eval script where the agent ran and merges into ``result.json``
    either an explicit verdict (``resolved``) or the official log (``eval_log``).
    Full grading needs the ``swebench`` grader plus the gold test spec, which live
    host-side — so this small helper does the last step here. It routes through
    the *same* `eval_result_from_official` mapping as ``--run-official``, so the
    rows are interchangeable and the parity gate (`parity_mismatches`) can
    cross-check the reuse verdict against the official harness (ADR collapse-scorer-seam-into-run-primitive).
    """

    from evals.swebench import harness

    instance_id = str(instance.get("instance_id") or "")
    prediction = harness.prediction_record(
        instance_id,
        model_name,
        str(result.get("model_patch", "")),
        dataset_name=dataset_name,
    )
    suite = harness.suite_for_instance(
        dataset_name=dataset_name, instance_id=instance_id
    )
    if result.get("resolved") is not None:
        official = _official_from_verdict(result)
    elif result.get("eval_log") is not None:
        official = _grade_reuse_log(
            instance,
            result,
            namespace=namespace,
            instance_image_tag=instance_image_tag,
            env_image_tag=env_image_tag,
        )
    else:
        official = {
            "resolved": False,
            "status": "no_reuse_verdict",
            "report_source": "in-environment reuse (no verdict)",
        }
    return eval_result_record(
        eval_result_from_official(prediction, official, suite=suite)
    )


def _official_from_verdict(result: Mapping[str, Any]) -> dict[str, Any]:
    """Map a self-grading bench's explicit verdict into the official-results shape."""

    resolved = bool(result.get("resolved"))
    return {
        "resolved": resolved,
        "status": str(
            result.get("status") or ("resolved" if resolved else "unresolved")
        ),
        "tests_status": result.get("tests_status"),
        "patch_successfully_applied": result.get("patch_successfully_applied"),
        "report_source": result.get("report_source") or "in-environment reuse",
    }


def _grade_reuse_log(
    instance: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    namespace: str,
    instance_image_tag: str,
    env_image_tag: str,
) -> dict[str, Any]:
    """Grade the captured official eval log host-side with the official grader."""

    from evals.swebench import harness

    try:
        from swebench.harness.grading import get_eval_report
    except Exception as exc:  # noqa: BLE001 — surface as an unresolved diagnostic
        return {
            "resolved": False,
            "status": f"swebench_unavailable: {exc}",
            "report_source": "in-environment reuse (host grading)",
        }

    instance_id = str(instance.get("instance_id") or "")
    spec = harness._make_swebench_test_spec(
        dict(instance),
        namespace=namespace,
        instance_image_tag=instance_image_tag,
        env_image_tag=env_image_tag,
    )
    with tempfile.TemporaryDirectory(prefix="sal-reuse-grade-") as tmp:
        log_path = Path(tmp) / "eval.log"
        log_path.write_text(str(result.get("eval_log") or ""), encoding="utf-8")
        prediction = {
            "instance_id": instance_id,
            "model_patch": str(result.get("model_patch", "")),
        }
        report = get_eval_report(
            spec, prediction, str(log_path), include_tests_status=True
        )
    instance_report = report.get(instance_id, {})
    resolved = bool(instance_report.get("resolved"))
    return {
        "resolved": resolved,
        "status": "resolved" if resolved else "unresolved",
        "tests_status": instance_report.get("tests_status"),
        "patch_successfully_applied": instance_report.get("patch_successfully_applied"),
        "report_source": "in-environment reuse (official eval script + grader)",
    }


def predictions_from_run_dirs(
    run_root: str | Path,
    *,
    run_id: str | None = None,
    model_name: str = "simple-agent-lab-containerized",
    dataset_name: str = DEFAULT_DATASET,
    expected_instance_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Shape generic run dirs into official SWE-bench prediction records.

    Generic runs write ``<run-root>/<run-id>/<instance-id>/out/result.json`` with
    the collected workspace diff under ``model_patch``. The official harness
    instead wants a predictions JSONL keyed by instance id + model name. This
    rebuilds that record via `harness.prediction_record` (so Verified,
    Multilingual, and Pro shapes stay correct), taking the instance id from the
    staged ``input/instance.json`` (falling back to the run-dir name). Empty-patch
    runs are kept — the harness counts them unresolved, so totals match the
    launched set. With ``run_id`` only that run is collected; without it, every
    run under ``run_root``.
    """

    from evals.swebench import harness

    root = Path(run_root)
    search = (root / canonical_run_id(run_id)).glob("*") if run_id else root.glob("*/*")
    expected = tuple(str(value) for value in (expected_instance_ids or ()))
    if len(set(expected)) != len(expected):
        raise ValueError("Expected instance ids contain duplicates")
    expected_set = set(expected)
    predictions_by_id: dict[str, dict[str, Any]] = {}
    for run_dir in sorted(p for p in search if p.is_dir()):
        instance_id = _instance_id_for_run_dir(run_dir)
        if instance_id in predictions_by_id:
            raise ValueError(
                f"Duplicate result for instance id {instance_id!r}: {run_dir}"
            )
        result_path = run_dir / "out" / "result.json"
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8") or "{}")
        if expected and instance_id not in expected_set:
            raise ValueError(
                f"Unexpected instance id {instance_id!r} under run {run_id!r}"
            )
        patch = str(result.get("model_patch", ""))
        predictions_by_id[instance_id] = harness.prediction_record(
            instance_id,
            model_name,
            patch,
            dataset_name=dataset_name,
        )
    if not expected:
        return [predictions_by_id[key] for key in sorted(predictions_by_id)]
    return [
        predictions_by_id.get(instance_id)
        or harness.prediction_record(
            instance_id,
            model_name,
            "",
            dataset_name=dataset_name,
        )
        for instance_id in expected
    ]


def _instance_id_for_run_dir(run_dir: Path) -> str:
    """Read the instance id from the staged input, else use the run-dir name."""

    instance_json = run_dir / "input" / "instance.json"
    if instance_json.is_file():
        try:
            record = json.loads(instance_json.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError:
            record = {}
        instance_id = str(record.get("instance_id") or "")
        if instance_id:
            return instance_id
    return run_dir.name


def run_scoped_eval_results_path(predictions: str) -> str:
    """Default eval-result output, bound to the run that produced the predictions.

    Lands at ``<predictions_dir>/<run_id>/eval_results.jsonl`` — inside the run
    directory, beside the per-instance trajectories — so each judge is tied to
    one run rather than accumulating in a suite-level file keyed only by
    instance_id (which collides across runs of the same instance). The run_id is
    the predictions stem with the ``.fixed`` / ``_predictions`` suffixes
    stripped, inverting how the collect-predictions step names the file
    (``<run_id>_predictions.jsonl``). The trace viewer derives the same run_id
    from this path and joins verdicts to trajectories by (run_id, instance_id).
    """
    p = Path(predictions)
    stem = p.name
    for ext in (".jsonl", ".json"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    if stem.endswith(".fixed"):
        stem = stem[: -len(".fixed")]
    if stem.endswith("_predictions"):
        stem = stem[: -len("_predictions")]
    return str(p.parent / stem / "eval_results.jsonl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or normalize SWE-bench (Verified / Multilingual / Pro) "
            "evaluation results."
        )
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
        "--multilingual",
        action="store_true",
        help="Evaluate SWE-bench Multilingual predictions instead of Verified.",
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
    parser.add_argument(
        "--verify-parity",
        action="store_true",
        help=(
            "Parity gate (ADR scorer-seam-and-scoring-topology): cross-check the official 'separate' rows "
            "against a 'reuse' eval-result JSONL (--reuse-results) and exit "
            "non-zero on any resolved disagreement."
        ),
    )
    parser.add_argument(
        "--reuse-results",
        help="Eval-result JSONL produced by the 'reuse' topology (for --verify-parity).",
    )
    parser.add_argument(
        "--collect-predictions",
        action="store_true",
        help=(
            "Shape generic run dirs "
            "(<run-root>/<run-id>/<instance>/out/result.json) into an official "
            "predictions JSONL at --predictions, then exit. Feed that file to a "
            "later --run-official run."
        ),
    )
    parser.add_argument(
        "--run-root",
        help="Run root for --collect-predictions (e.g. evals/out/swebench).",
    )
    parser.add_argument(
        "--model-name",
        default="simple-agent-lab-containerized",
        help="model_name_or_path label written into collected predictions.",
    )
    parser.add_argument(
        "--expected-ids-file",
        help=(
            "Expected instance ids, one per line. Missing run results are emitted "
            "as empty patches so failed instances remain in the denominator."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.pro and args.multilingual:
        raise SystemExit("Pass at most one of --pro or --multilingual.")
    if args.pro:
        args.predictions = args.predictions or str(DEFAULT_PRO_PREDICTIONS)
        args.official_output_dir = args.official_output_dir or str(
            DEFAULT_PRO_OFFICIAL_OUTPUT_DIR
        )
    elif args.multilingual:
        args.dataset_name = DEFAULT_MULTILINGUAL_DATASET
        args.predictions = args.predictions or str(DEFAULT_MULTILINGUAL_PREDICTIONS)
        args.official_output_dir = args.official_output_dir or str(
            DEFAULT_MULTILINGUAL_OFFICIAL_OUTPUT_DIR
        )
    else:
        args.predictions = args.predictions or str(DEFAULT_PREDICTIONS)
        args.official_output_dir = args.official_output_dir or str(
            DEFAULT_OFFICIAL_OUTPUT_DIR
        )
    # Default eval-result output is bound to the run that produced the
    # predictions (see run_scoped_eval_results_path); --jsonl still overrides.
    args.jsonl = args.jsonl or run_scoped_eval_results_path(args.predictions)
    return args


def main() -> None:
    args = parse_args()

    # Collect generic run dirs into a predictions JSONL, then exit. This bridges
    # the run phase (result.json per instance) to the official-harness scoring
    # phase (a predictions file), replacing the per-run prediction.jsonl the
    # legacy launcher used to write.
    if args.collect_predictions:
        if not args.run_root:
            raise SystemExit("--collect-predictions requires --run-root PATH")
        expected_instance_ids = (
            _load_expected_instance_ids(Path(args.expected_ids_file))
            if args.expected_ids_file
            else None
        )
        run_id = None if args.run_id == DEFAULT_RUN_ID else args.run_id
        predictions = predictions_from_run_dirs(
            args.run_root,
            run_id=run_id,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            expected_instance_ids=expected_instance_ids,
        )
        write_jsonl(args.predictions, predictions)
        empty = sum(
            1 for p in predictions if not (p.get("model_patch") or p.get("patch"))
        )
        suffix = f" ({empty} with an empty patch)" if empty else ""
        print(f"wrote {len(predictions)} predictions to {args.predictions}{suffix}")
        return

    # Run official harness if requested
    if args.run_official:
        if args.pro:
            run_official_pro_harness(args)
        else:
            run_official_harness(args)

    predictions = load_predictions(args.predictions)
    official_results = load_official_results(args)
    suite = (
        "swebench_pro"
        if args.pro
        else "swebench_multilingual"
        if args.multilingual
        else "swebench"
    )
    rows = eval_rows_from_official(
        predictions,
        official_results,
        suite=suite,
        allow_missing_reports=args.allow_missing_reports,
    )
    write_jsonl(args.jsonl, rows)

    print(f"wrote {len(rows)} SWE-bench eval results to {args.jsonl}")
    for row in rows:
        status = "pass" if row.get("passed") else "fail"
        print(
            f"{row.get('trace_id')}: {status} score={row.get('score')} "
            f"reason={row.get('reason')}"
        )

    if args.verify_parity:
        if not args.reuse_results:
            raise SystemExit("--verify-parity requires --reuse-results PATH")
        reuse_rows = [dict(record) for record in read_jsonl(Path(args.reuse_results))]
        mismatches = parity_mismatches(rows, reuse_rows)
        if mismatches:
            raise SystemExit(
                "Parity gate FAILED: reuse verdicts disagree with the official "
                f"harness for {len(mismatches)} instance(s): {mismatches}"
            )
        print(f"parity gate PASSED: reuse == separate on {len(rows)} instance(s)")


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


def _load_expected_instance_ids(path: Path) -> tuple[str, ...]:
    instance_ids: list[str] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        instance_id = line.split("#", 1)[0].strip()
        if not instance_id:
            continue
        if instance_id in seen:
            raise SystemExit(
                f"Duplicate instance id in {path} on line {lineno}: {instance_id}"
            )
        seen.add(instance_id)
        instance_ids.append(instance_id)
    if not instance_ids:
        raise SystemExit(f"Expected ids file is empty: {path}")
    return tuple(instance_ids)


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
