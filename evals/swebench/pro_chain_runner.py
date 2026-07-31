"""Shared host-side plumbing for the SWE-bench Pro chain experiments."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

from evals.swebench import harness
from evals.swebench.evaluate_predictions import predictions_from_run_dirs
from evals.swebench.pro_memory_chain import (
    MemoryChain,
    expand_auth_slots,
    lane_auth_slots,
)
from simple_long_horizon_agent.evals.runner import prepare_new_run_directory
from simple_long_horizon_agent.llm.env import (
    API_KIND_ENV,
    OPENAI_AUTH_ENV,
    OPENAI_ENV,
    OPENAI_MODEL_ENV,
    REASONING_EFFORT_ENV,
    container_provider_env,
)
from simple_long_horizon_agent.trace import write_jsonl_atomic

ROOT = Path(__file__).resolve().parents[2]
UnitT = TypeVar("UnitT")


@dataclass(frozen=True)
class AuthLanes:
    """Resolved worker count and the auth env held by each lane."""

    parallel: int
    slots: tuple[str, ...]
    spec: str

    def as_manifest(self) -> dict[str, Any]:
        return {"spec": self.spec, "lane_slots": list(self.slots)}


@dataclass(frozen=True)
class BatchOutput:
    """Run-level paths and the complete expected prediction set."""

    batch_dir: Path
    instances_json: Path
    predictions_path: Path
    expected_instance_ids: tuple[str, ...]


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    dataset_name: str,
    split: str,
    max_turns: int,
    api_kind: str,
) -> None:
    """Add arguments with identical behavior in both Pro chain runners."""

    parser.add_argument("--all", action="store_true", help="Run the selected split.")
    parser.add_argument(
        "--repos",
        nargs="*",
        default=[],
        help="Optional exact repo names to keep, e.g. NodeBB/NodeBB.",
    )
    parser.add_argument(
        "--chains-json",
        default=None,
        help=(
            "Required chain-nodes JSONL manifest. For the vendored deep manifest, "
            "pass evals/swebench/data/"
            "swe_bench_pro_chain_experiment_nodes_deep.jsonl."
        ),
    )
    parser.add_argument("--dataset-name", default=dataset_name)
    parser.add_argument("--split", default=split)
    parser.add_argument("--instance-json", help="Use a local JSON/JSONL dataset file.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the total number of instances after chain ordering.",
    )
    parser.add_argument(
        "--max-chains",
        type=int,
        default=None,
        help="Keep only the first N multi-issue chains after longest-first ordering.",
    )
    parser.add_argument(
        "--run-root",
        default=str(harness.DEFAULT_PRO_RUN_ROOT),
        help="Output root. Defaults to evals/out/swebench_pro.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Output run id. Defaults to a timestamped id derived from the arm.",
    )
    parser.add_argument(
        "--model", default=None, help="Override OPENAI_MODEL from the environment/.env."
    )
    parser.add_argument(
        "--api-kind",
        default=api_kind,
        help="LLM adapter API kind. Defaults to openai-responses.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="Override REASONING_EFFORT from the environment/.env.",
    )
    parser.add_argument("--max-turns", type=int, default=max_turns)
    parser.add_argument(
        "--parallel",
        default="slots",
        help="'slots' to size the pool from --provider-auth-envs, or an integer.",
    )
    parser.add_argument(
        "--provider-auth-envs",
        default=None,
        help=(
            "Comma-separated auth env slots, e.g. "
            "OPENAI_AUTH_TOKEN:12,OPENAI_AUTH_TOKEN2:11. Each concurrent lane "
            "holds one slot; slots cycle if there are more lanes than slots."
        ),
    )
    parser.add_argument("--dotenv", default=str(ROOT / ".env"))
    parser.add_argument("--network-mode", default="host")
    parser.add_argument("--mem-limit", default="16g")
    parser.add_argument("--wheelhouse", default=None)
    parser.add_argument("--prepare-wheelhouse", action="store_true")
    parser.add_argument("--uv-binary", default=harness.DEFAULT_UV_BINARY)
    parser.add_argument(
        "--pull",
        default="missing",
        choices=("missing", "always", "never"),
        help="Docker image pull policy for SWE-bench Pro instance images.",
    )
    parser.add_argument("--keep-container", action="store_true")
    parser.add_argument("--docker-timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--run-official-eval",
        action="store_true",
        help="Run the official SWE-bench Pro evaluator after inference.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write the chain plan and exit before provider or Docker setup.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove any leftover deterministic container before starting it.",
    )


def load_rows(
    *,
    instance_json: str | None,
    dataset_name: str,
    split: str,
    repos: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Load a local manifest or dataset split, optionally filtering repos."""

    if instance_json:
        rows = harness.load_instance_records(instance_json)
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise SystemExit(
                "Install SWE-bench extras first: uv sync --extra swebench"
            ) from exc
        rows = [dict(row) for row in load_dataset(dataset_name, split=split)]
    if repos:
        requested = set(repos)
        rows = [row for row in rows if str(row.get("repo") or "") in requested]
    return rows


def chains_json_path(value: str | None) -> Path:
    """Resolve the explicit chain manifest required by both experiments."""

    raw = str(value or "").strip()
    if not raw:
        raise SystemExit(
            "Pass --chains-json PATH for the exact chain manifest to use, e.g. "
            "evals/swebench/data/swe_bench_pro_chain_experiment_nodes_deep.jsonl."
        )
    path = Path(raw).expanduser()
    if not path.exists():
        raise SystemExit(
            f"--chains-json not found: {path}. Pass a chain-nodes JSONL manifest."
        )
    return path


def select_units(
    chains: Iterable[MemoryChain],
    *,
    max_chains: int | None,
    limit: int | None,
    limit_per_repo: int | None = None,
) -> list[MemoryChain]:
    """Apply shared chain and instance limits without changing plan order."""

    units = list(chains)
    if max_chains is not None:
        kept_chains = 0
        selected: list[MemoryChain] = []
        for unit in units:
            if unit.is_singleton:
                selected.append(unit)
            elif kept_chains < max_chains:
                selected.append(unit)
                kept_chains += 1
        units = selected
    if limit_per_repo is not None:
        counts: dict[str, int] = {}
        selected = []
        for unit in units:
            kept_rows = []
            for row in unit.rows:
                repo = str(row.get("repo") or unit.repo)
                if counts.get(repo, 0) >= limit_per_repo:
                    continue
                kept_rows.append(row)
                counts[repo] = counts.get(repo, 0) + 1
            if kept_rows:
                selected.append(replace(unit, rows=tuple(kept_rows)))
        units = selected
    if limit is not None:
        selected = []
        seen = 0
        for unit in units:
            if seen >= limit:
                break
            take = min(unit.length, limit - seen)
            selected.append(
                replace(unit, rows=unit.rows[:take]) if take < unit.length else unit
            )
            seen += take
        units = selected
    return units


def apply_provider_env_overrides(
    *, model: str | None, reasoning_effort: str | None
) -> None:
    """Apply only explicit CLI provider overrides to the process env."""

    model_value = str(model).strip() if model is not None else ""
    if model_value:
        os.environ[OPENAI_MODEL_ENV] = model_value
    reasoning_value = (
        str(reasoning_effort).strip() if reasoning_effort is not None else ""
    )
    if reasoning_value:
        os.environ[REASONING_EFFORT_ENV] = reasoning_value


def resolve_auth_lanes(spec: str | None, parallel: str) -> AuthLanes:
    """Expand auth slots and map them onto the requested worker lanes."""

    try:
        expanded = expand_auth_slots(spec, default_env=OPENAI_ENV.auth)
    except ValueError as exc:
        raise SystemExit(f"--provider-auth-envs {exc}") from None
    count = resolve_parallel(parallel, slot_count=len(expanded))
    return AuthLanes(
        parallel=count,
        slots=tuple(lane_auth_slots(expanded, count)),
        spec=spec or f"{OPENAI_ENV.auth}:1",
    )


def resolve_parallel(value: str, *, slot_count: int) -> int:
    if value == "slots":
        return max(1, slot_count)
    try:
        parsed = int(value)
    except ValueError:
        raise SystemExit("--parallel must be 'slots' or a positive integer") from None
    if parsed <= 0:
        raise SystemExit("--parallel must be positive")
    return parsed


def provider_auth_slot_summary(auth_envs: Sequence[str]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for auth_env in auth_envs:
        if summary and summary[-1]["auth_env"] == auth_env:
            summary[-1]["lanes"] += 1
        else:
            summary.append({"auth_env": auth_env, "lanes": 1})
    return summary


def provider_env_for_auth_env(auth_env: str, *, api_kind: str) -> dict[str, str]:
    """Build the common in-container provider environment for one auth slot."""

    model = (os.environ.get(OPENAI_MODEL_ENV) or "").strip()
    token = (os.environ.get(auth_env) or "").strip()
    missing = [
        name
        for name, value in ((OPENAI_MODEL_ENV, model), (auth_env, token))
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing required env vars for SWE-bench Pro container run: "
            + ", ".join(missing)
        )

    source = dict(os.environ)
    for name in harness.OPENAI_PASSTHROUGH_ENVS:
        if source.get(name):
            source[name] = source[name].strip()
    source[OPENAI_MODEL_ENV] = model
    source[OPENAI_AUTH_ENV] = token
    env = container_provider_env("openai", harness.OPENAI_PASSTHROUGH_ENVS, env=source)
    env[API_KIND_ENV] = api_kind
    return env


def validate_provider_envs(auth_envs: Sequence[str], *, api_kind: str) -> None:
    """Fail on the host before workers start if any assigned slot is unusable."""

    try:
        for auth_env in dict.fromkeys(auth_envs):
            provider_env_for_auth_env(auth_env, api_kind=api_kind)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None


def prepare_batch_output(
    *,
    run_root: Path,
    run_id: str,
    rows: list[dict[str, Any]],
    manifest: Mapping[str, Any],
) -> BatchOutput:
    """Create run-level files and retain the full expected prediction denominator."""

    try:
        batch_dir = prepare_new_run_directory(run_root=run_root, run_id=run_id)
    except FileExistsError as exc:
        raise SystemExit(str(exc)) from None
    instances_json = batch_dir / "instances.jsonl"
    write_jsonl_atomic(instances_json, rows)
    write_json_atomic(batch_dir / "experiment.json", manifest)
    return BatchOutput(
        batch_dir=batch_dir,
        instances_json=instances_json,
        predictions_path=batch_dir / f"{run_id}_predictions.jsonl",
        expected_instance_ids=tuple(str(row["instance_id"]) for row in rows),
    )


def write_predictions(
    output: BatchOutput,
    *,
    model_name: str,
    dataset_name: str,
) -> list[dict[str, Any]]:
    """Collect final results once all concurrent chain workers have stopped."""

    predictions = predictions_from_run_dirs(
        output.batch_dir.parent,
        run_id=output.batch_dir.name,
        model_name=model_name,
        dataset_name=dataset_name,
        expected_instance_ids=output.expected_instance_ids,
    )
    write_jsonl_atomic(output.predictions_path, predictions)
    return predictions


def run_auth_lanes(
    units: Iterable[UnitT],
    *,
    lanes: AuthLanes,
    chain_id: Callable[[UnitT], str],
    worker: Callable[[UnitT, str], dict[str, Any]],
    on_done: Callable[[str, dict[str, Any]], None],
) -> list[dict[str, str]]:
    """Run units concurrently while each unit holds and finally returns one slot."""

    slot_pool: queue.Queue[str] = queue.Queue()
    for slot in lanes.slots:
        slot_pool.put(slot)

    def assigned_worker(unit: UnitT) -> dict[str, Any]:
        auth_env = slot_pool.get()
        try:
            return worker(unit, auth_env)
        finally:
            slot_pool.put(auth_env)

    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=lanes.parallel) as pool:
        futures = {pool.submit(assigned_worker, unit): chain_id(unit) for unit in units}
        for future in as_completed(futures):
            unit_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                failures.append({"chain_id": unit_id, "error": error})
                print(f"[FAIL] {unit_id}: {error}", flush=True)
            else:
                on_done(unit_id, result)
    return failures


def run_official_eval(
    *,
    predictions_path: Path,
    instances_json: Path,
    run_id: str,
    max_workers: int,
) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "evals.swebench.evaluate_predictions",
            "--pro",
            "--run-official",
            "--predictions",
            str(predictions_path),
            "--instances",
            str(instances_json),
            "--run-id",
            run_id,
            "--max-workers",
            str(max_workers),
        ],
        cwd=ROOT,
        check=True,
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    try:
        tmp.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
