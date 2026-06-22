"""Host-side SWE-bench / DGM compatibility and official-scoring glue.

The generic v1 simple self-evolving path no longer builds its rollout here. It
registers SWE-bench factories in ``evals/swebench/self_evolving.py`` and then
composes ``Suite`` + ``AgentSurface`` + ``rollout_from_suite`` from the YAML
runner.

DGM remains recipe-local and still imports this module for its SWE-bench
compatibility helpers: layouts, seed/version artifact staging, rollout wrapping,
reward extraction, and official-scoring command builders. Keep this file
Docker-free by design (arch_lint restricts ``import docker`` to
``evals.backends``); Docker probing belongs in the recipe layer.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simple_agent_lab.evals.protocols import AGENT_PACKAGE_KEY
from simple_agent_lab.evals.suites.swebench import agent_package as agent_pkg
from simple_agent_lab.evolution.types import Run, Slice, Version
from simple_agent_lab.trace.jsonl import read_jsonl, write_jsonl

DEFAULT_DATASET = "princeton-nlp/SWE-bench_Verified"
DEFAULT_MODEL_NAME = "dgm-swebench"
AGENT_PREFIX = "agent/"
EVOLVING_CONTAINER_MODULE = "simple_agent_lab.evals.suites.swebench.evolving"
OPENAI_AUTH_ENV = "OPENAI_AUTH_TOKEN"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"


@dataclass(frozen=True)
class PerformanceLayout:
    output_root: Path
    run_id: str

    @property
    def run_root(self) -> Path:
        return self.output_root / self.run_id

    @property
    def evolution_workspace(self) -> Path:
        return self.run_root / "evolution"

    @property
    def swebench_runs(self) -> Path:
        return self.run_root / "swebench_runs"

    @property
    def official(self) -> Path:
        return self.run_root / "official"

    @property
    def predictions(self) -> Path:
        return self.official / f"{self.run_id}_predictions.jsonl"

    @property
    def eval_results(self) -> Path:
        return self.official / "eval_results.jsonl"

    @property
    def generation_metrics(self) -> Path:
        return self.run_root / "generation_metrics.jsonl"

    def create(self) -> None:
        for path in (
            self.evolution_workspace,
            self.swebench_runs,
            self.official,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class OfficialArtifacts:
    label: str
    root: Path
    predictions: Path
    eval_results: Path
    harness: Path
    run_id: str


def official_artifacts(layout: PerformanceLayout, label: str) -> OfficialArtifacts:
    safe = _safe_path_part(label)
    root = layout.official / safe
    return OfficialArtifacts(
        label=safe,
        root=root,
        predictions=root / f"{safe}_predictions.jsonl",
        eval_results=root / "eval_results.jsonl",
        harness=root / "harness",
        run_id=f"{layout.run_id}-{safe}",
    )


def load_dataset(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load full SWE-bench instance records from a JSONL dataset file."""

    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return tuple(rows)


def instance_ids(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(str(record.get("instance_id", "")) for record in records)


def collect_predictions_command(
    layout: PerformanceLayout,
    *,
    dataset_name: str,
    model_name: str,
    source_run_id: str | None = None,
    predictions: str | Path | None = None,
) -> list[str]:
    """Build the DGM command that collects rollout outputs into predictions JSONL."""

    return [
        sys.executable,
        "evals/swebench/evaluate_predictions.py",
        "--collect-predictions",
        "--run-root",
        str(layout.swebench_runs),
        "--run-id",
        source_run_id or layout.run_id,
        "--dataset-name",
        dataset_name,
        "--model-name",
        model_name,
        "--predictions",
        str(predictions or layout.predictions),
    ]


def official_eval_command(
    layout: PerformanceLayout,
    *,
    dataset_name: str,
    instance_ids: Sequence[str] = (),
    max_workers: int = 1,
    predictions: str | Path | None = None,
    eval_results: str | Path | None = None,
    official_output_dir: str | Path | None = None,
    run_id: str | None = None,
) -> list[str]:
    """Build the DGM command that runs or normalizes official SWE-bench scoring."""

    command = [
        sys.executable,
        "evals/swebench/evaluate_predictions.py",
        "--run-official",
        "--dataset-name",
        dataset_name,
        "--predictions",
        str(predictions or layout.predictions),
        "--jsonl",
        str(eval_results or layout.eval_results),
        "--official-output-dir",
        str(official_output_dir or layout.official / "harness"),
        "--run-id",
        run_id or layout.run_id,
        "--max-workers",
        str(max_workers),
    ]
    if instance_ids:
        command.extend(["--instance-ids", *instance_ids])
    return command


def summarize_official_eval_results(path: str | Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    total = len(rows)
    resolved = sum(
        1
        for row in rows
        if bool(row.get("passed")) or float(row.get("score", 0.0) or 0.0) > 0.0
    )
    return {
        "resolved": resolved,
        "total": total,
        "resolved_rate": resolved / total if total else 0.0,
    }


def generation_metric_record(
    *,
    generation: int,
    version_hash: str,
    parent_hash: str,
    parent_selection: str,
    decision_outcome: str,
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the compact DGM metrics row written after held-out scoring."""

    total = len(runs)
    resolved = sum(1 for run in runs if float(run.get("reward", 0.0)) > 0.0)
    patch_valid = sum(1 for run in runs if int(run.get("patch_chars", 0)) > 0)
    tokens = sum(int(run.get("tokens", 0)) for run in runs)
    return {
        "generation": generation,
        "version": version_hash,
        "parent": parent_hash,
        "parent_selection": parent_selection,
        "outcome": decision_outcome,
        "total": total,
        "resolved": resolved,
        "resolved_rate": resolved / total if total else 0.0,
        "patch_valid": patch_valid,
        "patch_valid_rate": patch_valid / total if total else 0.0,
        "tokens": tokens,
    }


def build_swebench_rollout(
    layout: PerformanceLayout,
    *,
    dataset_name: str = DEFAULT_DATASET,
    concurrency: int = 1,
    run_kwargs: Mapping[str, Any] | None = None,
    wheelhouse: str | Path | None = None,
    uv_binary: str | Path | None = None,
    in_env_scoring: bool = False,
    version_artifacts: Any = None,
    container_module: str | None = None,
):
    """Build the DGM rollout on the mature SWE-bench Suite path.

    ``version_artifacts`` is forwarded to ``dataset_rollout`` so a version's
    evolved files (e.g. the agent package) are staged into each run.
    ``container_module`` overrides the suite's container half (e.g. the
    ``evolving`` module that builds the agent from the staged package).

    Generic simple runs should prefer ``evals/swebench/self_evolving.py`` plus
    the YAML runner, which composes ``AgentSurface`` and ``rollout_from_suite``.
    This helper stays for DGM's recipe-local wiring and tests.
    """

    from evals.swebench.harness import DEFAULT_WHEELHOUSE_MOUNT
    from evals.swebench.suite import SwebenchSuite
    from simple_agent_lab.evals import LocalDirStore, LocalDockerBackend
    from simple_agent_lab.evolution.components.rollout import dataset_rollout

    extra_kwargs = dict(run_kwargs or {})
    if wheelhouse and "wheelhouse_mount" not in extra_kwargs:
        extra_kwargs["wheelhouse_mount"] = DEFAULT_WHEELHOUSE_MOUNT
    suite = SwebenchSuite(dataset_name=dataset_name, in_env_scoring=in_env_scoring)
    if container_module:
        suite.container_module = container_module
    return dataset_rollout(
        suite=suite,
        backend=LocalDockerBackend(wheelhouse=wheelhouse, uv_binary=uv_binary),
        store=LocalDirStore(layout.swebench_runs),
        runs_root=layout.swebench_runs,
        concurrency=concurrency,
        run_kwargs=extra_kwargs,
        version_artifacts=version_artifacts,
    )


def write_generation_metrics(
    path: str | Path, records: Sequence[Mapping[str, Any]]
) -> None:
    write_jsonl(path, records)


def _safe_path_part(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in value)
    return safe.strip("-") or "official"


def ensure_rollout_artifacts(
    runs: Sequence[Run], *, min_complete_fraction: float = 0.5
) -> None:
    """Tolerate a few missing ``result.json`` files; hard-fail only on systemic loss.

    A single straggler/interrupted container (or a transient per-instance crash)
    should not abort the whole evolution run: those instances simply score 0 via
    the reward path and re-roll on a later visit. But when most instances are
    missing their artifact the cause is systemic (Docker down, a stale wheel that
    crashes every container on import, an OOM storm), so re-running would just
    burn compute on garbage — surface it loudly instead.
    """

    if not runs:
        raise RuntimeError("SWE-bench rollout produced no run directories.")
    incomplete = [run for run in runs if not run.ok]
    complete = len(runs) - len(incomplete)
    if complete / len(runs) >= min_complete_fraction:
        return
    raise RuntimeError(
        "SWE-bench rollout missing result.json artifacts for "
        f"{len(incomplete)}/{len(runs)} instances (below the "
        f"{min_complete_fraction:.0%} completion floor). This usually means "
        "Docker/container execution failed systemically before the agent "
        "produced a patch (daemon down, stale wheel, or out-of-memory). "
        "Per-instance causes:\n" + "\n".join(_describe_incomplete(incomplete))
    )


def _describe_incomplete(runs: Sequence[Run]) -> list[str]:
    """One ``ref: reason`` line per incomplete run, read from its failure.json.

    The container's exit code and logs are persisted to ``out/failure.json`` by
    the generic runner, so the systemic-failure error can name the likely cause
    instead of just listing missing paths.
    """

    lines = []
    for run in runs:
        failure_path = run.dir / "out" / "failure.json"
        reason = "no failure.json (container left no diagnostic)"
        if failure_path.is_file():
            try:
                diagnostic = json.loads(failure_path.read_text(encoding="utf-8"))
                reason = str(diagnostic.get("likely_reason") or reason)
            except (json.JSONDecodeError, OSError):
                reason = "unreadable failure.json"
        lines.append(f"  {run.ref}: {reason}")
    return lines


def seed_files(*, model: str, api_kind: str, base_url: str = "") -> dict[str, str]:
    """Return DGM's seed version files for an evolvable SWE-bench agent package."""

    provider: dict[str, Any] = {
        "api": api_kind,
        "model": model,
        "api_key_env": OPENAI_AUTH_ENV,
    }
    if base_url:
        provider["base_url"] = base_url
    files: dict[str, str] = {
        "README.md": "# Real SWE-bench self-evolving agent\n",
        "provider.json": json.dumps(provider, indent=2, sort_keys=True) + "\n",
    }
    for name, text in agent_pkg.default_agent_package().items():
        files[AGENT_PREFIX + name] = text
    return files


def package_files(version: Version) -> dict[str, str]:
    """Collect DGM's versioned agent package, or the neutral default."""

    out: dict[str, str] = {}
    for name in version.files():
        if name.startswith(AGENT_PREFIX):
            out[name[len(AGENT_PREFIX) :]] = version.read(name)
    return out or agent_pkg.default_agent_package()


def version_package_artifacts(version: Version) -> dict[str, bytes]:
    """Stage DGM's versioned package for the in-container SWE-bench agent."""

    payload = json.dumps(package_files(version), ensure_ascii=False)
    return {AGENT_PACKAGE_KEY: payload.encode("utf-8")}


def reward_from_result(result: Mapping[str, Any]) -> float:
    """Extract the DGM train reward from a SWE-bench ``result.json`` mapping."""

    agent_package = result.get("agent_package", {})
    if isinstance(agent_package, Mapping) and agent_package.get("used_fallback"):
        return -1.0
    if "resolved" in result:
        return 1.0 if bool(result.get("resolved")) else 0.0
    if "score" in result:
        return float(result.get("score") or 0.0)
    value = result.get("reward", 0.0)
    return float(value or 0.0)


def swebench_reward(run: Run) -> float:
    """Return DGM's scalar reward for one SWE-bench rollout run."""

    return reward_from_result(run.result)


def apply_eval_score(run: Run, eval_row: Mapping[str, Any]) -> None:
    path = run.dir / "out" / "result.json"
    result = dict(run.result)
    metrics = eval_row.get("metrics", {})
    metrics = metrics if isinstance(metrics, Mapping) else {}
    score = float(eval_row.get("score", 0.0) or 0.0)
    result.update(
        {
            "resolved": bool(metrics.get("resolved") or eval_row.get("passed")),
            "status": str(metrics.get("status") or eval_row.get("reason") or ""),
            "score": score,
            "reward": score,
        }
    )
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def grade_reuse_runs(
    runs: Sequence[Run],
    instances: Sequence[Mapping[str, Any]],
    *,
    dataset_name: str,
    model_name: str,
) -> None:
    """Grade completed DGM rollout runs through the official reuse helper."""

    from evals.swebench.evaluate_predictions import reuse_eval_row

    by_id = {str(instance.get("instance_id")): instance for instance in instances}
    for run in runs:
        # Skip incomplete runs: grading writes result.json, which would mark the
        # instance "measured" and block the re-roll a missing artifact should get.
        if not run.ok:
            continue
        instance = by_id.get(run.instance_id)
        if instance is None:
            continue
        row = reuse_eval_row(
            instance, run.result, dataset_name=dataset_name, model_name=model_name
        )
        apply_eval_score(run, row)


def make_scaffold_rollout(base_rollout, *, dataset_name: str, model_name: str):
    """Wrap DGM's base rollout to enforce artifacts and grade with official reuse.

    The evolved agent package reaches the container via ``version_artifacts``
    staging on the base rollout, so instances are passed through unchanged here.
    """

    def rollout(version: Version, slice_: Slice) -> Sequence[Run]:
        runs = base_rollout(version, slice_)
        ensure_rollout_artifacts(runs)
        grade_reuse_runs(
            runs, slice_.instances, dataset_name=dataset_name, model_name=model_name
        )
        return runs

    return rollout
