"""Host-side helpers for the OneMillion-Bench (`omb`) eval suite.

The OneMillion-Bench analog of ``evals/swebench/harness.py``: the host-side
pieces the suite and the run entry share — dataset loading, agent-visible
sanitization (drop the rubrics the judge will score against), dotenv loading,
and the generator/judge environment the container half reads.

OneMillion-Bench is a light, Docker-free suite, so there is no image build or
wheelhouse here; runs go through ``LocalProcessBackend`` (in-process) by default.
The grading logic itself is ported into the wheel under
``simple_agent_lab.evals.suites.onemillion.grading`` so it ships with the
container half; this module only prepares inputs and environment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# The OpenAI/Judge env-var names and the `.env` loader are owned by
# `simple_agent_lab.llm.env` (single source of truth); this host-side harness
# only forwards them into the container half. See ADR consolidate-provider-env.
from simple_agent_lab.llm.env import (
    API_KIND_ENV,
    JUDGE_API_KIND_ENV,
    JUDGE_AUTH_ENV,
    JUDGE_BASE_URL_ENV,
    JUDGE_MODEL_ENV,
    OPENAI_AUTH_ENV,
    OPENAI_BASE_URL_ENV,
    OPENAI_LOG_ID_ENV,
    OPENAI_MODEL_ENV,
    OPENAI_REASONING_EFFORT_ENV,
    OPENAI_SESSION_ID_ENV,
    REASONING_EFFORT_ENV,
)

# Re-exported so the run entry (`runs/run_onemillion_suite.py`) keeps calling
# `harness.load_dotenv`; the implementation is owned by `llm.env`.
from simple_agent_lab.llm.env import load_dotenv as load_dotenv  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]

# Where the downloaded OneMillion-Bench dataset is expected, and where per-run
# artifacts land (ignored by git except the committed README — see .gitignore).
DEFAULT_DATASET_DIR = ROOT / "datasets" / "OneMillion-Bench"
DEFAULT_RUN_ROOT = ROOT / "evals" / "out" / "onemillion"
DEFAULT_WORKDIR = "/workspace"

SUITE_NAME = "onemillion"

# Generator (the model under test) uses the shared OpenAI-compatible env
# contract; the judge uses the parallel JUDGE_* set (falling back to OPENAI_*).
# Both name sets and the loader come from `simple_agent_lab.llm.env` (imported
# above); only the suite-local passthrough groupings live here.
API_KIND_CHOICES = ("openai-chat", "openai-responses")

OPENAI_PASSTHROUGH_ENVS = (
    OPENAI_MODEL_ENV,
    OPENAI_AUTH_ENV,
    OPENAI_BASE_URL_ENV,
    OPENAI_SESSION_ID_ENV,
    OPENAI_LOG_ID_ENV,
    API_KIND_ENV,
    # Reasoning depth for the generator, so a run-profile can tune it (matches
    # the swebench / programbench passthrough). Read from the process env into
    # the run's provider_env, the source the in-process generator builds from.
    REASONING_EFFORT_ENV,
    OPENAI_REASONING_EFFORT_ENV,
)
JUDGE_PASSTHROUGH_ENVS = (
    JUDGE_MODEL_ENV,
    JUDGE_AUTH_ENV,
    JUDGE_BASE_URL_ENV,
    JUDGE_API_KIND_ENV,
)

# Gold / scoring fields the agent must never see (the judge scores against them).
PRIVATE_INSTANCE_FIELDS = {
    "rubrics",
    "Rubrics",
    "scores",
    "rubric_auto_score",
    "rubric_auto_vs_human",
    "judge_cot",
    "model_response",
}


# --------------------------------------------------------------------------- #
# Instance shaping + dataset loading
# --------------------------------------------------------------------------- #
def normalize_case(data: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize upstream field-name variants (Prompt/Rubrics/...)."""

    record = dict(data)
    if "Prompt" in record and "prompt" not in record:
        record["prompt"] = record["Prompt"]
    if "question" in record and "prompt" not in record:
        record["prompt"] = record["question"]
    if "Rubrics" in record and "rubrics" not in record:
        record["rubrics"] = record["Rubrics"]
    if "System_Prompt" in record and "system_prompt" not in record:
        record["system_prompt"] = record["System_Prompt"]
    return record


def instance_id_for(record: dict[str, Any], *, fallback: str = "") -> str:
    """Stable per-case id: ``case_<case_id>`` when present, else a fallback."""

    existing = record.get("instance_id")
    if existing:
        return str(existing)
    case_id = record.get("case_id")
    if case_id is not None and str(case_id) != "":
        return f"case_{case_id}"
    return fallback


def sanitized_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Drop rubric/scoring fields so the agent sees only the question."""

    return {
        str(key): value
        for key, value in instance.items()
        if str(key) not in PRIVATE_INSTANCE_FIELDS
    }


def _records_from_file(path: Path) -> list[dict[str, Any]]:
    """Read one JSON file into normalized case records (single object or list)."""

    parsed = json.loads(path.read_text(encoding="utf-8"))
    items = parsed if isinstance(parsed, list) else [parsed]
    records: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        record = normalize_case(item)
        fallback = path.stem if len(items) == 1 else f"{path.stem}_{idx}"
        record["instance_id"] = instance_id_for(record, fallback=fallback)
        records.append(record)
    return records


def load_dataset(path: str | Path, *, recursive: bool = True) -> list[dict[str, Any]]:
    """Load every case under ``path`` (a JSON file or a directory of them)."""

    target = Path(path)
    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(target.rglob("*.json") if recursive else target.glob("*.json"))
    else:
        raise SystemExit(f"Dataset path does not exist: {target}")

    records: list[dict[str, Any]] = []
    for file in files:
        records.extend(_records_from_file(file))
    if not records:
        raise SystemExit(f"No OneMillion-Bench cases found under {target}")
    return records


def load_case(path: str | Path, instance_id: str | None) -> dict[str, Any]:
    """Load one case by ``instance_id`` (or the first case when ``None``)."""

    records = load_dataset(path)
    if instance_id is None:
        return records[0]
    for record in records:
        if str(record.get("instance_id")) == instance_id:
            return record
    raise SystemExit(f"Case {instance_id!r} not found under {path}")


def eval_payload(instance: dict[str, Any]) -> dict[str, Any] | None:
    """Gold the judge needs (prompt + rubrics + any human scores), or ``None``."""

    rubrics = instance.get("rubrics") or instance.get("Rubrics")
    if not rubrics:
        return None
    return {
        "prompt": instance.get("prompt") or instance.get("Prompt") or "",
        "rubrics": list(rubrics),
        "human_scores": {},
        "case_id": instance.get("case_id"),
    }


# --------------------------------------------------------------------------- #
# Provider + judge environment, dotenv
# --------------------------------------------------------------------------- #
def container_environment(provider: str) -> dict[str, str]:
    """Generator + judge env passed to the run (fake/oracle need none)."""

    env: dict[str, str] = {}
    if provider != "openai":
        return env
    for name in (*OPENAI_PASSTHROUGH_ENVS, *JUDGE_PASSTHROUGH_ENVS):
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
    """Return the requested generator adapter API kind, defaulting via API_KIND."""

    api_kind = (value or os.environ.get(API_KIND_ENV) or "openai-chat").strip()
    if api_kind not in API_KIND_CHOICES:
        raise SystemExit(
            f"Unsupported API_KIND {api_kind!r}; expected one of: "
            + ", ".join(API_KIND_CHOICES)
        )
    return api_kind
