#!/usr/bin/env python3
"""Unified benchmark entry point — one CLI over every bench.

Built so a thin dashboard can drive everything by shelling out and reading
JSON. Subcommands:

    uv run python runs/run_bench.py list [--json]
    uv run python runs/run_bench.py setup [bench ...] [--oracle] [--json]
    uv run python runs/run_bench.py <bench> [bench args ...] [--json]
    uv run python runs/run_bench.py score <bench> [scorer args ...] [--json]
    uv run python runs/run_bench.py oracle <bench> [bench args ...] [--json]
    uv run python runs/run_bench.py all --manifest M.json [--parallel N]

`<bench>` is one of the registered names (see `list`). Per-bench flags are
exactly the bench's own (including `--profile`); run
`uv run python runs/run_bench.py <bench> -h` to see them. With `--json` a
single run prints one machine-readable result object to stdout (human logs go
to stderr), so the dashboard gets a clean contract.

`setup` probes the environment (Python/uv, `.env` + provider creds, Docker,
datasets) and, with `--oracle`, runs a cheap model-free oracle smoke where a
bench supports it — a fast "is my environment wired correctly?" check.

`score` reaches a bench's official scorer (SWE-bench / ProgramBench delegate to
their `evals/<suite>/evaluate_*.py`; the test run is already inline, so this is
the parity-grade host parse). A bench that grades inline (OneMillion) says so
and does nothing. `oracle` runs the gold/model-free reference solution — sugar
for the run path with `--provider oracle` — as a deterministic wiring check.

`all` runs every entry of a JSON manifest, each as an isolated subprocess, and
prints a combined JSON summary:

    {"runs": [{"bench": "...", "args": ["case_1", "--provider", "oracle"]}, ...],
     "parallel": 1}

The per-bench logic lives in internal modules (runs/_benches/<bench>.py); this
file imports their `run()` / `_build_parser()` and is the one supported entry.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(ROOT / "src"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _benches import onemillion as _onemillion  # noqa: E402
from _benches import programbench as _programbench  # noqa: E402
from _benches import swebench as _swebench  # noqa: E402
from evals.onemillion import harness as _omb_harness  # noqa: E402
from simple_agent_lab.evals import parse_with_profile  # noqa: E402


class Bench:
    """A registered benchmark: its module plus what its environment needs."""

    def __init__(
        self,
        module: Any,
        *,
        needs_docker: bool,
        dataset_dir: Path | None,
        oracle_smoke: bool,
    ) -> None:
        self.module = module
        self.name: str = module.NAME
        self.description: str = module.DESCRIPTION
        self.needs_docker = needs_docker
        self.dataset_dir = dataset_dir
        self.oracle_smoke = oracle_smoke


BENCHES: dict[str, Bench] = {
    b.name: b
    for b in (
        Bench(_swebench, needs_docker=True, dataset_dir=None, oracle_smoke=False),
        Bench(_programbench, needs_docker=True, dataset_dir=None, oracle_smoke=False),
        Bench(
            _onemillion,
            needs_docker=False,
            dataset_dir=_omb_harness.DEFAULT_DATASET_DIR,
            oracle_smoke=True,
        ),
    )
}


def _emit_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False))


def cmd_list(json_mode: bool) -> int:
    items = [
        {"name": b.name, "description": b.description, "needs_docker": b.needs_docker}
        for b in BENCHES.values()
    ]
    if json_mode:
        _emit_json({"benches": items})
    else:
        for it in items:
            print(f"{it['name']:22}  {it['description']}")
    return 0


def cmd_bench(name: str, rest: list[str], json_mode: bool) -> int:
    bench = BENCHES[name]
    parser = bench.module._build_parser()
    if not json_mode:
        outcome = bench.module.run(parse_with_profile(parser, argv=rest))
        return int(outcome.get("status_code", 0))
    # JSON mode: keep stdout clean for the result; human logs go to stderr.
    with redirect_stdout(sys.stderr):
        try:
            outcome = bench.module.run(parse_with_profile(parser, argv=rest))
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            outcome = {
                "bench": name,
                "status_code": code,
                "error": None if isinstance(exc.code, int) else str(exc.code),
            }
    _emit_json(outcome)
    return int(outcome.get("status_code", 0))


def cmd_score(name: str, rest: list[str], json_mode: bool) -> int:
    bench = BENCHES[name]
    scorer = getattr(bench.module, "SCORER", None)
    if not scorer:
        # No separate scorer: the bench grades itself during the run.
        detail = (
            f"{name} scores inline during the run (graded in-environment); "
            "there is no separate scorer to invoke."
        )
        if json_mode:
            _emit_json(
                {"bench": name, "action": "score", "inline": True, "detail": detail}
            )
        else:
            print(detail)
        return 0
    cmd = [sys.executable, *scorer, *rest]
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if json_mode:
        _emit_json(
            {
                "bench": name,
                "action": "score",
                "inline": False,
                "status_code": proc.returncode,
            }
        )
    return proc.returncode


def _provider_accepts_oracle(parser: argparse.ArgumentParser) -> bool:
    for action in parser._actions:
        if "--provider" in action.option_strings:
            return bool(action.choices) and "oracle" in action.choices
    return False


def cmd_oracle(name: str, rest: list[str], json_mode: bool) -> int:
    bench = BENCHES[name]
    if not _provider_accepts_oracle(bench.module._build_parser()):
        detail = (
            f"{name} has no oracle mode (no apply_oracle / --provider oracle). "
            "Oracle runs apply the reference solution model-free as a wiring check."
        )
        sys.stderr.write(detail + "\n")
        if json_mode:
            _emit_json(
                {"bench": name, "action": "oracle", "supported": False, "error": detail}
            )
        return 2
    # Oracle is the run path with the gold/model-free provider.
    return cmd_bench(name, ["--provider", "oracle", *rest], json_mode)


def _check(name: str, ok: bool, detail: str, *, warn_only: bool = False) -> dict:
    return {
        "check": name,
        "status": "ok" if ok else ("warn" if warn_only else "fail"),
        "detail": detail,
    }


def _provider_creds_present(env_path: Path) -> bool:
    import os

    keys = ("OPENAI_MODEL", "OPENAI_AUTH_TOKEN")
    if all(os.environ.get(k) for k in keys):
        return True
    if not env_path.exists():
        return False
    text = env_path.read_text(encoding="utf-8", errors="ignore")
    present = {
        line.split("=", 1)[0].strip()
        for line in text.splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    }
    return all(k in present for k in keys)


def _probe_docker() -> dict:
    try:
        import docker
    except Exception as exc:  # noqa: BLE001 — report, don't crash setup
        return _check("docker SDK", False, f"import failed: {exc}")
    try:
        client = docker.from_env(timeout=5)
        client.ping()
        return _check("docker daemon", True, "reachable")
    except Exception as exc:  # noqa: BLE001
        return _check("docker daemon", False, f"not reachable: {exc}")


def _probe_dataset(dataset_dir: Path) -> dict:
    ok = dataset_dir.exists() and any(dataset_dir.iterdir())
    detail = (
        str(dataset_dir)
        if ok
        else f"missing/empty: {dataset_dir} (download OneMillion-Bench)"
    )
    return _check(f"dataset ({dataset_dir.name})", ok, detail, warn_only=True)


def _oracle_smoke(bench: Bench) -> dict:
    """Run one model-free oracle case via subprocess; report what happened."""
    if not bench.oracle_smoke:
        return {"bench": bench.name, "ran": False, "detail": "no creds-free oracle"}
    if bench.dataset_dir is None or not bench.dataset_dir.exists():
        return {"bench": bench.name, "ran": False, "detail": "dataset not present"}
    try:
        instances = _omb_harness.load_dataset(bench.dataset_dir)
    except Exception as exc:  # noqa: BLE001
        return {"bench": bench.name, "ran": False, "detail": f"load failed: {exc}"}
    if not instances:
        return {"bench": bench.name, "ran": False, "detail": "dataset empty"}
    case_id = str(instances[0]["instance_id"])
    cmd = [
        sys.executable,
        str(HERE / "run_bench.py"),
        bench.name,
        "--json",
        case_id,
        "--provider",
        "oracle",
        "--no-scoring",
        "--dataset",
        str(bench.dataset_dir),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    outcome = _parse_last_json(proc.stdout)
    code = outcome.get("status_code", proc.returncode) if outcome else proc.returncode
    return {
        "bench": bench.name,
        "ran": True,
        "case": case_id,
        "status_code": code,
        "ok": code == 0,
        "detail": "oracle smoke passed" if code == 0 else (proc.stderr or "")[-500:],
    }


def cmd_setup(names: list[str], oracle: bool, json_mode: bool) -> int:
    selected = [BENCHES[n] for n in names] if names else list(BENCHES.values())
    env_path = ROOT / ".env"

    checks: list[dict] = [
        _check(
            "python",
            True,
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        ),
        _check(
            "uv", shutil.which("uv") is not None, shutil.which("uv") or "not on PATH"
        ),
        _check(
            ".env",
            env_path.exists(),
            str(env_path) if env_path.exists() else "missing (needed for real runs)",
            warn_only=True,
        ),
        _check(
            "provider creds",
            _provider_creds_present(env_path),
            "OPENAI_MODEL + OPENAI_AUTH_TOKEN present"
            if _provider_creds_present(env_path)
            else "missing — only oracle/fake runs will work",
            warn_only=True,
        ),
    ]

    docker_check = _probe_docker() if any(b.needs_docker for b in selected) else None
    if docker_check is not None:
        checks.append(docker_check)

    dataset_checks: dict[Path, dict] = {}
    bench_reports: list[dict] = []
    for b in selected:
        blockers: list[str] = []
        if (
            b.needs_docker
            and docker_check is not None
            and docker_check["status"] != "ok"
        ):
            blockers.append(docker_check["detail"])
        if b.dataset_dir is not None:
            ds = dataset_checks.get(b.dataset_dir)
            if ds is None:
                ds = _probe_dataset(b.dataset_dir)
                dataset_checks[b.dataset_dir] = ds
                checks.append(ds)
            if ds["status"] != "ok":
                blockers.append(ds["detail"])
        bench_reports.append(
            {
                "name": b.name,
                "needs_docker": b.needs_docker,
                "ready": not blockers,
                "blockers": blockers,
            }
        )

    report: dict[str, Any] = {
        "checks": checks,
        "benches": bench_reports,
        "ok": all(c["status"] != "fail" for c in checks),
    }
    if oracle:
        report["oracle"] = [_oracle_smoke(b) for b in selected]

    if json_mode:
        _emit_json(report)
    else:
        print("==> Environment probe")
        for c in checks:
            mark = {"ok": "✓", "warn": "!", "fail": "✗"}[c["status"]]
            print(f"  [{mark}] {c['check']}: {c['detail']}")
        print("==> Bench readiness")
        for r in report["benches"]:
            mark = "✓" if r["ready"] else "✗"
            extra = "" if r["ready"] else f" — {'; '.join(r['blockers'])}"
            print(f"  [{mark}] {r['name']}{extra}")
        for o in report.get("oracle", []):
            if o.get("ran"):
                mark = "✓" if o.get("ok") else "✗"
                print(
                    f"  [{mark}] oracle {o['bench']} ({o.get('case')}): {o['detail']}"
                )
            else:
                print(f"  [-] oracle {o['bench']}: {o['detail']}")
        print(f"==> {'OK' if report['ok'] else 'PROBLEMS FOUND'}")
    return 0 if report["ok"] else 1


def _parse_last_json(text: str) -> dict | None:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def _run_manifest_entry(entry: dict) -> dict:
    bench = entry.get("bench")
    if bench not in BENCHES:
        return {"bench": bench, "status_code": 2, "error": f"unknown bench {bench!r}"}
    argv = list(entry.get("args", []))
    if entry.get("profile"):
        argv = ["--profile", str(entry["profile"]), *argv]
    cmd = [sys.executable, str(HERE / "run_bench.py"), bench, "--json", *argv]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    sys.stderr.write(proc.stderr)
    outcome = _parse_last_json(proc.stdout)
    if outcome is None:
        return {
            "bench": bench,
            "status_code": proc.returncode or 1,
            "error": "no JSON result (run crashed); see stderr",
        }
    outcome.setdefault("status_code", proc.returncode)
    return outcome


def cmd_all(manifest_path: str, parallel: int, json_mode: bool) -> int:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    entries = [
        e
        for e in manifest.get("runs", [])
        if not str(e.get("bench", "")).startswith("_")
    ]
    if not entries:
        raise SystemExit(f"manifest {manifest_path} has no 'runs' entries")
    parallel = parallel or int(manifest.get("parallel", 1))

    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            outcomes = list(pool.map(_run_manifest_entry, entries))
    else:
        outcomes = [_run_manifest_entry(e) for e in entries]

    combined = {
        "runs": outcomes,
        "ok": all(o.get("status_code", 1) == 0 for o in outcomes),
    }
    # `all` is the dashboard's one-click: always emit the combined JSON.
    _emit_json(combined)
    if not json_mode:
        for o in outcomes:
            mark = "✓" if o.get("status_code") == 0 else "✗"
            sys.stderr.write(
                f"  [{mark}] {o.get('bench')}: status {o.get('status_code')}\n"
            )
    return 0 if combined["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or raw[0] in ("-h", "--help"):
        print(__doc__)
        print("benches:", ", ".join(BENCHES))
        return 0
    cmd, rest = raw[0], raw[1:]
    # A global flag the bench parsers don't know; peel it before delegating.
    json_mode = "--json" in rest
    rest = [a for a in rest if a != "--json"]

    if cmd == "list":
        return cmd_list(json_mode)
    if cmd == "setup":
        pre = argparse.ArgumentParser(prog="run_bench.py setup", add_help=True)
        pre.add_argument("benches", nargs="*")
        pre.add_argument("--oracle", action="store_true")
        ns = pre.parse_args(rest)
        names = list(ns.benches or [])
        unknown = [n for n in names if n not in BENCHES]
        if unknown:
            raise SystemExit(
                f"unknown bench(es): {unknown}; choose from {list(BENCHES)}"
            )
        return cmd_setup(names, ns.oracle, json_mode)
    if cmd == "all":
        pre = argparse.ArgumentParser(prog="run_bench.py all", add_help=True)
        pre.add_argument("--manifest", required=True)
        pre.add_argument("--parallel", type=int, default=0)
        ns = pre.parse_args(rest)
        return cmd_all(ns.manifest, ns.parallel, json_mode)
    if cmd in ("score", "oracle"):
        if not rest or rest[0] not in BENCHES:
            sys.stderr.write(
                f"usage: run_bench.py {cmd} <bench> [args]; "
                f"bench is one of {list(BENCHES)}\n"
            )
            return 2
        handler = cmd_score if cmd == "score" else cmd_oracle
        return handler(rest[0], rest[1:], json_mode)
    if cmd in BENCHES:
        return cmd_bench(cmd, rest, json_mode)

    sys.stderr.write(
        f"unknown command {cmd!r}. Use: "
        f"list | setup | score | oracle | all | {' | '.join(BENCHES)}\n"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
