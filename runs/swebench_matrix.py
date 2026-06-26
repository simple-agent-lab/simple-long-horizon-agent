"""Print a resolved-rate matrix across arm runs from the official Pro verdicts.

Reads each run's authoritative official verdict
(``evals/out/swebench_pro_official/<run-id>/reports/official/eval_results.json``,
an ``{instance_id: resolved_bool}`` map) and prints a per-case OK/fail matrix
plus per-column totals. Works for both whole-arm run-ids (one grader call per
arm) and per-case run-ids (``<base>__<instance>``) — pass a glob with ``*``.

Usage:
    uv run python runs/swebench_matrix.py \
        baseline=arms-20260622-215507-baseline \
        loop=arms-20260622-215507-loop \
        'loop_aligned=arms-20260622-212255-loop__*'
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

OFFICIAL = Path("evals/out/swebench_pro_official")


def verdicts(run_id_or_glob: str) -> dict[str, bool]:
    out: dict[str, bool] = {}
    pat = f"{run_id_or_glob}/reports/official/eval_results.json"
    for p in glob.glob(str(OFFICIAL / pat)):
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(d, dict):
            for iid, v in d.items():
                out[iid] = v is True or (
                    isinstance(v, dict)
                    and (v.get("resolved") or v.get("status") == "resolved")
                )
    return out


def short(iid: str) -> str:
    repo = iid.replace("instance_", "").split("__")[0][:12]
    tail = iid.split("-")[1][:8] if "-" in iid else ""
    return f"{repo:<12} {tail}"


def main() -> None:
    cols = [a for a in sys.argv[1:] if "=" in a]
    if not cols:
        raise SystemExit("pass label=run_id[ glob] columns")
    labels = [c.split("=", 1)[0] for c in cols]
    data = [verdicts(c.split("=", 1)[1]) for c in cols]
    ids = sorted({iid for d in data for iid in d})

    w = max(22, *(len(lbl) + 2 for lbl in labels))
    hdr = f"{'case':<22}" + "".join(f"{lbl:<{w}}" for lbl in labels)
    print(hdr)
    print("-" * len(hdr))
    for iid in ids:
        cells = "".join(
            f"{('OK' if d.get(iid) else ('fail' if iid in d else '-')):<{w}}"
            for d in data
        )
        print(f"{short(iid):<22}{cells}")
    print("-" * len(hdr))
    totals = "".join(
        f"{str(sum(1 for v in d.values() if v)) + '/' + str(len(d)):<{w}}" for d in data
    )
    print(f"{'resolved':<22}{totals}")


if __name__ == "__main__":
    main()
