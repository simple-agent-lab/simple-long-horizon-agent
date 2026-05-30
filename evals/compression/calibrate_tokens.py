"""Empirically calibrate the char/token estimate against a real tokenizer.

`CHARS_PER_TOKEN = 4` is a guess, and the compression eval already showed it
is ~0.8x off for prose and ~2x off for structured/tool content — in opposite
directions, so no single constant fits both. This measures the *real* ratio
per content type so any future estimator tuning is grounded in data, not a
hand-picked number.

Method (differential, so the provider's fixed per-call template overhead
cancels): for each sample, real marginal tokens = tokens(baseline + sample) −
tokens(baseline). We sweep several sizes per content type and fit
`tokens = slope · chars` through them; the measured chars-per-token is
`1 / slope`. Token counts are deterministic (the tokenizer doesn't vary with
temperature, and our normalized input count is cache-invariant), so one
reading per input suffices.

The result is provider/tokenizer specific — it calibrates whatever
OPENAI_* model is configured, which is exactly the point: re-run it per
provider rather than trusting one baked-in constant.

    bash runs/run_compression_eval.sh   # (offline suite)
    uv run python -m evals.compression.calibrate_tokens   # (live calibration)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_agent_lab.context_view import CHARS_PER_TOKEN  # noqa: E402
from simple_agent_lab.messages import user_message  # noqa: E402
from simple_agent_lab.trajectory import write_jsonl  # noqa: E402

from evals.compression.provider import (  # noqa: E402
    build_provider_from_env,
    count_input_tokens,
    request_extra_from_env,
)

DEFAULT_OUT_DIR = ROOT / "evals/out/compression"

# Target sample sizes (characters). Big enough that the marginal token cost
# dominates measurement granularity, spread so the regression slope is stable.
SIZE_STEPS = (500, 1500, 3000, 6000)


# One representative "unit" per content type; repeated to hit each size step.
_UNITS: dict[str, str] = {
    "prose": (
        "The migration plan proceeds in three phases, each gated on a passing "
        "integration run before the next begins, so that a regression surfaces "
        "against the smallest possible change set. "
    ),
    "code": (
        "def resolve(node, seen):\n"
        "    if node.id in seen:\n"
        "        return None\n"
        "    seen.add(node.id)\n"
        "    return [resolve(c, seen) for c in node.children if c.active]\n\n"
    ),
    "json": (
        '{"tool":"bash","arguments":{"command":"psql -c \\"select * from users '
        'where active=true\\"","timeout_ms":30000},"id":"call_8f3a","meta":'
        '{"retries":0,"cache":false}}\n'
    ),
    "logs": (
        "2026-05-30T09:12:44.812Z INFO  worker[4] processed batch=8821 "
        "rows=4096 lag_ms=37 cache_hit=0.92 retries=0 status=ok\n"
    ),
}


@dataclass(frozen=True)
class CalibrationPoint:
    category: str
    chars: int
    real_tokens: int

    @property
    def chars_per_token(self) -> float:
        return self.chars / self.real_tokens if self.real_tokens else 0.0


def _sample(unit: str, target_chars: int) -> str:
    reps = max(1, target_chars // len(unit))
    return (unit * reps)[:target_chars]


def _slope_through_origin(xs: Sequence[int], ys: Sequence[int]) -> float:
    """Least-squares slope for y = slope·x (overhead already differenced out)."""
    num = sum(x * y for x, y in zip(xs, ys))
    den = sum(x * x for x in xs)
    return num / den if den else 0.0


def measure(provider, *, request_extra) -> list[CalibrationPoint]:
    overhead = count_input_tokens(provider, [], request_extra=request_extra)
    points: list[CalibrationPoint] = []
    for category, unit in _UNITS.items():
        for size in SIZE_STEPS:
            text = _sample(unit, size)
            total = count_input_tokens(
                provider,
                [user_message(text, target="probe")],
                request_extra=request_extra,
            )
            points.append(
                CalibrationPoint(
                    category=category,
                    chars=len(text),
                    real_tokens=max(1, total - overhead),
                )
            )
    return points


def summarize(points: list[CalibrationPoint]) -> dict[str, float]:
    """Measured chars-per-token per category (regression slope) + overall."""
    result: dict[str, float] = {}
    by_cat: dict[str, list[CalibrationPoint]] = {}
    for point in points:
        by_cat.setdefault(point.category, []).append(point)
    for category, pts in by_cat.items():
        slope = _slope_through_origin(
            [p.chars for p in pts], [p.real_tokens for p in pts]
        )
        result[category] = (1.0 / slope) if slope else 0.0
    overall_slope = _slope_through_origin(
        [p.chars for p in points], [p.real_tokens for p in points]
    )
    result["__overall__"] = (1.0 / overall_slope) if overall_slope else 0.0
    return result


def print_report(points: list[CalibrationPoint], cpt: dict[str, float]) -> None:
    print(f"\nCurrent heuristic: CHARS_PER_TOKEN = {CHARS_PER_TOKEN}\n")
    print(f"{'category':<10}{'chars':>8}{'tokens':>8}{'chars/tok':>11}")
    print("-" * 37)
    for point in points:
        print(
            f"{point.category:<10}{point.chars:>8}{point.real_tokens:>8}"
            f"{point.chars_per_token:>11.2f}"
        )
    print("-" * 37)
    header = f"vs {CHARS_PER_TOKEN} error"
    print(f"\n{'category':<12}{'measured c/t':>14}{header:>16}")
    print("-" * 42)
    for category, value in cpt.items():
        # Error factor if we kept the current default: a sample of N real tokens
        # is estimated as chars/default = N·(measured/default) tokens.
        error = (value / CHARS_PER_TOKEN) if value else 0.0
        label = "OVERALL" if category == "__overall__" else category
        print(f"{label:<12}{value:>14.2f}{error:>15.2f}x")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate char/token ratios.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args(argv)

    provider = build_provider_from_env()
    if provider is None:
        print(
            "OPENAI_MODEL/OPENAI_AUTH_TOKEN unset; calibration needs a live provider.",
            file=sys.stderr,
        )
        return 2

    print(f"Calibrating char/token against model {provider.model!r}...")
    points = measure(provider, request_extra=request_extra_from_env())
    cpt = summarize(points)
    print_report(points, cpt)

    run_id = args.run_id or _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = Path(args.out_dir) / run_id / "token_calibration.jsonl"
    write_jsonl(
        out_path,
        [
            {
                "type": "token_calibration",
                "model": provider.model,
                "current_chars_per_token": CHARS_PER_TOKEN,
                "measured_chars_per_token": cpt,
                "points": [
                    {
                        "category": p.category,
                        "chars": p.chars,
                        "real_tokens": p.real_tokens,
                        "chars_per_token": round(p.chars_per_token, 4),
                    }
                    for p in points
                ],
            }
        ],
    )
    print(f"\nWrote calibration to {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
