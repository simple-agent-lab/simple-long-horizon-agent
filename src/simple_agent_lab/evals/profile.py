"""JSON run-profile: one file that bundles the two launch surfaces.

A run-profile is a small JSON document with two sections — `env` (the catalogued
environment knobs) and `run` (the `run_*_suite.py` long-option names) — so one
committed file names a runnable agent-on-a-bench arm instead of a remembered
mix of exports and CLI flags. See ADR run-profile-file.

The loader is deliberately suite-agnostic: it returns an env map and a run map
and never knows any suite's flag set. `apply_profile_env` fills env gaps (a real
export wins, exactly like `load_dotenv`), and `profile_run_argv` turns the run
map into an argv prefix the caller places *before* the real command line, so an
explicit CLI flag overrides the profile. JSON (not YAML/TOML) matches ADR
model-config-file: no new dependency, and a leading-`_` key is an ignored
comment so the example file documents itself.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_TOP_KEYS = frozenset({"env", "run"})
_SCALAR = (str, int, float, bool)


@dataclass(frozen=True)
class RunProfile:
    """A validated run-profile: environment knobs plus run-shape flags."""

    env: dict[str, str] = field(default_factory=dict)
    run: dict[str, Any] = field(default_factory=dict)


def _declared_keys(mapping: dict[str, Any]) -> set[str]:
    """Real keys; a leading-``_`` key is an ignored JSON comment (top + section)."""
    return {key for key in mapping if not key.startswith("_")}


def load_run_profile(
    path: str | Path,
    *,
    missing_exc: Callable[[str], BaseException] = SystemExit,
) -> RunProfile:
    """Read and validate a JSON run-profile file into a `RunProfile`.

    Strict shape: only ``env`` / ``run`` top-level keys (plus ``_`` comments),
    each a flat object of scalar values. Any other shape raises ``missing_exc``
    (``SystemExit`` by default, for a clean CLI message with no traceback).
    """
    config_path = Path(path)
    if not config_path.exists():
        raise missing_exc(f"Run-profile file not found: {config_path}")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise missing_exc(f"Invalid JSON in {config_path}: {exc}") from exc
    return _load_raw(raw, path=config_path, missing_exc=missing_exc)


def _load_raw(
    raw: Any, *, path: Path, missing_exc: Callable[[str], BaseException]
) -> RunProfile:
    if not isinstance(raw, dict):
        raise missing_exc(f"Run-profile {path} must be a JSON object.")
    unknown = _declared_keys(raw) - _TOP_KEYS
    if unknown:
        raise missing_exc(
            f"Unknown top-level keys in {path}: {sorted(unknown)}; "
            f"expected {sorted(_TOP_KEYS)}."
        )
    env = _scalar_section(raw.get("env", {}), section="env", path=path, exc=missing_exc)
    run = _scalar_section(raw.get("run", {}), section="run", path=path, exc=missing_exc)
    # env values are environment strings; coerce scalars (e.g. a JSON number) to
    # the string form the process environment requires.
    return RunProfile(
        env={key: _env_str(value) for key, value in env.items()},
        run=run,
    )


def _scalar_section(
    value: Any, *, section: str, path: Path, exc: Callable[[str], BaseException]
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise exc(f"'{section}' in {path} must be an object.")
    cleaned: dict[str, Any] = {}
    for key, item in value.items():
        if key.startswith("_"):
            continue
        if not isinstance(item, _SCALAR):
            raise exc(
                f"'{section}.{key}' in {path} must be a string, number, or "
                f"boolean; got {type(item).__name__}."
            )
        cleaned[key] = item
    return cleaned


def _env_str(value: Any) -> str:
    # JSON booleans would render "True"/"False"; emit lowercase so a profile's
    # `"FLAG": true` matches the "1"/"true" reading the env knobs expect.
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def apply_profile_env(
    env: dict[str, str], *, environ: MutableMapping[str, str] | None = None
) -> None:
    """Apply profile env vars without overriding existing values (fill-the-gaps).

    Identical to `load_dotenv` semantics: an already-exported value wins, so the
    profile is a default and an ad-hoc export stays the override.
    """
    # env-ok: the .env/profile loader writes into the process env
    target = os.environ if environ is None else environ
    for key, value in env.items():
        if key not in target:
            target[key] = value


def profile_run_argv(run: dict[str, Any]) -> list[str]:
    """Turn the run section into an argv prefix of ``--long-option`` tokens.

    A boolean ``true`` becomes a bare ``--flag`` (argparse ``store_true``); a
    boolean ``false`` is omitted. Everything else becomes ``--key value``. The
    caller places this *before* the real command line so explicit flags win.
    """
    argv: list[str] = []
    for key, value in run.items():
        option = f"--{key}"
        if isinstance(value, bool):
            if value:
                argv.append(option)
            continue
        argv.extend([option, str(value)])
    return argv


def parse_with_profile(
    parser: argparse.ArgumentParser,
    *,
    argv: Sequence[str] | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> argparse.Namespace:
    """Parse args, pre-seeding env + an argv prefix from a ``--profile`` file.

    `parser` is the run script's full parser (it should still declare ``--profile``
    for ``--help``). A first pass pulls ``--profile`` out; if present, its `env`
    fills env gaps and its `run` flags are injected *before* the remaining command
    line, so an explicit CLI flag always wins over the profile. Without
    ``--profile`` this is a plain `parser.parse_args`.
    """
    raw = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--profile")
    pre_args, remaining = pre.parse_known_args(raw)
    prefix: list[str] = []
    if pre_args.profile:
        profile = load_run_profile(pre_args.profile)
        apply_profile_env(profile.env, environ=environ)
        prefix = profile_run_argv(profile.run)
    return parser.parse_args(prefix + remaining)
