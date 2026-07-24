"""Lint: environment knobs go through the config registry, not scattered reads.

A new behavioral env knob belongs in `simple_agent_lab.config` — one `EnvVar`
with one precedence rule. To keep knobs from scattering back into ad-hoc
`os.environ` / `os.getenv` reads across the package, this lint forbids direct
environment access in `src/simple_agent_lab/` except:

- in the designated env-owner modules (ALLOWLIST below) — the registry itself
  and the provider / model-alias env layer; and
- on a line carrying an inline ``# env-ok: <reason>`` marker, for a genuine
  one-off infra read (process-env scrubbing, a host->container handoff, the
  `.env`/profile loader, an optional-mapping fallback) that is not a behavioral
  config knob.

So the only way to add a new env *knob* is to declare it in the registry; any
other environment touch must be a consciously-marked exception.

Run: uv run python -m scripts.env_lint
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "simple_agent_lab"

# Modules whose job IS to read the environment (paths relative to SRC). These
# are the documented env owners (see docs/agent-native/configuration.md's
# boundary rule); a knob lives with exactly one of them.
ALLOWLIST = {
    "config.py",  # the env registry itself (EnvVar.get)
    "agent_flavors.py",  # owns the AGENT_FLAVOR name + flavor_from_env
    "model_metadata.py",  # owns the price-book / context-window override env
    "llm/env.py",  # provider / credentials / reasoning env (the owner)
    "llm/config.py",  # model-alias registry (<ALIAS>_* env)
    "llm/registry.py",  # model-alias resolution
}

# Direct global-environment access. A passed-in `environ`/`env` Mapping (the
# good pattern — `EnvVar.get(environ=...)`, harness passthrough) is NOT matched.
ACCESS = re.compile(r"\bos\.(environ|getenv)\b")
MARKER = "env-ok:"


def main() -> int:
    violations: list[tuple[Path, int, str]] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel in ALLOWLIST:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for lineno, line in enumerate(lines, start=1):
            if not ACCESS.search(line) or MARKER in line:
                continue
            # A long statement can carry the marker on the line directly above.
            prev = lines[lineno - 2].strip() if lineno >= 2 else ""
            if prev.startswith("#") and MARKER in prev:
                continue
            violations.append((path.relative_to(ROOT), lineno, line.strip()))

    if violations:
        print("env_lint: direct environment access outside the config registry.")
        print(
            "Declare the knob in `simple_agent_lab.config` (an EnvVar), or — for "
            "a genuine\none-off infra read — add an inline `# env-ok: <reason>`.\n"
        )
        for rel, lineno, snippet in violations:
            print(f"  {rel}:{lineno}: {snippet}")
        return 1

    print(
        "env_lint: all environment access goes through the config registry "
        "(or is marked env-ok)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
