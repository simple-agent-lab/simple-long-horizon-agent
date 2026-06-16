"""Show how the model registry resolves aliases — single-model and from a file.

Deterministic and credential-free: it feeds the registry an explicit env dict
(instead of the real environment) for the single-model path and a temp JSON
config for the multi-model path, and prints how each alias resolves, so both
surfaces are visible without a provider call.

    uv run python scripts/run_model_registry_demo.py

See ADRs model-alias-registry and model-config-file.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_agent_lab.llm import ModelRegistry  # noqa: E402


def _print_aliases(registry: ModelRegistry) -> None:
    for alias in registry.aliases():
        provider = registry.get(alias)
        print(
            f"  -> {alias:>6}: model={provider.model!r} api={provider.api} "
            f"key_env={provider.api_key_env!r} base_url={provider.base_url!r}"
        )


def main() -> None:
    # Single-model, main-compatible: no MODEL_CONFIG, so every default alias
    # resolves to the one OPENAI_* provider. Code asking for "strong"/"fast"
    # works with nothing but OPENAI_MODEL + OPENAI_AUTH_TOKEN.
    env = {
        "OPENAI_MODEL": "big-model",
        "OPENAI_AUTH_TOKEN": "tok",
        "OPENAI_BASE_URL": "https://gateway.invalid/v1",
    }
    print("\n=== single model: ModelRegistry.load() with OPENAI_* (no file) ===")
    for line in sorted(f"  {k}={v}" for k, v in env.items()):
        print(line)
    _print_aliases(ModelRegistry.load(env=env))

    # Multi-model: a JSON file with a shared `defaults` block plus per-alias
    # overrides. The file's aliases are whatever it names.
    config = {
        "defaults": {
            "base_url": "https://gateway.invalid/v1",
            "auth_token_env": "OPENAI_AUTH_TOKEN",
            "reasoning_effort": "medium",
        },
        "aliases": {
            "strong": {"model": "big-model", "api_kind": "anthropic-messages"},
            "fast": {"model": "small-model"},
            "judge": {"model": "grader-model", "auth_token_env": "JUDGE_AUTH_TOKEN"},
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "models.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        print("\n=== multi model: JSON config file (defaults + per-alias) ===")
        for line in path.read_text(encoding="utf-8").splitlines():
            print(f"  {line}")
        _print_aliases(ModelRegistry.from_file(path))


if __name__ == "__main__":
    main()
