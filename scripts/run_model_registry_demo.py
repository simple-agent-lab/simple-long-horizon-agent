"""Show how the model-alias registry resolves ``strong`` / ``fast`` from env.

Deterministic and credential-free: it feeds the registry explicit env dicts
(instead of the real environment) and prints how each alias resolves, so the
fallback behaviour is visible without a provider call.

    uv run python scripts/run_model_registry_demo.py

See ADR model-alias-registry.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_agent_lab.llm import ModelRegistry  # noqa: E402


def _show(label: str, env: dict[str, str]) -> None:
    registry = ModelRegistry.from_env(env=env)
    print(f"\n=== {label} ===")
    for line in sorted(f"  {k}={v}" for k, v in env.items()):
        print(line)
    for alias in registry.aliases():
        provider = registry.get(alias)
        print(
            f"  -> {alias:>6}: model={provider.model!r} api={provider.api} "
            f"key_env={provider.api_key_env!r} base_url={provider.base_url!r}"
        )


def main() -> None:
    # Single-model deployment: both aliases fall back to OPENAI_* — nothing to
    # configure for code that asks for "strong" or "fast".
    _show(
        "one model serves every alias (fallback to OPENAI_*)",
        {"OPENAI_MODEL": "big-model", "OPENAI_AUTH_TOKEN": "tok"},
    )

    # Peel `fast` onto a cheaper model while sharing the gateway token/base url.
    _show(
        "fast peeled onto a cheaper model, shared token",
        {
            "OPENAI_MODEL": "big-model",
            "OPENAI_AUTH_TOKEN": "tok",
            "OPENAI_BASE_URL": "https://gateway.invalid/v1",
            "FAST_MODEL": "small-model",
        },
    )

    # Fully independent endpoints per alias.
    _show(
        "independent endpoints per alias",
        {
            "STRONG_MODEL": "big-model",
            "STRONG_AUTH_TOKEN": "strong-tok",
            "FAST_MODEL": "small-model",
            "FAST_AUTH_TOKEN": "fast-tok",
            "FAST_API_KIND": "openai-responses",
        },
    )


if __name__ == "__main__":
    main()
