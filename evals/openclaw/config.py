"""Model configuration for ClawEvalkit eval via Simple Agent Lab.

Loads model configs from ClawEvalkit's configs/models/ YAML files,
falling back to environment variables and sensible defaults.

SAL's Provider expects:
  - id: caller-chosen label
  - api: ApiKind ("fake" | "openai-chat" | "openai-responses" | "anthropic-messages")
  - model: provider model id
  - base_url: optional URL override
  - api_key_env: env var name for the API key

ClawEvalkit uses provider names like "openrouter", "glm", "minimax", etc.
We map these to SAL's ApiKind (most use "openai-chat" since they expose
OpenAI-compatible APIs).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for an LLM provider, compatible with SAL's Provider."""

    name: str  # caller-chosen label, e.g. "claude-sonnet"
    model: str  # provider model id, e.g. "anthropic/claude-sonnet-4.6"
    api_url: str  # API base URL
    api_key_env: str  # env var name for API key (SAL reads this)
    api_kind: str = "openai-chat"  # SAL ApiKind
    timeout: int = 300
    max_turns: int = 20


# ClawEvalkit provider name → SAL ApiKind mapping
# Most ClawEvalkit providers expose OpenAI-compatible APIs
CLAWEVALKIT_PROVIDER_TO_API = {
    "openrouter": "openai-chat",
    "glm": "openai-chat",
    "minimax": "openai-chat",
    "azure_openai": "azure_openai",  # needs AzureOpenAI client
    "dashscope": "openai-chat",
    "deepseek": "openai-chat",
    "ark": "openai-chat",
    "anthropic": "anthropic-messages",
    "fake": "fake",
}

# Provider → env var mapping for API keys
PROVIDER_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "glm": "GLM_API_KEY",
    "minimax": "MINIMAX_API_KEY",
    "azure_openai": "MODELHUB_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "ark": "ARK_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "fake": "",
}


def _try_load_clawevalkit_yaml(
    model_key: str, clawevalkit_dir: Path | None
) -> dict | None:
    """Try loading model config from ClawEvalkit's configs/models/ YAML."""
    if clawevalkit_dir is None:
        return None
    yaml_path = clawevalkit_dir / "configs" / "models" / f"{model_key}.yaml"
    if not yaml_path.exists():
        return None
    try:
        import yaml

        return yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _get_key_env(provider: str) -> str:
    """Get the env var name for an API key based on provider."""
    return PROVIDER_KEY_ENV.get(provider, "OPENROUTER_API_KEY")


def _get_api_kind(provider: str) -> str:
    """Map ClawEvalkit provider name to SAL ApiKind."""
    return CLAWEVALKIT_PROVIDER_TO_API.get(provider, "openai-chat")


def build_model_config(
    model_key: str,
    *,
    clawevalkit_dir: Path | None = None,
    api_url: str | None = None,
    api_key_env: str | None = None,
    provider: str | None = None,
    max_turns: int = 20,
    timeout: int = 300,
) -> ModelConfig:
    """Build a ModelConfig, trying ClawEvalkit YAML first, then env vars.

    Args:
        model_key: Model identifier (e.g. "claude-sonnet", "glm-4.7")
        clawevalkit_dir: Path to ClawEvalkit root (for loading YAML configs)
        api_url: Override API base URL
        api_key_env: Override API key env var name
        provider: Override ClawEvalkit provider name
        max_turns: Max agent turns per task
        timeout: Per-task timeout in seconds
    """
    cfg = _try_load_clawevalkit_yaml(model_key, clawevalkit_dir)

    if cfg:
        ce_provider = provider or cfg.get("provider", "openrouter")
        return ModelConfig(
            name=model_key,
            model=cfg.get("model", model_key),
            api_url=api_url or cfg.get("api_url", ""),
            api_key_env=api_key_env or _get_key_env(ce_provider),
            api_kind=_get_api_kind(ce_provider),
            max_turns=max_turns,
            timeout=timeout,
        )

    # Fallback: construct from provider arg or defaults
    ce_provider = provider or "openrouter"
    return ModelConfig(
        name=model_key,
        model=model_key,
        api_url=api_url or "https://openrouter.ai/api/v1",
        api_key_env=api_key_env or _get_key_env(ce_provider),
        api_kind=_get_api_kind(ce_provider),
        max_turns=max_turns,
        timeout=timeout,
    )
