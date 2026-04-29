"""Provider boundary for the event-driven loop."""

from __future__ import annotations

import os

from dataclasses import dataclass

from core import Message, MetaMode, ModelClient, model_messages


@dataclass(frozen=True)
class ModelConfig:
    provider: str = "fake"
    model: str = "fake-model"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("provider must be non-empty")
        if not self.model:
            raise ValueError("model must be non-empty")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")


@dataclass(frozen=True)
class FakeModel(ModelClient):
    name: str = "fake-model"

    def generate(self, messages: list[Message], meta: MetaMode = "none") -> Message:
        payload = model_messages(messages, meta=meta)
        assistant_turns = [
            message for message in messages if message.role == "assistant"
        ]
        if not assistant_turns:
            return Message(
                "assistant",
                f"{self.name} generated a first response from {len(payload)} message(s).",
                sender=self.name,
                kind="thought",
                data={"provider": "fake", "payload": payload},
            )
        return Message(
            "assistant",
            "Final: runtime events expose every loop transition.",
            sender=self.name,
            kind="final",
            data={"provider": "fake", "payload": payload},
        )


class OpenAIResponsesClient(ModelClient):
    def __init__(self, config: ModelConfig):
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing API key env var: {config.api_key_env}")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("install openai to use OpenAIResponsesClient") from exc

        self.config = config
        self.client = OpenAI(api_key=api_key, base_url=config.base_url)

    def generate(self, messages: list[Message], meta: MetaMode = "none") -> Message:
        response = self.client.responses.create(
            model=self.config.model,
            input=model_messages(messages, meta=meta),
            temperature=self.config.temperature,
        )
        return Message(
            "assistant",
            response.output_text,
            sender=self.config.model,
            kind="final",
            data={
                "provider": "openai",
                "response_id": response.id,
            },
        )


def load_model(config: ModelConfig) -> ModelClient:
    if config.provider == "fake":
        return FakeModel(config.model)
    if config.provider == "openai":
        return OpenAIResponsesClient(config)
    raise ValueError(f"unknown provider {config.provider!r}")
