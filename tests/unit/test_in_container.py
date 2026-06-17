from __future__ import annotations

import unittest

from simple_agent_lab.evals.in_container import (
    LLM_TIMEOUT_ENV,
    _llm_timeout_from_env,
    provider_from_env,
    request_extra_from_env,
)
from simple_agent_lab.llm.env import (
    OPENAI_AUTH_ENV,
    OPENAI_LOG_ID_ENV,
    OPENAI_MODEL_ENV,
    OPENAI_SESSION_ID_ENV,
    REASONING_EFFORT_ENV,
)


class InContainerRequestExtraTest(unittest.TestCase):
    def test_llm_timeout_env_parses_positive_number(self) -> None:
        self.assertEqual(_llm_timeout_from_env({LLM_TIMEOUT_ENV: "600"}), 600.0)
        self.assertIsNone(_llm_timeout_from_env({}))

    def test_llm_timeout_env_rejects_invalid_value(self) -> None:
        with self.assertRaises(SystemExit):
            _llm_timeout_from_env({LLM_TIMEOUT_ENV: "0"})

    def test_reasoning_effort_env_becomes_provider_default_reasoning(self) -> None:
        provider = provider_from_env(
            kind="openai",
            api_kind="openai-responses",
            env={
                OPENAI_MODEL_ENV: "gpt-x",
                OPENAI_AUTH_ENV: "token",
                REASONING_EFFORT_ENV: "high",
            },
        )

        self.assertEqual(provider.default_reasoning, "high")
        self.assertEqual(provider.api, "openai-responses")

    def test_request_extra_contains_session_headers_only(self) -> None:
        extra = request_extra_from_env(
            env={
                OPENAI_SESSION_ID_ENV: "session-1",
                OPENAI_LOG_ID_ENV: "log-1",
                REASONING_EFFORT_ENV: "high",
            }
        )

        self.assertNotIn("reasoning", extra)
        self.assertEqual(extra["extra_headers"]["X-TT-logid"], "log-1")
        self.assertEqual(extra["extra_headers"]["extra"], '{"session_id":"session-1"}')


if __name__ == "__main__":
    unittest.main()
