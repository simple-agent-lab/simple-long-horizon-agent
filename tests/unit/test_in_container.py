from __future__ import annotations

import unittest

from simple_agent_lab.evals.in_container import (
    OPENAI_LOG_ID_ENV,
    OPENAI_REASONING_EFFORT_ENV,
    OPENAI_SESSION_ID_ENV,
    request_extra_from_env,
)


class InContainerRequestExtraTest(unittest.TestCase):
    def test_reasoning_effort_env_becomes_request_extra(self) -> None:
        extra = request_extra_from_env(
            env={OPENAI_REASONING_EFFORT_ENV: "high"}
        )

        self.assertEqual(extra, {"reasoning": {"effort": "high"}})

    def test_headers_and_reasoning_effort_can_coexist(self) -> None:
        extra = request_extra_from_env(
            env={
                OPENAI_SESSION_ID_ENV: "session-1",
                OPENAI_LOG_ID_ENV: "log-1",
                OPENAI_REASONING_EFFORT_ENV: "high",
            }
        )

        self.assertEqual(extra["reasoning"], {"effort": "high"})
        self.assertEqual(extra["extra_headers"]["X-TT-logid"], "log-1")
        self.assertEqual(extra["extra_headers"]["extra"], '{"session_id":"session-1"}')


if __name__ == "__main__":
    unittest.main()
