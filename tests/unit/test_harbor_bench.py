"""Tests for the Harbor benchmark wrapper."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from runs._benches import harbor


def _load_harbor_bench():
    return harbor


class HarborBenchTest(unittest.TestCase):
    def setUp(self) -> None:
        environment = patch.dict(os.environ, {}, clear=True)
        dotenv = patch.object(harbor, "load_dotenv")
        environment.start()
        dotenv.start()
        self.addCleanup(dotenv.stop)
        self.addCleanup(environment.stop)

    def test_dry_run_builds_harbor_run_command(self) -> None:
        harbor = _load_harbor_bench()
        parser = harbor._build_parser()
        args = parser.parse_args(
            [
                "--dataset",
                "futurehouse/demo",
                "--include-task-name",
                "abc*",
                "--n-tasks",
                "2",
                "--model",
                "openai/gpt-test",
                "--job-name",
                "sal-demo",
                "--jobs-dir",
                "evals/out/harbor/jobs",
                "--dry-run",
            ]
        )

        result = harbor.run(args)

        self.assertEqual(result["bench"], "harbor")
        self.assertEqual(result["status_code"], 0)
        command = result["command"]
        self.assertEqual(command[:2], ["harbor", "run"])
        self.assertIn("--agent", command)
        self.assertIn(
            "simple_long_horizon_agent.evals.harbor.agent:SimpleAgentLabHarborAgent",
            command,
        )
        self.assertIn("--dataset", command)
        self.assertIn("futurehouse/demo", command)
        self.assertIn("--include-task-name", command)
        self.assertIn("abc*", command)
        self.assertIn("--agent-kwarg", command)
        self.assertIn("api_kind=openai-responses", command)
        self.assertIn("max_turns=150", command)
        self.assertNotIn("harbor_exec", " ".join(command))

    def test_requires_one_dataset_source(self) -> None:
        harbor = _load_harbor_bench()
        parser = harbor._build_parser()
        args = parser.parse_args(["--dry-run"])

        with self.assertRaises(SystemExit):
            harbor.run(args)

    def test_api_kind_default_does_not_depend_on_environment(self) -> None:
        with patch.dict(os.environ, {"API_KIND": "openai-chat"}):
            harbor = _load_harbor_bench()
            parser = harbor._build_parser()
            args = parser.parse_args(["--dataset", "futurehouse/demo", "--dry-run"])

        result = harbor.run(args)

        self.assertIn("api_kind=openai-responses", result["command"])

    def test_setup_proxy_from_env_uses_private_agent_env_templates(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HTTP_PROXY": "http://proxy.example:8118",
                "HTTPS_PROXY": "http://proxy.example:8118",
                "NO_PROXY": "localhost,127.0.0.1",
            },
            clear=False,
        ):
            harbor = _load_harbor_bench()
            parser = harbor._build_parser()
            args = parser.parse_args(
                [
                    "--dataset",
                    "futurehouse/demo",
                    "--setup-proxy-from-env",
                    "--dry-run",
                ]
            )

            result = harbor.run(args)

        command = result["command"]
        self.assertIn("SAL_HARBOR_SETUP_HTTP_PROXY=${HTTP_PROXY}", command)
        self.assertIn("SAL_HARBOR_SETUP_HTTPS_PROXY=${HTTPS_PROXY}", command)
        self.assertIn("SAL_HARBOR_SETUP_NO_PROXY=${NO_PROXY}", command)
        self.assertNotIn("HTTP_PROXY=${HTTP_PROXY}", command)
        self.assertNotIn("HTTPS_PROXY=${HTTPS_PROXY}", command)
        self.assertNotIn("NO_PROXY=${NO_PROXY}", command)

    def test_setup_pip_index_defaults_to_public_pypi_private_env(self) -> None:
        harbor = _load_harbor_bench()
        parser = harbor._build_parser()
        args = parser.parse_args(["--dataset", "futurehouse/demo", "--dry-run"])

        result = harbor.run(args)

        command = result["command"]
        self.assertIn(
            "SAL_HARBOR_SETUP_PIP_INDEX_URL=https://pypi.org/simple",
            command,
        )
        self.assertNotIn("PIP_INDEX_URL=https://pypi.org/simple", command)

    def test_setup_pip_index_default_does_not_override_agent_env(self) -> None:
        harbor = _load_harbor_bench()
        parser = harbor._build_parser()
        args = parser.parse_args(
            [
                "--dataset",
                "futurehouse/demo",
                "--agent-env",
                "SAL_HARBOR_SETUP_PIP_INDEX_URL=https://mirror.example/simple",
                "--dry-run",
            ]
        )

        result = harbor.run(args)

        command = result["command"]
        self.assertIn(
            "SAL_HARBOR_SETUP_PIP_INDEX_URL=https://mirror.example/simple",
            command,
        )
        self.assertNotIn(
            "SAL_HARBOR_SETUP_PIP_INDEX_URL=https://pypi.org/simple",
            command,
        )
