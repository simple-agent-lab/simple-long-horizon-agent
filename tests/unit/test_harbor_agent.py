"""Tests for the Harbor installed-agent adapter surface."""

from __future__ import annotations

import importlib
import asyncio
import logging
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Any


class _FakeExecResult:
    def __init__(self, return_code: int = 0, stdout: str = "", stderr: str = ""):
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr


class _FakeHarborEnvironment:
    default_user = "agent"

    def __init__(self, *, workdir: str | None = None) -> None:
        self.commands: list[dict[str, Any]] = []
        self.uploads: list[tuple[Path, str]] = []
        self._installed_check_count = 0
        self.task_env_config = type("TaskEnvConfig", (), {"workdir": workdir})()

    async def exec(
        self,
        *,
        command: str,
        user: str | int | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_sec: int | None = None,
    ) -> _FakeExecResult:
        self.commands.append(
            {
                "command": command,
                "user": user,
                "env": env,
                "cwd": cwd,
                "timeout_sec": timeout_sec,
            }
        )
        if (
            "import simple_agent_lab.evals.harbor.runner" in command
            and self._installed_check_count == 0
        ):
            self._installed_check_count += 1
            return _FakeExecResult(return_code=1)
        return _FakeExecResult()

    async def upload_file(self, *, source_path: Path, target_path: str) -> None:
        self.uploads.append((source_path, target_path))


class HarborAgentOptionalImportTest(unittest.TestCase):
    def test_module_import_without_harbor_dependency(self) -> None:
        module = importlib.import_module("simple_agent_lab.evals.harbor.agent")
        self.assertEqual(
            module.AGENT_IMPORT_PATH,
            "simple_agent_lab.evals.harbor.agent:SimpleAgentLabHarborAgent",
        )

    def test_run_command_uses_container_local_runner(self) -> None:
        module = importlib.import_module("simple_agent_lab.evals.harbor.agent")
        command = module.build_sal_runner_command(
            instruction_path="/logs/agent/instruction.txt",
            cwd="/workspace",
            max_turns=25,
            provider="openai",
            api_kind="openai-chat",
            agent_flavor="bash_task_read",
            trace_path="/logs/agent/sal-trajectory.jsonl",
            summary_path="/logs/agent/sal-summary.json",
        )

        self.assertIn("simple_agent_lab.evals.harbor.runner", command)
        self.assertIn("--instruction-file", command)
        self.assertIn("/logs/agent/instruction.txt", command)
        self.assertIn("--cwd", command)
        self.assertIn("/workspace", command)
        self.assertNotIn("harbor_exec", command)

    def test_default_install_timeout_allows_slow_dependency_downloads(self) -> None:
        module = importlib.import_module("simple_agent_lab.evals.harbor.agent")

        self.assertEqual(3000, module._DEFAULT_INSTALL_TIMEOUT_SEC)

    def test_system_dependencies_command_prefers_existing_python(self) -> None:
        module = importlib.import_module("simple_agent_lab.evals.harbor.agent")
        command = module.build_sal_system_dependencies_command()

        self.assertLess(command.index("python3 -m venv"), command.index("apt-get"))
        self.assertIn("apk add --no-cache", command)
        self.assertIn("yum install -y", command)
        self.assertIn("dnf install -y", command)
        self.assertIn("ca-certificates", command)
        self.assertIn("Warning: No known package manager found", command)
        self.assertNotIn("apt-get update -qq", command)

    def test_venv_command_prefers_system_python_before_uv_bootstrap(self) -> None:
        module = importlib.import_module("simple_agent_lab.evals.harbor.agent")
        command = module.build_sal_venv_command(
            venv_path="/opt/simple-agent-lab-venv",
            python_version="3.12",
        )

        self.assertLess(
            command.index("python3 -m venv"), command.index("uv python install")
        )
        self.assertIn("python3 -m venv", command)
        self.assertIn("curl -LsSf https://astral.sh/uv/install.sh | sh", command)

    def test_package_install_command_uses_pip_without_installing_uv(self) -> None:
        module = importlib.import_module("simple_agent_lab.evals.harbor.agent")
        command = module.build_sal_package_install_command(
            venv_path="/opt/simple-agent-lab-venv",
            install_target="/tmp/simple-agent-lab-src",
        )

        self.assertIn("if command -v uv", command)
        self.assertIn("/opt/simple-agent-lab-venv/bin/python -m pip install", command)
        self.assertIn("import simple_agent_lab.evals.harbor.runner", command)
        self.assertNotIn("astral.sh/uv", command)

    def test_install_uses_layered_bootstrap_sequence(self) -> None:
        module = importlib.import_module("simple_agent_lab.evals.harbor.agent")
        if module._HARBOR_IMPORT_ERROR is not None:
            self.skipTest("Harbor is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            (root / "src/simple_agent_lab").mkdir(parents=True)
            (root / "src/simple_agent_lab/__init__.py").write_text("", encoding="utf-8")
            (root / "pyproject.toml").write_text(
                '[project]\nname = "simple-agent-lab"\n',
                encoding="utf-8",
            )
            logs_dir = Path(tmp) / "logs"
            agent = module.SimpleAgentLabHarborAgent(
                logs_dir=logs_dir,
                sal_source=str(root),
            )
            environment = _FakeHarborEnvironment()

            previous_disable_level = logging.root.manager.disable
            logging.disable(logging.CRITICAL)
            try:
                asyncio.run(agent.install(environment))
            finally:
                logging.disable(previous_disable_level)

        commands = [entry["command"] for entry in environment.commands]
        self.assertIn("import simple_agent_lab.evals.harbor.runner", commands[0])
        self.assertTrue(any("python3 -m venv" in command for command in commands))
        self.assertTrue(any("python -m venv" in command for command in commands))
        self.assertTrue(
            any("apt-get update && apt-get install" in command for command in commands)
        )
        self.assertTrue(any(" -m pip install" in command for command in commands))
        self.assertFalse(any("apt-get update -qq" in command for command in commands))
        self.assertEqual(1, len(environment.uploads))

    def test_install_maps_setup_only_proxy_env_to_bootstrap_commands(self) -> None:
        module = importlib.import_module("simple_agent_lab.evals.harbor.agent")
        if module._HARBOR_IMPORT_ERROR is not None:
            self.skipTest("Harbor is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            (root / "src/simple_agent_lab").mkdir(parents=True)
            (root / "src/simple_agent_lab/__init__.py").write_text("", encoding="utf-8")
            (root / "pyproject.toml").write_text(
                '[project]\nname = "simple-agent-lab"\n',
                encoding="utf-8",
            )
            agent = module.SimpleAgentLabHarborAgent(
                logs_dir=Path(tmp) / "logs",
                sal_source=str(root),
                extra_env={
                    "SAL_HARBOR_SETUP_HTTP_PROXY": "http://proxy.example:8118",
                    "SAL_HARBOR_SETUP_no_proxy": "localhost,127.0.0.1",
                },
            )
            environment = _FakeHarborEnvironment()

            previous_disable_level = logging.root.manager.disable
            logging.disable(logging.CRITICAL)
            try:
                asyncio.run(agent.install(environment))
            finally:
                logging.disable(previous_disable_level)

        install_envs = [
            entry["env"]
            for entry in environment.commands
            if entry["env"] is not None
            and (
                "apt-get update" in entry["command"]
                or "python3 -m venv" in entry["command"]
                or " -m pip install" in entry["command"]
            )
        ]
        self.assertGreaterEqual(len(install_envs), 3)
        for env in install_envs:
            self.assertEqual("http://proxy.example:8118", env["HTTP_PROXY"])
            self.assertEqual("localhost,127.0.0.1", env["no_proxy"])
            self.assertNotIn("SAL_HARBOR_SETUP_HTTP_PROXY", env)

    def test_runner_env_does_not_include_setup_only_proxy_env(self) -> None:
        module = importlib.import_module("simple_agent_lab.evals.harbor.agent")
        if module._HARBOR_IMPORT_ERROR is not None:
            self.skipTest("Harbor is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            agent = module.SimpleAgentLabHarborAgent(
                logs_dir=Path(tmp) / "logs",
                extra_env={
                    "OPENAI_AUTH_TOKEN": "token",
                    "SAL_HARBOR_SETUP_HTTP_PROXY": "http://proxy.example:8118",
                },
            )

            env = agent._runner_env()

        self.assertEqual("token", env["OPENAI_AUTH_TOKEN"])
        self.assertNotIn("SAL_HARBOR_SETUP_HTTP_PROXY", env)
        self.assertNotIn("HTTP_PROXY", env)

    def test_source_archive_contains_only_installable_package_inputs(self) -> None:
        module = importlib.import_module("simple_agent_lab.evals.harbor.agent")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src/simple_agent_lab").mkdir(parents=True)
            (root / "src/simple_agent_lab/__init__.py").write_text("", encoding="utf-8")
            (root / "src/simple_agent_lab/core.py").write_text(
                "VALUE = 1\n", encoding="utf-8"
            )
            (root / "src/simple_agent_lab/__pycache__").mkdir()
            (
                root / "src/simple_agent_lab/__pycache__/core.cpython-312.pyc"
            ).write_bytes(b"compiled")
            (root / ".venv").mkdir()
            (root / ".venv/secret.txt").write_text("skip", encoding="utf-8")
            (root / "evals/out").mkdir(parents=True)
            (root / "evals/out/large.log").write_text("skip", encoding="utf-8")
            (root / "pyproject.toml").write_text(
                '[project]\nname = "simple-agent-lab"\n', encoding="utf-8"
            )
            (root / "README.md").write_text("# test\n", encoding="utf-8")
            (root / "LICENSE").write_text("license\n", encoding="utf-8")
            archive = root / "dist/source.tar.gz"

            module.build_sal_source_archive(root, archive)

            with tarfile.open(archive, "r:gz") as tar:
                names = set(tar.getnames())

        self.assertIn("pyproject.toml", names)
        self.assertIn("README.md", names)
        self.assertIn("LICENSE", names)
        self.assertIn("src/simple_agent_lab/__init__.py", names)
        self.assertIn("src/simple_agent_lab/core.py", names)
        self.assertFalse(any(name.startswith(".venv/") for name in names))
        self.assertFalse(any(name.startswith("evals/") for name in names))
        self.assertFalse(any("__pycache__" in name for name in names))

    def test_run_does_not_pass_relative_default_cwd_to_environment_exec(self) -> None:
        module = importlib.import_module("simple_agent_lab.evals.harbor.agent")
        if module._HARBOR_IMPORT_ERROR is not None:
            self.skipTest("Harbor is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            agent = module.SimpleAgentLabHarborAgent(logs_dir=Path(tmp) / "logs")
            environment = _FakeHarborEnvironment()
            context = type("Context", (), {"metadata": {}})()

            asyncio.run(agent.run("Do the task", environment, context))

        runner_commands = [
            entry
            for entry in environment.commands
            if "simple_agent_lab.evals.harbor.runner" in entry["command"]
        ]
        self.assertEqual(1, len(runner_commands))
        self.assertIsNone(runner_commands[0]["cwd"])
        self.assertIn("--cwd .", runner_commands[0]["command"])

    def test_run_passes_absolute_task_workdir_to_environment_exec(self) -> None:
        module = importlib.import_module("simple_agent_lab.evals.harbor.agent")
        if module._HARBOR_IMPORT_ERROR is not None:
            self.skipTest("Harbor is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            agent = module.SimpleAgentLabHarborAgent(logs_dir=Path(tmp) / "logs")
            environment = _FakeHarborEnvironment(workdir="/app/personal-site")
            context = type("Context", (), {"metadata": {}})()

            asyncio.run(agent.run("Do the task", environment, context))

        runner_commands = [
            entry
            for entry in environment.commands
            if "simple_agent_lab.evals.harbor.runner" in entry["command"]
        ]
        self.assertEqual(1, len(runner_commands))
        self.assertEqual("/app/personal-site", runner_commands[0]["cwd"])
        self.assertIn("--cwd /app/personal-site", runner_commands[0]["command"])


if __name__ == "__main__":
    unittest.main()
