from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest


def _install_harbor_stubs() -> None:
    harbor = types.ModuleType("harbor")
    agents = types.ModuleType("harbor.agents")
    installed = types.ModuleType("harbor.agents.installed")
    base = types.ModuleType("harbor.agents.installed.base")
    environments = types.ModuleType("harbor.environments")
    env_base = types.ModuleType("harbor.environments.base")
    models = types.ModuleType("harbor.models")
    agent_models = types.ModuleType("harbor.models.agent")
    context_mod = types.ModuleType("harbor.models.agent.context")

    class BaseInstalledAgent:
        def __init__(
            self,
            logs_dir: Path,
            *,
            extra_env: dict[str, str] | None = None,
            model_name: str = "openai/model-x",
        ) -> None:
            self.logs_dir = Path(logs_dir)
            self._extra_env = dict(extra_env or {})
            self.model_name = model_name
            self._parsed_model_name = (
                model_name.split("/", 1)[1] if "/" in model_name else None
            )

        def _get_env(self, key: str) -> str | None:
            return self._extra_env.get(key)

        async def exec_as_root(self, environment, command, env=None, cwd=None):
            raise NotImplementedError

        async def exec_as_agent(self, environment, command, env=None, cwd=None):
            raise NotImplementedError

    class BaseEnvironment:
        pass

    class AgentContext:
        n_input_tokens: int | None = None
        n_cache_tokens: int | None = None
        n_output_tokens: int | None = None
        cost_usd: float | None = None
        metadata: dict[str, object] | None = None

    def with_prompt_template(fn):
        return fn

    base.BaseInstalledAgent = BaseInstalledAgent
    base.with_prompt_template = with_prompt_template
    env_base.BaseEnvironment = BaseEnvironment
    context_mod.AgentContext = AgentContext

    sys.modules.update(
        {
            "harbor": harbor,
            "harbor.agents": agents,
            "harbor.agents.installed": installed,
            "harbor.agents.installed.base": base,
            "harbor.environments": environments,
            "harbor.environments.base": env_base,
            "harbor.models": models,
            "harbor.models.agent": agent_models,
            "harbor.models.agent.context": context_mod,
        }
    )


def _load_module():
    _install_harbor_stubs()
    sys.modules.pop("adapters.harbor_agent", None)
    return importlib.import_module("adapters.harbor_agent")


class HarborAgentTest(unittest.TestCase):
    def test_build_container_env_uses_extra_env_and_marathon_defaults(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            agent = module.SimpleAgentLab(
                Path(tmp),
                model_name="openai/gpt-5.4",
                extra_env={
                    "OPENAI_API_KEY": "host-key",
                    "OPENAI_BASE_URL": "https://gateway.example/v1",
                    "API_KIND": "openai-responses",
                    "OPENAI_REASONING_EFFORT": "low",
                    "OPENAI_SESSION_ID": "session-1",
                    "OPENAI_LOG_ID": "log-1",
                    "SAL_MAX_TURNS": "7",
                    "SAL_AGENT_FLAVOR": "bash_skills",
                    "SAL_WORKDIR": "/workspace/project",
                },
            )

            env = agent._build_container_env()

        self.assertEqual(env["OPENAI_MODEL"], "gpt-5.4")
        self.assertEqual(env["OPENAI_AUTH_TOKEN"], "host-key")
        self.assertEqual(env["OPENAI_BASE_URL"], "https://gateway.example/v1")
        self.assertEqual(env["API_KIND"], "openai-responses")
        self.assertEqual(env["REASONING_EFFORT"], "low")
        self.assertEqual(env["OPENAI_SESSION_ID"], "session-1")
        self.assertEqual(env["OPENAI_LOG_ID"], "log-1")
        self.assertEqual(env["SAL_MAX_TURNS"], "7")
        self.assertEqual(env["SAL_AGENT_FLAVOR"], "bash_skills")
        self.assertEqual(env["AGENT_FLAVOR"], "bash_skills")
        self.assertEqual(env["SAL_WORKDIR"], "/workspace/project")

    def test_build_container_env_defaults_to_bash_task(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            agent = module.SimpleAgentLab(
                Path(tmp),
                model_name="openai/gpt-5.4",
                extra_env={"OPENAI_AUTH_TOKEN": "tok"},
            )

            env = agent._build_container_env()

        self.assertEqual(env["SAL_AGENT_FLAVOR"], "bash_task")
        self.assertEqual(env["AGENT_FLAVOR"], "bash_task")
        self.assertEqual(env["SAL_MAX_TURNS"], str(module.MAX_TURNS))

    def test_container_script_detects_workdir_builds_flavor_and_writes_traces(
        self,
    ) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            agent = module.SimpleAgentLab(Path(tmp), model_name="openai/gpt-5.4")

            script = agent._container_script()
            command = agent._container_command()

        self.assertIn("def resolve_workdir()", script)
        self.assertIn("'/workspace/rust-java-lsp'", script)
        self.assertIn("'/app/rj-rust'", script)
        self.assertIn("build_flavor_agent(", script)
        self.assertIn("for _event in events:", script)
        self.assertIn("atif_trajectory_from_run(", script)
        self.assertIn("/logs/agent/sal/trajectory.jsonl", script)
        self.assertIn("/logs/agent/trajectory.json", script)
        self.assertIn("\nPY\n) 2>&1 | tee /logs/agent/sal.log", command)

    def test_install_uses_configured_source_without_unconditional_apt(self) -> None:
        module = _load_module()

        class RecordingAgent(module.SimpleAgentLab):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.commands: list[tuple[str, str, dict[str, str] | None]] = []

            async def exec_as_root(self, environment, command, env=None, cwd=None):
                self.commands.append(("root", command, env))

            async def exec_as_agent(self, environment, command, env=None, cwd=None):
                self.commands.append(("agent", command, env))

        with tempfile.TemporaryDirectory() as tmp:
            agent = RecordingAgent(
                Path(tmp),
                model_name="openai/gpt-5.4",
                extra_env={
                    "SIMPLE_AGENT_LAB_SOURCE": "/opt/simple-agent-lab/source",
                    "SIMPLE_AGENT_LAB_PIP_ARGS": "--no-index --find-links /wheels",
                },
            )

            asyncio.run(agent.install(object()))

        root_command = agent.commands[0][1]
        pip_command = agent.commands[1][1]
        self.assertIn("python3 -m pip --version", root_command)
        self.assertIn("apt-get update", root_command)
        self.assertIn("--no-index --find-links /wheels", pip_command)
        self.assertIn("/opt/simple-agent-lab/source", pip_command)

    def test_populate_context_post_run_reads_atif_metrics(self) -> None:
        module = _load_module()
        context_mod = sys.modules["harbor.models.agent.context"]
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp)
            (logs / "trajectory.json").write_text(
                json.dumps(
                    {
                        "final_metrics": {
                            "total_prompt_tokens": 12,
                            "total_cached_tokens": 3,
                            "total_completion_tokens": 4,
                            "total_cost_usd": 0.25,
                        },
                        "extra": {"sal_trace_path": "sal/trajectory.jsonl"},
                    }
                ),
                encoding="utf-8",
            )
            agent = module.SimpleAgentLab(logs, model_name="openai/gpt-5.4")
            context = context_mod.AgentContext()

            agent.populate_context_post_run(context)

        self.assertEqual(context.n_input_tokens, 12)
        self.assertEqual(context.n_cache_tokens, 3)
        self.assertEqual(context.n_output_tokens, 4)
        self.assertEqual(context.cost_usd, 0.25)
        self.assertEqual(context.metadata, {"sal_trace_path": "sal/trajectory.jsonl"})


if __name__ == "__main__":
    unittest.main()
