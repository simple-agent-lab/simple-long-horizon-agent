from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import run_self_evolving_swebench
from simple_agent_lab.evolution.types import Run, Version


ROOT = Path(__file__).resolve().parents[2]


class RunSelfEvolvingSwebenchTest(unittest.TestCase):
    def test_seed_files_includes_agent_package(self) -> None:
        seed = run_self_evolving_swebench.seed_files(
            model="gpt-test",
            api_kind="openai-chat",
            base_url="https://example.test/v1",
        )

        self.assertTrue(any(name.startswith("agent/") for name in seed))
        self.assertIn("agent/agent_program.py", seed)
        self.assertIn("build_agent", seed["agent/agent_program.py"])
        self.assertIn("provider.json", seed)
        self.assertIn("README.md", seed)
        provider = json.loads(seed["provider.json"])
        self.assertEqual(provider["model"], "gpt-test")
        self.assertEqual(provider["api"], "openai-chat")
        self.assertEqual(provider["base_url"], "https://example.test/v1")

    def test_version_package_artifacts_contains_default(self) -> None:
        from simple_agent_lab.evals.protocols import AGENT_PACKAGE_KEY
        from simple_agent_lab.evolution.kernel import store
        from simple_agent_lab.evolution.types import Manifest

        with tempfile.TemporaryDirectory() as tmp:
            v = store.stage(
                Path(tmp),
                base=None,
                edits={"agent/agent_program.py": "def build_agent(): ...\n"},
                manifest=Manifest(producer="seed"),
            )
            art = run_self_evolving_swebench.version_package_artifacts(v)

        files = json.loads(art[AGENT_PACKAGE_KEY].decode("utf-8"))
        self.assertIn("agent_program.py", files)

    def test_package_files_falls_back_to_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            version_dir = Path(tmp) / "version"
            version_dir.mkdir()

            files = run_self_evolving_swebench.package_files(Version(version_dir))

        self.assertIn("agent_program.py", files)
        self.assertIn("build_agent", files["agent_program.py"])

    def test_pick_best_node_selects_highest_valid_reward(self) -> None:
        from simple_agent_lab.evolution import archive

        nodes = (
            archive.ArchiveNode(hash="a", scores={"reward": 0.5}),
            archive.ArchiveNode(hash="b", scores={"reward": 0.9}),
            archive.ArchiveNode(hash="c", scores={"reward": 0.7}),
            archive.ArchiveNode(hash="d", scores={"reward": 1.0}, valid_parent=False),
            archive.ArchiveNode(hash="e", scores={}),
        )
        best = run_self_evolving_swebench.pick_best_node(nodes)
        assert best is not None
        self.assertEqual(best.hash, "b")

    def test_pick_best_node_returns_none_when_empty(self) -> None:
        self.assertIsNone(run_self_evolving_swebench.pick_best_node(()))

    def test_swebench_reward_prefers_resolved_then_reward(self) -> None:
        self.assertEqual(
            run_self_evolving_swebench.reward_from_result({"resolved": True}), 1.0
        )
        self.assertEqual(
            run_self_evolving_swebench.reward_from_result({"resolved": False}), 0.0
        )
        self.assertEqual(
            run_self_evolving_swebench.reward_from_result({"reward": 0.5}), 0.5
        )
        self.assertEqual(
            run_self_evolving_swebench.reward_from_result({"score": 0.75}), 0.75
        )

    def test_apply_eval_score_updates_result_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run" / "case-1"
            out = run_dir / "out"
            out.mkdir(parents=True)
            (out / "result.json").write_text('{"model_patch": "diff"}\n')

            run_self_evolving_swebench.apply_eval_score(
                Run(run_dir),
                {
                    "passed": True,
                    "score": 1.0,
                    "reason": "resolved",
                    "metrics": {"resolved": True, "status": "resolved"},
                },
            )

            result = json.loads((out / "result.json").read_text())
        self.assertTrue(result["resolved"])
        self.assertEqual(result["reward"], 1.0)
        self.assertEqual(result["score"], 1.0)

    def test_heldout_run_id_uses_final_version_and_test_slice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            version_dir = Path(tmp) / "abc123"
            (version_dir / "scaffold").mkdir(parents=True)
            (version_dir / "scaffold" / "agent_scaffold.py").write_text(
                "def build_system_prompt(*, base_prompt):\n    return base_prompt\n",
                encoding="utf-8",
            )

            run_id = run_self_evolving_swebench.heldout_run_id(
                Version(version_dir), ({"instance_id": "test-1"},)
            )

        self.assertTrue(run_id.startswith("abc123-"))

    def test_ensure_rollout_artifacts_fails_when_result_json_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run" / "case-1"
            (run_dir / "out").mkdir(parents=True)

            with self.assertRaisesRegex(RuntimeError, "missing result.json"):
                run_self_evolving_swebench.ensure_rollout_artifacts([Run(run_dir)])

    def test_check_docker_available_wraps_daemon_errors(self) -> None:
        def broken_client():
            raise RuntimeError("socket missing")

        with self.assertRaisesRegex(SystemExit, "Docker is required"):
            run_self_evolving_swebench.check_docker_available(
                client_factory=broken_client
            )

    def test_cleanup_reset_containers_removes_swebench_containers_for_run_root(
        self,
    ) -> None:
        class FakeContainer:
            def __init__(self) -> None:
                self.name = "swebench.astropy__astropy-14369.3b89f9b34cec-9119f5eb01dc"
                self.removed = False

            def remove(self, force: bool = False) -> None:
                self.removed = force

        class FakeContainers:
            def __init__(self) -> None:
                self.container = FakeContainer()
                self.filters = []

            def list(self, *, all: bool = False, filters=None):
                self.all = all
                self.filters.append(filters)
                return [self.container]

        class FakeClient:
            def __init__(self) -> None:
                self.containers = FakeContainers()

        client = FakeClient()

        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "hyperagents-trend-8x4-g4"
            (run_root / "swebench_runs" / "3b89f9b34cec-9119f5eb01dc").mkdir(
                parents=True
            )

            removed = run_self_evolving_swebench.cleanup_reset_containers(
                run_root, client_factory=lambda: client
            )

        self.assertEqual(removed, 1)
        self.assertTrue(client.containers.all)
        self.assertEqual(
            client.containers.filters,
            [{"name": "3b89f9b34cec-9119f5eb01dc"}],
        )
        self.assertTrue(client.containers.container.removed)

    def test_load_dotenv_sets_missing_provider_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "OPENAI_MODEL=dotenv-model\n"
                "OPENAI_AUTH_TOKEN=dotenv-token\n"
                "OPENAI_BASE_URL=https://dotenv.test/v1\n",
                encoding="utf-8",
            )
            old_values = {
                key: os.environ.get(key)
                for key in ("OPENAI_MODEL", "OPENAI_AUTH_TOKEN", "OPENAI_BASE_URL")
            }
            for key in old_values:
                os.environ.pop(key, None)

            try:
                run_self_evolving_swebench.load_dotenv(env_path)
                self.assertEqual(os.environ["OPENAI_MODEL"], "dotenv-model")
                self.assertEqual(os.environ["OPENAI_AUTH_TOKEN"], "dotenv-token")
                self.assertEqual(
                    os.environ["OPENAI_BASE_URL"], "https://dotenv.test/v1"
                )
            finally:
                for key, value in old_values.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_wrapper_help_documents_real_execution_inputs(self) -> None:
        result = subprocess.run(
            ["bash", "runs/run_self_evolving_swebench.sh", "--help"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--train-dataset", result.stdout)
        self.assertIn("--test-dataset", result.stdout)
        self.assertIn("--execute", result.stdout)


if __name__ == "__main__":
    unittest.main()
