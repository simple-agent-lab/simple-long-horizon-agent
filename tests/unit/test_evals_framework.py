"""Unit-smoke for the generic containerized eval framework (ADR 0017).

No Docker. Exercises the two seams (`ContainerBackend`, `ArtifactStore`) with
the in-memory `FakeBackend`, the `LocalDirStore` (bind-mount) and the
batteries-included `HostHttpStore` over real loopback HTTP, and drives the
SWE-bench container half through the generic runner with the fake provider.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

from simple_agent_lab.evals import (
    INSTANCE_KEY,
    RESULT_KEY,
    TRACE_KEY,
    ContainerPlan,
    FakeBackend,
    HostHttpStore,
    HttpArtifactClient,
    LocalDirStore,
    Suite,
    bootstrap_script,
    run_suite_instance,
)
from simple_agent_lab.evals.backends.fake import FakeContainerHandle

SWEBENCH_CONTAINER = "simple_agent_lab.evals.suites.swebench.container"


class _DemoSuite:
    """Minimal suite: the entire host-side surface for a new benchmark."""

    name = "demo"
    container_module = "demo.container"

    def container_plan(self, instance: Mapping[str, Any]) -> ContainerPlan:
        return ContainerPlan(image="demo:latest", workdir="/work")

    def sanitize_instance(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in instance.items() if k != "gold"}

    def prediction_record(
        self, instance: Mapping[str, Any], *, model_name: str, result: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "instance_id": str(instance["instance_id"]),
            "model_name_or_path": model_name,
            "answer": result.get("answer", ""),
        }


def _fake_container_writing_result(answer: str):
    """An on_start that reconstructs the container-side store and writes outputs.

    The fake "container" runs in this host process, so for a bind-mounted
    `LocalDirStore` it follows the mount back to the host source (the container
    path ``/agent/run`` only exists inside a real container). The HTTP store
    needs no remap — it reaches the host server over loopback.
    """

    def on_start(handle: FakeContainerHandle) -> None:
        from simple_agent_lab.evals.stores import container_store_from_env

        if handle.env.get("SAL_STORE") == "localdir":
            mount = handle.env["SAL_STORE_ROOT"]
            host_root = next(
                k for k, v in handle.mounts.items() if v.get("bind") == mount
            )
            store = LocalDirStore(host_root)
        else:
            store = container_store_from_env(handle.env)
        instance = json.loads(store.get(INSTANCE_KEY).decode("utf-8"))
        store.put(
            TRACE_KEY,
            (json.dumps({"trace_id": instance["instance_id"]}) + "\n").encode(),
        )
        store.put(RESULT_KEY, (json.dumps({"answer": answer}) + "\n").encode())

    return on_start


class EvalFrameworkSmokeTest(unittest.TestCase):
    def test_demo_suite_satisfies_protocol(self) -> None:
        self.assertIsInstance(_DemoSuite(), Suite)

    def test_run_suite_instance_local_dir_store(self) -> None:
        suite = _DemoSuite()
        instance = {"instance_id": "demo-1", "problem": "p", "gold": "SECRET"}

        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp).resolve()
            backend = FakeBackend(
                on_start=_fake_container_writing_result("42"), log_text="ok\n"
            )
            artifacts = run_suite_instance(
                suite=suite,
                instance=instance,
                backend=backend,
                store=LocalDirStore(run_root),
                run_root=run_root,
                run_id="run-x",
                model_name="m",
                provider_env={"OPENAI_MODEL": "m"},
            )

            handle = backend.created[0]
            self.assertTrue(handle.started)
            self.assertTrue(handle.removed)
            self.assertEqual(artifacts.status_code, 0)

            # The framework built the command itself (python -m the generic runner).
            self.assertIn("simple_agent_lab.evals.in_container", handle.command[-1])

            # Sanitized instance.json was written through the store (gold dropped).
            written = json.loads(
                (artifacts.run_dir / "input" / "instance.json").read_text()
            )
            self.assertNotIn("gold", written)

            # Live trace + result landed; host shaped prediction.jsonl from result.
            self.assertTrue(artifacts.trajectory_path.exists())
            prediction = json.loads(artifacts.prediction_path.read_text())
            self.assertEqual(prediction["answer"], "42")
            self.assertEqual(prediction["model_name_or_path"], "m")

    def test_run_suite_instance_host_http_store(self) -> None:
        """The batteries-included store works over real loopback HTTP, no Docker."""

        suite = _DemoSuite()
        instance = {"instance_id": "demo-2", "problem": "p"}

        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp).resolve()
            backend = FakeBackend(on_start=_fake_container_writing_result("99"))
            # container_host=127.0.0.1 so the in-process fake container can reach it.
            with HostHttpStore(run_root, container_host="127.0.0.1") as store:
                artifacts = run_suite_instance(
                    suite=suite,
                    instance=instance,
                    backend=backend,
                    store=store,
                    run_root=run_root,
                    run_id="run-http",
                )
                # The container got an HTTP store binding, not a bind mount.
                self.assertEqual(backend.created[0].env["SAL_STORE"], "http")
                prediction = json.loads(artifacts.prediction_path.read_text())
                self.assertEqual(prediction["answer"], "99")

    def test_bootstrap_script_is_suite_agnostic(self) -> None:
        script = bootstrap_script(
            runner_argv=("-m", "simple_agent_lab.evals.in_container", "--x", "a b"),
            wheelhouse_mount="/agent/wheelhouse",
        )
        self.assertIn("AGENT_PYTHON", script)
        self.assertIn("--no-index --find-links /agent/wheelhouse", script)
        self.assertIn("'a b'", script)  # spaced arg stays one token


class HostHttpStoreTest(unittest.TestCase):
    def test_http_client_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            with HostHttpStore(base, container_host="127.0.0.1") as store:
                bound = store.bind(base / "run-1" / "inst-1")
                binding = bound.container_binding()
                client = HttpArtifactClient(
                    binding.env["SAL_STORE_URL"], binding.env["SAL_STORE_TOKEN"]
                )
                client.put("out/result.json", b'{"ok": true}')
                # Host reads the same bytes off disk; client reads them over HTTP.
                self.assertEqual(bound.get("out/result.json"), b'{"ok": true}')
                self.assertEqual(client.get("out/result.json"), b'{"ok": true}')
                self.assertFalse(client.exists("out/missing.json"))

    def test_bad_token_is_rejected(self) -> None:
        import urllib.error

        with tempfile.TemporaryDirectory() as tmp:
            with HostHttpStore(Path(tmp), container_host="127.0.0.1") as store:
                url = store.bind(Path(tmp)).container_binding().env["SAL_STORE_URL"]
                client = HttpArtifactClient(url, "wrong-token")
                with self.assertRaises(urllib.error.HTTPError):
                    client.put("x", b"y")


class SwebenchSuiteDriverTest(unittest.TestCase):
    def test_pro_instance_plan_is_data(self) -> None:
        from evals.swebench.suite import SwebenchSuite

        suite = SwebenchSuite(dataset_name="SWE-bench_Pro")
        self.assertIsInstance(suite, Suite)
        self.assertEqual(suite.container_module, SWEBENCH_CONTAINER)
        instance = {
            "instance_id": "instance_acme__widget-abc123",
            "repo": "acme/widget",
            "dockerhub_tag": "acme.widget-abc123",
        }
        plan = suite.container_plan(instance)
        self.assertEqual(plan.workdir, "/app")
        self.assertEqual(plan.shell, ("/bin/sh", "-lc"))
        self.assertEqual(plan.entrypoint, "")
        self.assertIn("sweap-images", plan.image)


class GenericInContainerRunnerTest(unittest.TestCase):
    """Ideal state: the SWE-bench container half driven by the generic runner."""

    def test_swebench_container_half_through_generic_runner(self) -> None:
        from simple_agent_lab.evals.in_container import run_in_container
        from simple_agent_lab.llm import Provider

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "testbed"
            repo.mkdir()

            def git(*args: str) -> None:
                subprocess.run(
                    ["git", *args], cwd=repo, check=True, capture_output=True
                )

            git("init")
            git("config", "user.email", "t@example.invalid")
            git("config", "user.name", "T")
            git("config", "commit.gpgsign", "false")
            (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
            git("add", "-A")
            git("commit", "-m", "base")
            base = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                check=True,
                capture_output=True,
                encoding="utf-8",
            ).stdout.strip()

            instance = {
                "instance_id": "demo__repo-1",
                "base_commit": base,
                "problem_statement": "Make it better.",
                "language": "python",
            }
            store = LocalDirStore(Path(tmp)).bind(Path(tmp))
            result, state = run_in_container(
                instance=instance,
                container_module=SWEBENCH_CONTAINER,
                provider=Provider(id="fake", api="fake", model="fake-model"),
                workdir=repo,
                max_turns=3,
                store=store,
                trace_id="swebench.demo__repo-1",
                producer="suite:swebench",
                suite_name="swebench",
            )

            # extract_result product returned, loop ran, result + trace persisted.
            self.assertIn("model_patch", result)
            self.assertEqual(state.data["result"], result)
            persisted = json.loads(store.get(RESULT_KEY).decode("utf-8"))
            self.assertEqual(persisted, result)
            record = json.loads(store.get(TRACE_KEY).decode("utf-8"))
            self.assertEqual(record["meta"]["suite"], "swebench")
            self.assertFalse(record["meta"]["in_progress"])


if __name__ == "__main__":
    unittest.main()
