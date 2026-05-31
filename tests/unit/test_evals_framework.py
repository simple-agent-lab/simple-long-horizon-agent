"""Unit-smoke for the generic eval framework (ADR 0017).

No Docker. Covers the two seams — `ContainerBackend` (`FakeBackend` for
orchestration, `LocalProcessBackend` for a real in-process agent run) and
`ArtifactStore` (`LocalDirStore`, plus `HostHttpStore` over real loopback HTTP)
— and the unified `run_suite_instance` entry point that runs identically in or
out of a container by swapping the backend.
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
    LocalProcessBackend,
    RunSpec,
    Suite,
    build_command,
    run_suite_instance,
)

SWEBENCH_CONTAINER = "simple_agent_lab.evals.suites.swebench.container"


class _DemoSuite:
    """Minimal host-side surface for a new benchmark."""

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


def _simulate(answer: str):
    """A FakeBackend `on_run` that writes outputs through the bound store."""

    def on_run(spec: RunSpec, store) -> None:
        json.loads(store.get(INSTANCE_KEY).decode("utf-8"))  # container reads input
        store.put(TRACE_KEY, b'{"meta": {"suite": "demo"}}\n')
        store.put(RESULT_KEY, (json.dumps({"answer": answer}) + "\n").encode("utf-8"))

    return on_run


class OrchestrationTest(unittest.TestCase):
    def test_demo_suite_satisfies_protocol(self) -> None:
        self.assertIsInstance(_DemoSuite(), Suite)

    def test_run_suite_instance_fake_backend(self) -> None:
        instance = {"instance_id": "demo-1", "problem": "p", "gold": "SECRET"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            backend = FakeBackend(on_run=_simulate("42"), log_text="ok\n")
            artifacts = run_suite_instance(
                suite=_DemoSuite(),
                instance=instance,
                backend=backend,
                store=LocalDirStore(root),
                run_root=root,
                run_id="run-x",
                model_name="m",
                provider="fake",
            )
            # The backend received a structured spec, not a shell command.
            self.assertEqual(backend.runs[0].suite_name, "demo")
            self.assertEqual(backend.runs[0].instance_id, "demo-1")
            self.assertEqual(artifacts.logs, "ok\n")

            written = json.loads(
                (artifacts.run_dir / "input" / "instance.json").read_text()
            )
            self.assertNotIn("gold", written)  # sanitized through the store

            prediction = json.loads(artifacts.prediction_path.read_text())
            self.assertEqual(prediction["answer"], "42")
            self.assertEqual(prediction["model_name_or_path"], "m")

    def test_build_command_targets_the_generic_runner(self) -> None:
        spec = RunSpec(
            suite_name="s",
            container_module="m",
            instance_id="i",
            plan=ContainerPlan(image="img", workdir="/w"),
            max_turns=5,
            provider="fake",
            api_kind="openai-chat",
            wheelhouse_mount="/wh",
            run_name="n",
        )
        cmd = build_command(spec)
        self.assertEqual(cmd[:2], ("bash", "-lc"))
        self.assertIn("simple_agent_lab.evals.in_container", cmd[-1])
        self.assertIn("--find-links /wh", cmd[-1])


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


class _SwebenchLikeSuite:
    """Reuses the real SWE-bench container half; trivial host half (no swebench dep)."""

    name = "swebench"
    container_module = SWEBENCH_CONTAINER

    def container_plan(self, instance: Mapping[str, Any]) -> ContainerPlan:
        return ContainerPlan(image="(in-process)", workdir="(in-process)")

    def sanitize_instance(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        return dict(instance)

    def prediction_record(
        self, instance: Mapping[str, Any], *, model_name: str, result: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "instance_id": str(instance["instance_id"]),
            "model_name_or_path": model_name,
            "model_patch": result.get("model_patch", ""),
        }


class LocalProcessBackendTest(unittest.TestCase):
    """Unified entry point runs a real agent in-process — no Docker, no network."""

    def test_run_suite_instance_in_process(self) -> None:
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

            root = Path(tmp) / "runs"
            store = LocalDirStore(root)
            instance = {
                "instance_id": "demo__repo-1",
                "problem_statement": "Make it better.",
                "language": "python",
            }
            artifacts = run_suite_instance(
                suite=_SwebenchLikeSuite(),
                instance=instance,
                backend=LocalProcessBackend(workspace=repo),
                store=store,
                run_root=root,
                run_id="r",
                provider="fake",
                max_turns=3,
            )

            self.assertEqual(artifacts.status_code, 0)
            bound = store.bind(artifacts.run_dir)
            result = json.loads(bound.get(RESULT_KEY).decode("utf-8"))
            self.assertIn("model_patch", result)
            trace = json.loads(bound.get(TRACE_KEY).decode("utf-8"))
            self.assertEqual(trace["meta"]["suite"], "swebench")
            self.assertFalse(trace["meta"]["in_progress"])
            prediction = json.loads(artifacts.prediction_path.read_text())
            self.assertIn("model_patch", prediction)


class RunDatasetTest(unittest.TestCase):
    """The minimal controller: run many instances over a pool, aggregate outcomes."""

    def test_concurrent_run_with_ordering_and_callback(self) -> None:
        from simple_agent_lab.evals import run_dataset

        instances = [{"instance_id": f"i-{n}", "n": n} for n in range(6)]
        seen: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            backend = FakeBackend(on_run=_simulate("ok"))
            report = run_dataset(
                suite=_DemoSuite(),
                instances=instances,
                backend=backend,
                store=LocalDirStore(root),
                run_root=root,
                run_id="batch",
                concurrency=4,
                provider="fake",
                on_result=lambda r: seen.append(r.instance_id),
            )
        # All ran; results follow input order regardless of completion order.
        self.assertEqual(report.summary(), {"total": 6, "ok": 6, "failed": 0})
        self.assertEqual(
            [r.instance_id for r in report.results], [f"i-{n}" for n in range(6)]
        )
        self.assertEqual(len(seen), 6)
        # Each instance got its own run dir under the batch.
        self.assertEqual(len(backend.runs), 6)

    def test_error_is_captured_and_retried(self) -> None:
        from simple_agent_lab.evals import run_dataset

        attempts: dict[str, int] = {}

        def flaky(spec, store) -> None:
            n = attempts.get(spec.instance_id, 0) + 1
            attempts[spec.instance_id] = n
            if spec.instance_id == "bad":
                raise RuntimeError("boom")
            _simulate("ok")(spec, store)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            backend = FakeBackend(on_run=flaky)
            report = run_dataset(
                suite=_DemoSuite(),
                instances=[{"instance_id": "good"}, {"instance_id": "bad"}],
                backend=backend,
                store=LocalDirStore(root),
                run_root=root,
                run_id="batch",
                concurrency=2,
                max_attempts=3,
                provider="fake",
            )
        self.assertEqual(report.summary(), {"total": 2, "ok": 1, "failed": 1})
        bad = [r for r in report.results if r.instance_id == "bad"][0]
        self.assertFalse(bad.ok)
        self.assertIn("boom", bad.error)
        self.assertEqual(bad.attempts, 3)  # retried up to max_attempts
        self.assertEqual(attempts["bad"], 3)


class _FakeRemoteContainer:
    """Stand-in for a docker-py container: tar in (put), local FS, tar out (get)."""

    def __init__(self, root: Path) -> None:
        self.root = root  # the container's filesystem rooted here

    def put_archive(self, dest: str, tar_bytes: bytes) -> bool:
        import tarfile
        from io import BytesIO

        with tarfile.open(fileobj=BytesIO(tar_bytes), mode="r") as tar:
            tar.extractall(self.root / dest.lstrip("/"))
        return True

    def get_archive(self, path: str):
        import io
        import tarfile

        src = self.root / path.lstrip("/")
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            # docker names members relative to the requested path's parent, e.g.
            # get_archive(".../out") -> "out/<file>".
            tar.add(src, arcname=src.name)
        return [buffer.getvalue()], {}


class RemoteDockerHostPullTest(unittest.TestCase):
    """Host-pull copy logic without a daemon: push instance in, pull out/ back."""

    def test_push_inputs_then_pull_outputs(self) -> None:
        from simple_agent_lab.evals.backends._archive import (
            pack_file_to_root,
            unpack_members,
        )
        from simple_agent_lab.evals.backends.remote_docker import (
            RUN_MOUNT,
            pull_outputs,
            push_inputs,
        )

        # pure-archive round trip
        tar = pack_file_to_root("/agent/run/input/instance.json", b'{"x":1}')
        self.assertEqual(
            unpack_members(tar)["agent/run/input/instance.json"], b'{"x":1}'
        )

        with tempfile.TemporaryDirectory() as tmp:
            host_root = Path(tmp) / "host"
            container_fs = Path(tmp) / "worker"
            container_fs.mkdir()
            store = LocalDirStore(host_root).bind(host_root / "run" / "inst")
            store.put(INSTANCE_KEY, b'{"instance_id":"inst"}')

            container = _FakeRemoteContainer(container_fs)
            # 1. host pushes the instance into the worker container
            push_inputs(container, store)
            staged = container_fs / RUN_MOUNT.lstrip("/") / INSTANCE_KEY
            self.assertTrue(staged.exists())

            # 2. worker writes its outputs locally (simulated)
            out = container_fs / RUN_MOUNT.lstrip("/") / "out"
            out.mkdir(parents=True)
            (out / "result.json").write_text('{"model_patch":"diff"}')
            (out / "trajectory.jsonl").write_text('{"meta":{"in_progress":false}}\n')

            # 3. host pulls out/ back into its store (host-initiated, no reverse conn)
            pulled = pull_outputs(container, store)
            self.assertEqual(set(pulled), {RESULT_KEY, TRACE_KEY})
            self.assertEqual(json.loads(store.get(RESULT_KEY))["model_patch"], "diff")


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


if __name__ == "__main__":
    unittest.main()
