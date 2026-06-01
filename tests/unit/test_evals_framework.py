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
from unittest import mock

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
    RunOutcome,
    RunSpec,
    Suite,
    build_command,
    container_name,
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

    def test_workspace_factory_isolates_concurrent_runs(self) -> None:
        """A workspace factory gives each run its own dir, so concurrency is safe."""
        from simple_agent_lab.evals import run_dataset

        with tempfile.TemporaryDirectory() as tmp:
            ws_base = Path(tmp) / "ws"
            root = Path(tmp) / "runs"
            instances = [
                {
                    "instance_id": f"i-{n}",
                    "problem_statement": "p",
                    "language": "python",
                }
                for n in range(5)
            ]
            # Each run gets ws_base/<instance_id>; no two runs share a workspace.
            backend = LocalProcessBackend(
                workspace=lambda spec: ws_base / spec.instance_id
            )
            report = run_dataset(
                suite=_SwebenchLikeSuite(),
                instances=instances,
                backend=backend,
                store=LocalDirStore(root),
                run_root=root,
                run_id="batch",
                concurrency=4,
                provider="fake",
                max_turns=2,
            )
            self.assertEqual(report.summary(), {"total": 5, "ok": 5, "failed": 0})
            # Each run materialized its own workspace dir.
            for n in range(5):
                self.assertTrue((ws_base / f"i-{n}").is_dir())


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


class SubmitReconcileTest(unittest.TestCase):
    """Host-reentrant batch: submit, drop all memory, reconcile from disk only."""

    def test_submit_then_reconcile_from_fresh_process(self) -> None:
        from simple_agent_lab.evals import reconcile_dataset, submit_dataset

        instances = [{"instance_id": f"i-{n}", "problem": "p"} for n in range(4)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()

            # --- "first process": submit, then throw the backend/store away ---
            submit_dataset(
                suite=_DemoSuite(),
                instances=instances,
                backend=FakeBackend(on_run=_simulate("42")),
                store=LocalDirStore(root),
                run_root=root,
                run_id="batch",
                model_name="m",
                provider="fake",
            )
            # The manifest is on disk; nothing else carried over.
            self.assertTrue((root / "batch" / "batch.json").exists())

            # --- "fresh process": new backend + new store, recover from disk ---
            seen: list[str] = []
            report = reconcile_dataset(
                suite=_DemoSuite(),
                backend=FakeBackend(),  # no on_run, no memory of the submit
                store=LocalDirStore(root),
                run_root=root,
                run_id="batch",
                poll_interval_s=0,
                on_result=lambda r: seen.append(r.instance_id),
            )

            self.assertEqual(report.summary(), {"total": 4, "ok": 4, "failed": 0})
            self.assertEqual(len(seen), 4)
            # Predictions were shaped from each run's result.json during reconcile.
            for r in report.results:
                prediction = json.loads(r.artifacts.prediction_path.read_text())
                self.assertEqual(prediction["answer"], "42")
                self.assertEqual(prediction["model_name_or_path"], "m")

    def test_reconcile_waits_on_pending_runs(self) -> None:
        from simple_agent_lab.evals import reconcile_dataset, submit_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            submit_dataset(
                suite=_DemoSuite(),
                instances=[{"instance_id": "slow"}],
                backend=FakeBackend(on_run=_simulate("ok")),
                store=LocalDirStore(root),
                run_root=root,
                run_id="b",
                provider="fake",
            )
            # poll() returns None twice before reporting done — exercises the loop.
            report = reconcile_dataset(
                suite=_DemoSuite(),
                backend=FakeBackend(pending_polls=2),
                store=LocalDirStore(root),
                run_root=root,
                run_id="b",
                poll_interval_s=0,
            )
        self.assertEqual(report.summary(), {"total": 1, "ok": 1, "failed": 0})

    def test_submit_requires_detaching_backend(self) -> None:
        from simple_agent_lab.evals import LocalProcessBackend, submit_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with self.assertRaises(TypeError):
                submit_dataset(
                    suite=_DemoSuite(),
                    instances=[{"instance_id": "x"}],
                    backend=LocalProcessBackend(),  # run-only, cannot outlive host
                    store=LocalDirStore(root),
                    run_root=root,
                    run_id="b",
                )

    def test_mid_submit_crash_leaves_recoverable_manifest(self) -> None:
        """A crash partway through submit still records every started container."""
        from simple_agent_lab.evals import reconcile_dataset, submit_dataset
        from simple_agent_lab.evals.batch import BATCH_KEY, _batch_store

        class _CrashAtThird(FakeBackend):
            def __init__(self) -> None:
                super().__init__(on_run=_simulate("ok"))
                self.n = 0

            def submit(self, spec, *, store, binding):  # type: ignore[no-untyped-def]
                self.n += 1
                if self.n == 3:
                    raise RuntimeError("host died mid-submit")
                return super().submit(spec, store=store, binding=binding)

        instances = [{"instance_id": f"i-{n}"} for n in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            store = LocalDirStore(root)
            with self.assertRaises(RuntimeError):
                submit_dataset(
                    suite=_DemoSuite(),
                    instances=instances,
                    backend=_CrashAtThird(),
                    store=store,
                    run_root=root,
                    run_id="b",
                    provider="fake",
                )
            # The 2 containers started before the crash are in the manifest.
            manifest = json.loads(
                _batch_store(store, root, "b").get(BATCH_KEY).decode("utf-8")
            )
            self.assertEqual(len(manifest), 2)

            # A fresh process can reconcile the partial batch — no orphans.
            report = reconcile_dataset(
                suite=_DemoSuite(),
                backend=FakeBackend(),
                store=LocalDirStore(root),
                run_root=root,
                run_id="b",
                poll_interval_s=0,
            )
            self.assertEqual(report.summary(), {"total": 2, "ok": 2, "failed": 0})

    def test_reconcile_uses_result_when_poll_never_reports_done(self) -> None:
        """poll() that never returns done still completes if result.json exists."""
        from simple_agent_lab.evals import reconcile_dataset, submit_dataset

        class _NeverDone(FakeBackend):
            def poll(self, handle):  # type: ignore[no-untyped-def]
                return None  # daemon never reports completion (e.g. already gone)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            submit_dataset(
                suite=_DemoSuite(),
                instances=[{"instance_id": "x"}],
                backend=FakeBackend(on_run=_simulate("ok")),  # writes result.json
                store=LocalDirStore(root),
                run_root=root,
                run_id="b",
                provider="fake",
            )
            report = reconcile_dataset(
                suite=_DemoSuite(),
                backend=_NeverDone(),
                store=LocalDirStore(root),
                run_root=root,
                run_id="b",
                poll_interval_s=0,
            )
        self.assertEqual(report.summary(), {"total": 1, "ok": 1, "failed": 0})

    def test_finish_records_error_when_instance_record_missing(self) -> None:
        """A missing input/instance.json is a per-instance error, not a batch crash."""
        from simple_agent_lab.evals import reconcile_dataset, submit_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            submit_dataset(
                suite=_DemoSuite(),
                # _simulate writes result+trace but the instance.json is written by
                # submit_dataset itself; delete it to simulate a partial submit.
                instances=[{"instance_id": "x"}],
                backend=FakeBackend(on_run=_simulate("ok")),
                store=LocalDirStore(root),
                run_root=root,
                run_id="b",
                provider="fake",
            )
            (root / "b" / "x" / "input" / "instance.json").unlink()
            report = reconcile_dataset(
                suite=_DemoSuite(),
                backend=FakeBackend(),  # poll() -> done immediately
                store=LocalDirStore(root),
                run_root=root,
                run_id="b",
                poll_interval_s=0,
            )
            # One result, marked failed with an error — not an exception.
            self.assertEqual(report.summary(), {"total": 1, "ok": 0, "failed": 1})
            self.assertIn("cannot load instance record", report.results[0].error)

    def test_reconcile_floors_the_idle_wait(self) -> None:
        """poll_interval_s=0 must not busy-spin: the loop sleeps a real floor."""
        from simple_agent_lab.evals import reconcile_dataset, submit_dataset

        slept: list[float] = []

        # Returns None a few times so the loop has to wait, then completes.
        class _SlowThenDone(FakeBackend):
            def __init__(self) -> None:
                super().__init__()
                self.n = 0

            def poll(self, handle):  # type: ignore[no-untyped-def]
                self.n += 1
                return None if self.n < 3 else RunOutcome(status_code=0)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            # No on_run → no result.json, so poll()==None falls through to the
            # idle wait (instead of the result-on-disk short-circuit).
            submit_dataset(
                suite=_DemoSuite(),
                instances=[{"instance_id": "x"}],
                backend=FakeBackend(),
                store=LocalDirStore(root),
                run_root=root,
                run_id="b",
                provider="fake",
            )
            reconcile_dataset(
                suite=_DemoSuite(),
                backend=_SlowThenDone(),
                store=LocalDirStore(root),
                run_root=root,
                run_id="b",
                poll_interval_s=0,  # caller asks for 0 …
                sleep_fn=slept.append,
            )
        # … but the loop floored every wait above 0 (no busy-spin).
        self.assertTrue(slept)
        self.assertTrue(all(s >= 0.5 for s in slept))


class ReviewFixesTest(unittest.TestCase):
    """Docker-path guards (no real Docker): exit-code parsing, atomic put, names."""

    def test_exit_status_guards_null_and_nonint(self) -> None:
        from simple_agent_lab.evals.backends.docker_local import exit_status

        self.assertEqual(exit_status({"ExitCode": 0}), 0)
        self.assertEqual(exit_status({"ExitCode": 137}), 137)
        self.assertEqual(exit_status({"ExitCode": None}), 1)  # null → failure
        self.assertEqual(exit_status({}), 1)  # missing → failure

    def test_host_http_put_is_atomic(self) -> None:
        """put writes via a temp file + replace (no torn read), like LocalDirStore."""
        import os as _os

        from simple_agent_lab.evals import HostHttpStore

        seen_tmp: list[str] = []
        real_replace = _os.replace

        def spy_replace(src, dst):  # type: ignore[no-untyped-def]
            seen_tmp.append(str(src))
            return real_replace(src, dst)

        with tempfile.TemporaryDirectory() as tmp:
            store = HostHttpStore(Path(tmp), container_host="127.0.0.1")
            with mock.patch(
                "simple_agent_lab.evals.stores.host_http.os.replace", spy_replace
            ):
                with store:
                    bound = store.bind(Path(tmp) / "r" / "i")
                    bound.put("out/result.json", b'{"ok": true}')
                    self.assertEqual(bound.get("out/result.json"), b'{"ok": true}')
        self.assertTrue(seen_tmp and seen_tmp[0].endswith(".tmp"))

    def test_container_name_clamped_and_distinct(self) -> None:
        short = container_name("swebench", "sympy__sympy-23824", "run-1")
        self.assertLessEqual(len(short), 200)
        self.assertEqual(short, "swebench.sympy__sympy-23824.run-1")  # unchanged

        long_a = container_name("swebench", "x" * 300, "run-1")
        long_b = container_name("swebench", "x" * 300 + "y", "run-1")
        self.assertLessEqual(len(long_a), 200)
        self.assertNotEqual(long_a, long_b)  # distinct overflowing names stay distinct


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
        # Pro images carry no test-spec caps (Verified ones come from the spec,
        # which needs the swebench harness installed — not asserted here).
        self.assertEqual(plan.cap_add, ())


class SafePartTest(unittest.TestCase):
    def test_distinct_ids_never_collide_after_sanitization(self) -> None:
        from simple_agent_lab.evals.runner import _safe_part

        # Plain ids (alnum / _.-) are unchanged — SWE-bench ids stay readable.
        self.assertEqual(_safe_part("sympy__sympy-23824"), "sympy__sympy-23824")
        # Ids differing only in replaced chars must not map to the same dir/name.
        self.assertNotEqual(_safe_part("org/repo#1"), _safe_part("org_repo_1"))
        self.assertNotEqual(_safe_part("a:b"), _safe_part("a_b"))
        # Deterministic: same raw id → same safe part.
        self.assertEqual(_safe_part("a:b"), _safe_part("a:b"))


class CreateKwargsTest(unittest.TestCase):
    def test_shared_create_kwargs_carries_plan_fields(self) -> None:
        from simple_agent_lab.evals.backends.docker_local import _create_kwargs

        spec = RunSpec(
            suite_name="s",
            container_module="m",
            instance_id="i",
            plan=ContainerPlan(
                image="img", workdir="/w", cap_add=("SYS_PTRACE",), entrypoint=""
            ),
            max_turns=3,
            provider="fake",
            api_kind="openai-chat",
            run_name="run-1",
        )
        from simple_agent_lab.evals import ContainerBinding

        binding = ContainerBinding(
            mounts={"/host": {"bind": "/agent/run", "mode": "rw"}},
            env={"SAL_STORE": "localdir"},
        )
        kwargs = _create_kwargs(spec, binding, user="root", environment={"X": "1"})
        self.assertEqual(kwargs["image"], "img")
        self.assertEqual(kwargs["name"], "run-1")
        self.assertEqual(kwargs["cap_add"], ["SYS_PTRACE"])  # plan cap_add plumbed
        self.assertEqual(
            kwargs["volumes"], {"/host": {"bind": "/agent/run", "mode": "rw"}}
        )
        self.assertEqual(kwargs["environment"], {"X": "1"})
        self.assertEqual(
            kwargs["entrypoint"], ""
        )  # "" included (clears image ENTRYPOINT)
        self.assertNotIn("platform", kwargs)  # omitted when unset


if __name__ == "__main__":
    unittest.main()
