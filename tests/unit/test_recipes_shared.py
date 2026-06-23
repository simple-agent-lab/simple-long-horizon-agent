import os
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import recipes.runtime as runtime
from simple_agent_lab.evals.protocols import LaunchSpec


class BranchConcurrencyTest(unittest.TestCase):
    def test_never_exceeds_global(self):
        self.assertEqual(runtime.branch_concurrency(global_workers=19, branches=3), 6)

    def test_floor_is_one(self):
        self.assertEqual(runtime.branch_concurrency(global_workers=2, branches=5), 1)


class ResolveParallelTest(unittest.TestCase):
    def test_honors_explicit(self):
        res = runtime.resolve_parallel_workers("4", 8)
        self.assertEqual(res.workers, 4)
        self.assertIn("explicit", res.detail)

    def test_default_is_one(self):
        res = runtime.resolve_parallel_workers(None, 8)
        self.assertEqual(res.workers, 1)
        self.assertIn("explicit", res.detail)

    def test_does_not_probe_docker(self):
        def broken():
            raise AssertionError("docker should not be probed")

        res = runtime.resolve_parallel_workers("2", 8, client_factory=broken)
        self.assertEqual(res.workers, 2)

    def test_rejects_invalid(self):
        with self.assertRaises(SystemExit):
            runtime.resolve_parallel_workers("zero", 8)
        with self.assertRaises(SystemExit):
            runtime.resolve_parallel_workers("auto", 8)
        with self.assertRaises(SystemExit):
            runtime.resolve_parallel_workers("0", 8)


class DockerHelpersTest(unittest.TestCase):
    def test_check_docker_available_wraps_errors(self):
        def broken():
            raise RuntimeError("socket missing")

        with self.assertRaisesRegex(SystemExit, "Docker is required"):
            runtime.check_docker_available(client_factory=broken)

    def test_cleanup_reset_containers_removes_matching(self):
        class FakeContainer:
            def __init__(self):
                self.name = "swebench.astropy__astropy-14369.3b89f9b34cec-9119f5eb01dc"
                self.removed = False

            def remove(self, force=False):
                self.removed = force

        class FakeContainers:
            def __init__(self):
                self.container = FakeContainer()
                self.filters = []

            def list(self, *, all=False, filters=None):
                self.all = all
                self.filters.append(filters)
                return [self.container]

        class FakeClient:
            def __init__(self):
                self.containers = FakeContainers()

        client = FakeClient()
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "dgm-trend-8x4-g4"
            (run_root / "swebench_runs" / "3b89f9b34cec-9119f5eb01dc").mkdir(
                parents=True
            )
            removed = runtime.cleanup_reset_containers(
                run_root, client_factory=lambda: client
            )
        self.assertEqual(removed, 1)
        self.assertTrue(client.containers.container.removed)

    def test_preflight_suite_images_checks_unique_images_once(self):
        class FakeImages:
            def __init__(self):
                self.checked = []

            def get(self, image):
                self.checked.append(image)

        class FakeClient:
            def __init__(self):
                self.images = FakeImages()

        class FakeSuite:
            def launch_spec(self, instance):
                return LaunchSpec(image=instance["image"], workdir="/work")

        client = FakeClient()

        with contextlib.redirect_stdout(io.StringIO()):
            checked = runtime.preflight_suite_images(
                FakeSuite(),
                ({"image": "img:a"}, {"image": "img:a"}, {"image": "img:b"}),
                client_factory=lambda: client,
                pull="never",
                label="train",
            )

        self.assertEqual(checked, 2)
        self.assertEqual(client.images.checked, ["img:a", "img:b"])

    def test_preflight_suite_images_reports_missing_images_before_running(self):
        class FakeImages:
            def get(self, image):
                raise RuntimeError(f"missing {image}")

        class FakeClient:
            def __init__(self):
                self.images = FakeImages()

        class FakeSuite:
            def launch_spec(self, instance):
                return LaunchSpec(image=instance["image"], workdir="/work")

        with self.assertRaisesRegex(SystemExit, "SWE-bench image preflight failed"):
            runtime.preflight_suite_images(
                FakeSuite(),
                ({"image": "img:missing"},),
                client_factory=lambda: FakeClient(),
                pull="never",
                label="train",
            )


class LoadDotenvTest(unittest.TestCase):
    def test_sets_missing_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "OPENAI_MODEL=dotenv-model\nOPENAI_AUTH_TOKEN=dotenv-token\n",
                encoding="utf-8",
            )
            old = {k: os.environ.get(k) for k in ("OPENAI_MODEL", "OPENAI_AUTH_TOKEN")}
            for k in old:
                os.environ.pop(k, None)
            try:
                runtime.load_dotenv(env_path)
                self.assertEqual(os.environ["OPENAI_MODEL"], "dotenv-model")
                self.assertEqual(os.environ["OPENAI_AUTH_TOKEN"], "dotenv-token")
            finally:
                for k, v in old.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
