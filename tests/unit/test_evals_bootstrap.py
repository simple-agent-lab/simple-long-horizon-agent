from __future__ import annotations

import unittest

from simple_agent_lab.evals.bootstrap import bootstrap_script


class EvalBootstrapTest(unittest.TestCase):
    def test_uv_path_requires_managed_python_for_cp311_wheelhouse(self) -> None:
        script = bootstrap_script(
            runner_argv=("-m", "demo.container"),
            wheelhouse_mount="/agent/wheelhouse",
        )

        self.assertIn("--managed-python", script)
        self.assertNotIn("--python python3", script)
        self.assertIn("refusing to fall back", script)

    def test_bootstrap_prepends_candidate_pythonpath_before_runner(self) -> None:
        script = bootstrap_script(
            runner_argv=("-m", "simple_agent_lab.evals.in_container"),
            install=False,
            extra_pythonpath=("/agent/run/input/source_tree/src",),
        )

        assert (
            'export PYTHONPATH="/agent/run/input/source_tree/src${PYTHONPATH:+:$PYTHONPATH}"'
            in script
        )
        assert script.index("export PYTHONPATH=") < script.index('"$AGENT_PYTHON" -m')


if __name__ == "__main__":
    unittest.main()
