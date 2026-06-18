from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from simple_agent_lab.evolution.run_config import load_self_evolving_config


CONFIG = """
run:
  id: demo
  output_root: evals/out/demo
  execute: false
  reset: false
  dotenv: .env
suite:
  name: swebench
  args:
    dataset_name: demo-dataset
surface:
  name: python_agent_package
  editable_components: [everything]
  artifact_key: input/agent_package.json
  default: simple_agent_package
instances:
  train:
    id: train
    path: train.jsonl
execution:
  backend:
    name: fake
  store:
    name: local_dir
  parallel: 1
  max_turns: 3
model:
  api_kind: openai-chat
  model_env: OPENAI_MODEL
  api_key_env: OPENAI_AUTH_TOKEN
strategy:
  name: model_program
  args:
    system_prompt: demo
evolution:
  algorithm: simple
  rounds: 2
  criterion:
    name: promote_not_worse
    args:
      dim: reward
evaluation:
  baseline_heldout: false
  final_heldout: false
  heldout_every_rounds: 0
  repeats: 1
  official_scoring: false
"""


class RunConfigTest(unittest.TestCase):
    def test_load_self_evolving_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(CONFIG, encoding="utf-8")

            config = load_self_evolving_config(path)

        self.assertEqual(config.run.id, "demo")
        self.assertEqual(config.suite.name, "swebench")
        self.assertEqual(config.surface.editable_components, ("everything",))
        self.assertEqual(config.evolution.rounds, 2)

    def test_missing_required_section_names_the_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("run: {id: demo}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "suite"):
                load_self_evolving_config(path)


if __name__ == "__main__":
    unittest.main()
