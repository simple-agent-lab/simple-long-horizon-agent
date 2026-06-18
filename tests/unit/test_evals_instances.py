from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from simple_agent_lab.evals.instances import InstanceSet, load_jsonl_instances
from simple_agent_lab.evolution.types import Slice


class InstanceSetTest(unittest.TestCase):
    def test_sha_is_stable_and_order_insensitive(self) -> None:
        a = InstanceSet(
            "train",
            (
                {"instance_id": "b", "value": 2},
                {"instance_id": "a", "value": 1},
            ),
        )
        b = InstanceSet(
            "train",
            (
                {"instance_id": "a", "value": 1},
                {"instance_id": "b", "value": 2},
            ),
        )

        self.assertEqual(a.sha, b.sha)
        self.assertEqual(a.n, 2)

    def test_sha_uses_index_when_instance_id_is_missing(self) -> None:
        items = InstanceSet("custom", ({"x": 1}, {"x": 2}))

        self.assertEqual(items.n, 2)
        self.assertEqual(len(items.sha), 12)

    def test_load_jsonl_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "instances.jsonl"
            path.write_text(
                json.dumps({"instance_id": "i1"})
                + "\n\n"
                + json.dumps({"instance_id": "i2"})
                + "\n",
                encoding="utf-8",
            )

            loaded = load_jsonl_instances(path)

        self.assertEqual(tuple(row["instance_id"] for row in loaded), ("i1", "i2"))

    def test_load_jsonl_instances_rejects_non_object_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "instances.jsonl"
            path.write_text(
                json.dumps(["not", "an", "object"]) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "non-object JSONL row"):
                load_jsonl_instances(path)

    def test_slice_remains_constructible_as_compatibility_alias(self) -> None:
        slice_ = Slice("train", ({"instance_id": "i1"},))

        self.assertEqual(slice_.id, "train")
        self.assertEqual(slice_.n, 1)
        self.assertEqual(len(slice_.sha), 12)

    def test_slice_rejects_assigning_new_attributes(self) -> None:
        slice_ = Slice("train")

        with self.assertRaises(FrozenInstanceError):
            setattr(slice_, "extra", "changed")

    def test_top_level_exports_work(self) -> None:
        from simple_agent_lab.evals import InstanceSet as EvalsInstanceSet
        from simple_agent_lab.evals import load_jsonl_instances as evals_loader
        from simple_agent_lab.evolution import InstanceSet as EvolutionInstanceSet
        from simple_agent_lab.evolution import Slice as EvolutionSlice

        self.assertIs(EvalsInstanceSet, InstanceSet)
        self.assertIs(evals_loader, load_jsonl_instances)
        self.assertIs(EvolutionInstanceSet, InstanceSet)
        self.assertIs(EvolutionSlice, Slice)


if __name__ == "__main__":
    unittest.main()
