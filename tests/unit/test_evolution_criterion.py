from __future__ import annotations

import unittest

from simple_agent_lab.evolution.components.criterion import (
    guarded,
    improve,
    not_worse,
    promote_not_worse,
)


class CriterionTest(unittest.TestCase):
    def test_improve_accepts_strict_gain(self) -> None:
        base = {"i1": {"reward": 0.0}, "i2": {"reward": 0.0}}
        cand = {"i1": {"reward": 1.0}, "i2": {"reward": 0.0}}
        v = improve("reward")(base, cand)
        self.assertTrue(v.accepted)
        self.assertAlmostEqual(v.deltas["reward"], 0.5)

    def test_improve_rejects_no_gain(self) -> None:
        base = {"i1": {"reward": 0.5}}
        cand = {"i1": {"reward": 0.5}}
        self.assertFalse(improve("reward")(base, cand).accepted)

    def test_improve_missing_dimension_raises(self) -> None:
        with self.assertRaises(KeyError):
            improve("nope")({"i1": {"reward": 0.1}}, {"i1": {"reward": 0.2}})

    def test_not_worse_guard(self) -> None:
        base = {"i1": {"reward": 1.0}}
        cand = {"i1": {"reward": 0.99}}
        self.assertTrue(not_worse("reward", tol=0.05)(base, cand).accepted)
        self.assertFalse(not_worse("reward", tol=0.0)(base, cand).accepted)

    def test_promote_not_worse_accepts_strict_gain(self) -> None:
        base = {"i1": {"reward": 0.0}, "i2": {"reward": 0.0}}
        cand = {"i1": {"reward": 1.0}, "i2": {"reward": 0.0}}
        verdict = promote_not_worse("reward")(base, cand)
        self.assertTrue(verdict.accepted)
        self.assertAlmostEqual(verdict.deltas["reward"], 0.5)
        self.assertIn("improved", verdict.reason)

    def test_promote_not_worse_accepts_tie(self) -> None:
        base = {"i1": {"reward": 0.5}}
        cand = {"i1": {"reward": 0.5}}
        verdict = promote_not_worse("reward")(base, cand)
        self.assertTrue(verdict.accepted)
        self.assertAlmostEqual(verdict.deltas["reward"], 0.0)
        self.assertIn("not worse", verdict.reason)

    def test_promote_not_worse_rejects_regression(self) -> None:
        base = {"i1": {"reward": 0.5}}
        cand = {"i1": {"reward": 0.4}}
        verdict = promote_not_worse("reward")(base, cand)
        self.assertFalse(verdict.accepted)
        self.assertAlmostEqual(verdict.deltas["reward"], -0.1)
        self.assertIn("regressed", verdict.reason)

    def test_guarded_requires_objective_and_all_guards(self) -> None:
        crit = guarded(improve("reward"), [not_worse("safety")])
        base = {"i1": {"reward": 0.0, "safety": 1.0}}
        good = {"i1": {"reward": 1.0, "safety": 1.0}}
        bad = {"i1": {"reward": 1.0, "safety": 0.0}}
        self.assertTrue(crit(base, good).accepted)
        self.assertFalse(crit(base, bad).accepted)


if __name__ == "__main__":
    unittest.main()
