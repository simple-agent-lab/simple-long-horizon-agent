from __future__ import annotations

import unittest

from simple_agent_lab.state import State


class StateForkTest(unittest.TestCase):
    def _state_with_history(self) -> State:
        state = State(task="t")
        state.send("task", "user", "agent", "hello")
        state.send("step", "agent", "user", "working")
        return state

    def test_fork_is_independent(self) -> None:
        state = self._state_with_history()
        fork = state.fork()

        self.assertIsNot(fork, state)
        self.assertIsNot(fork.events, state.events)
        self.assertIsNot(fork.snapshot, state.snapshot)
        self.assertEqual(len(fork.messages), len(state.messages))

        before = len(state.events)
        fork.send("step", "agent", "user", "more")
        # Advancing the fork must not touch the parent.
        self.assertEqual(len(state.events), before)
        self.assertGreater(len(fork.events), before)

    def test_fork_preserves_monotonic_origin(self) -> None:
        # A fork's event timings must stay on the parent's clock, not restart.
        state = self._state_with_history()
        fork = state.fork()
        self.assertEqual(fork._monotonic_origin, state._monotonic_origin)

    def test_fork_deep_copies_data_scratchpad(self) -> None:
        state = self._state_with_history()
        state.data["patch"] = ["line1"]
        fork = state.fork()

        fork.data["patch"].append("line2")
        # The parent's mutable scratchpad value is unaffected.
        self.assertEqual(state.data["patch"], ["line1"])


if __name__ == "__main__":
    unittest.main()
