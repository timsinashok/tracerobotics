"""Tests for policy adapters."""

import numpy as np

from trace.policy_adapter.random_policy import RandomPolicy


class TestRandomPolicy:
    def test_act_returns_correct_shape(self, random_policy, dummy_observation):
        action = random_policy.act(dummy_observation)
        assert action.shape == (7,)
        assert action.dtype == np.float32

    def test_act_bounded(self, random_policy, dummy_observation):
        for _ in range(100):
            action = random_policy.act(dummy_observation)
            assert np.all(action >= -1.0)
            assert np.all(action <= 1.0)

    def test_reset_reproducibility(self, dummy_observation):
        policy = RandomPolicy(action_dim=7, seed=123)
        a1 = policy.act(dummy_observation)
        policy.reset()
        a2 = policy.act(dummy_observation)
        np.testing.assert_array_equal(a1, a2)

    def test_metadata(self, random_policy):
        meta = random_policy.metadata()
        assert meta.name == "RandomPolicy"
        assert meta.action_space["dim"] == 7
