"""Tests for stressor implementations."""

import numpy as np

from trace.stressor_engine.base import StressorConfig
from trace.stressor_engine.latency import LatencyStressor
from trace.stressor_engine.dropout import DropoutStressor
from trace.stressor_engine.long_horizon import LongHorizonDriftStressor


class TestLatencyStressor:
    def test_zero_intensity_passthrough(self, dummy_observation):
        config = StressorConfig(name="latency", intensity=0.0, seed=0)
        stressor = LatencyStressor(config)
        stressor.on_episode_start(None)

        action = np.ones(7, dtype=np.float32)
        result = stressor.perturb_action(action)
        np.testing.assert_array_equal(result, action)

    def test_high_intensity_delays(self, dummy_observation):
        config = StressorConfig(
            name="latency", intensity=1.0, seed=0, params={"max_delay_steps": 5}
        )
        stressor = LatencyStressor(config)
        stressor.on_episode_start(None)

        # First actions should be zeros (buffering)
        action = np.ones(7, dtype=np.float32)
        for _ in range(5):
            result = stressor.perturb_action(action)
            np.testing.assert_array_equal(result, np.zeros(7))

        # After buffer fills, should get delayed actions
        result = stressor.perturb_action(action)
        np.testing.assert_array_equal(result, action)

    def test_observation_passthrough(self, dummy_observation):
        config = StressorConfig(name="latency", intensity=0.5, seed=0)
        stressor = LatencyStressor(config)
        result = stressor.perturb_observation(dummy_observation)
        for key in dummy_observation:
            np.testing.assert_array_equal(result[key], dummy_observation[key])


class TestDropoutStressor:
    def test_zero_intensity_passthrough(self, dummy_observation):
        config = StressorConfig(name="dropout", intensity=0.0, seed=0)
        stressor = DropoutStressor(config)
        stressor.on_episode_start(None)

        result = stressor.perturb_observation(dummy_observation)
        for key in dummy_observation:
            np.testing.assert_array_equal(result[key], dummy_observation[key])

    def test_full_dropout_zeros(self, dummy_observation):
        config = StressorConfig(
            name="dropout", intensity=1.0, seed=0, params={"mode": "zero"}
        )
        stressor = DropoutStressor(config)
        stressor.on_episode_start(None)

        result = stressor.perturb_observation(dummy_observation)
        for key in result:
            assert np.all(result[key] == 0.0)

    def test_action_passthrough(self):
        config = StressorConfig(name="dropout", intensity=1.0, seed=0)
        stressor = DropoutStressor(config)
        action = np.ones(7, dtype=np.float32)
        result = stressor.perturb_action(action)
        np.testing.assert_array_equal(result, action)


class TestLongHorizonDrift:
    def test_drift_grows_over_time(self, dummy_observation):
        config = StressorConfig(
            name="drift", intensity=0.5, seed=42,
            params={"obs_noise_growth": 0.1, "action_noise_growth": 0.05},
        )
        stressor = LongHorizonDriftStressor(config)
        stressor.on_episode_start(None)

        action = np.ones(7, dtype=np.float32)
        deviations = []

        for _ in range(50):
            perturbed = stressor.perturb_action(action)
            deviation = np.mean(np.abs(perturbed - action))
            deviations.append(deviation)

        # Later deviations should generally be larger
        early_mean = np.mean(deviations[:10])
        late_mean = np.mean(deviations[40:])
        assert late_mean > early_mean

    def test_zero_intensity_passthrough(self, dummy_observation):
        config = StressorConfig(name="drift", intensity=0.0, seed=0)
        stressor = LongHorizonDriftStressor(config)
        stressor.on_episode_start(None)

        action = np.ones(7, dtype=np.float32)
        result = stressor.perturb_action(action)
        np.testing.assert_array_equal(result, action)
