"""Shared test fixtures."""

import pytest
import numpy as np

from trace.policy_adapter.random_policy import RandomPolicy
from trace.stressor_engine.base import StressorConfig


@pytest.fixture
def random_policy() -> RandomPolicy:
    return RandomPolicy(action_dim=7, seed=42)


@pytest.fixture
def dummy_observation() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "image": rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8),
        "proprioception": rng.random(14, dtype=np.float32),
    }


@pytest.fixture
def default_stressor_config() -> StressorConfig:
    return StressorConfig(name="test_stressor", intensity=0.5, seed=42)
