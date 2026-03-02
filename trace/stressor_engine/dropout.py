"""Observation dropout stressor.

Simulates sensor failures: camera blackout, noisy readings, or missing data.

Intensity controls dropout probability:
    0.0 -> observations always clean
    1.0 -> observations always dropped (replaced with zeros or noise)
"""

from typing import Any

import numpy as np

from trace.stressor_engine.base import BaseStressor, StressorConfig
from trace.task_spec.base import Observation


class DropoutStressor(BaseStressor):
    """Randomly drops or corrupts observation channels."""

    def __init__(self, config: StressorConfig) -> None:
        super().__init__(config)
        self._noise_scale: float = config.params.get("noise_scale", 0.1)
        self._mode: str = config.params.get("mode", "zero")  # "zero", "noise", "freeze"
        self._frozen_obs: dict[str, np.ndarray] | None = None

    def on_episode_start(self, task: Any) -> None:
        self._frozen_obs = None

    def perturb_observation(self, observation: Observation) -> Observation:
        if self.intensity == 0.0:
            return observation

        result = {}
        for key, value in observation.items():
            if self._rng.random() < self.intensity:
                result[key] = self._apply_dropout(key, value)
            else:
                result[key] = value
                if self._mode == "freeze":
                    if self._frozen_obs is None:
                        self._frozen_obs = {}
                    self._frozen_obs[key] = value.copy()

        return result

    def _apply_dropout(self, key: str, value: np.ndarray) -> np.ndarray:
        if self._mode == "zero":
            return np.zeros_like(value)
        elif self._mode == "noise":
            noise = self._rng.normal(0, self._noise_scale, size=value.shape)
            return (value + noise).astype(value.dtype)
        elif self._mode == "freeze":
            if self._frozen_obs and key in self._frozen_obs:
                return self._frozen_obs[key]
            return value
        else:
            return np.zeros_like(value)

    def perturb_action(self, action: np.ndarray) -> np.ndarray:
        return action  # Dropout only affects observations
