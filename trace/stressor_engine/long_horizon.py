"""Long-horizon drift stressor.

Simulates gradual degradation over time: accumulating observation noise,
slowly shifting physics, or increasing action perturbations as the episode
progresses.

Intensity controls the rate and magnitude of drift:
    0.0 -> no drift
    1.0 -> fast, large drift
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from trace.stressor_engine.base import BaseStressor, StressorConfig


class LongHorizonDriftStressor(BaseStressor):
    """Applies time-varying perturbations that grow over the episode."""

    def __init__(self, config: StressorConfig) -> None:
        super().__init__(config)
        self._obs_noise_growth: float = config.params.get("obs_noise_growth", 0.01)
        self._action_noise_growth: float = config.params.get("action_noise_growth", 0.005)
        self._current_step: int = 0

    def on_episode_start(self, task: Any) -> None:
        self._current_step = 0

    @property
    def _drift_factor(self) -> float:
        """Drift magnitude increases linearly with step count and intensity."""
        return self.intensity * self._current_step

    def perturb_observation(
        self, observation: dict[str, NDArray[np.floating]]
    ) -> dict[str, NDArray[np.floating]]:
        if self.intensity == 0.0:
            return observation

        noise_scale = self._obs_noise_growth * self._drift_factor
        result = {}
        for key, value in observation.items():
            noise = self._rng.normal(0, max(noise_scale, 1e-8), size=value.shape)
            result[key] = (value + noise).astype(value.dtype)
        return result

    def perturb_action(self, action: NDArray[np.floating]) -> NDArray[np.floating]:
        self._current_step += 1

        if self.intensity == 0.0:
            return action

        noise_scale = self._action_noise_growth * self._drift_factor
        noise = self._rng.normal(0, max(noise_scale, 1e-8), size=action.shape)
        return (action + noise).astype(action.dtype)
