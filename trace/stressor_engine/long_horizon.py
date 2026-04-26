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

from trace.stressor_engine.base import BaseStressor, StressorConfig
from trace.task_spec.base import Observation


class LongHorizonDriftStressor(BaseStressor):
    """Applies time-varying perturbations that grow over the episode."""

    def __init__(self, config: StressorConfig) -> None:
        super().__init__(config)
        self._obs_noise_growth: float = config.params.get("obs_noise_growth", 0.01)
        self._action_noise_growth: float = config.params.get("action_noise_growth", 0.005)
        self._gripper_dims: int = config.params.get("gripper_dims", 1)
        self._current_step: int = 0

    def on_episode_start(self, task: Any) -> None:
        self._current_step = 0

    @property
    def _drift_factor(self) -> float:
        """Drift magnitude increases linearly with step count and intensity."""
        return self.intensity * self._current_step

    def perturb_observation(self, observation: Observation) -> Observation:
        if self.intensity == 0.0:
            return observation

        noise_scale = self._obs_noise_growth * self._drift_factor
        result = {}
        for key, value in observation.items():
            if value.dtype == np.uint8:
                result[key] = value  # Skip drift for image obs (Phase 5 visual stressors)
                continue
            noise = self._rng.normal(0, max(noise_scale, 1e-8), size=value.shape)
            result[key] = (value + noise).astype(value.dtype)
        return result

    def perturb_action(self, action: np.ndarray) -> np.ndarray:
        self._current_step += 1

        if self.intensity == 0.0:
            return action

        noise_scale = self._action_noise_growth * self._drift_factor
        noise = self._rng.normal(0, max(noise_scale, 1e-8), size=action.shape)
        # Don't perturb gripper dimensions — noise can flip open/close commands
        if self._gripper_dims > 0:
            noise[-self._gripper_dims:] = 0.0
        return (action + noise).astype(action.dtype)
