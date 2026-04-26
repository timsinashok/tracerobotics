"""Embodiment mismatch stressor.

Simulates differences between the training embodiment and the deployment robot.
Modifies arm link lengths, joint limits, and actuator gains.

Intensity controls how far the embodiment deviates from nominal:
    0.0 -> exact training embodiment
    1.0 -> maximum perturbation
"""

from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from trace.stressor_engine.base import BaseStressor, StressorConfig
from trace.task_spec.base import Observation


class EmbodimentStressor(BaseStressor):
    """Perturbs the robot embodiment parameters."""

    def __init__(self, config: StressorConfig) -> None:
        super().__init__(config)
        self._link_length_range: tuple[float, float] = tuple(
            config.params.get("link_length_range", [0.9, 1.1])
        )
        self._joint_limit_range: tuple[float, float] = tuple(
            config.params.get("joint_limit_range", [0.85, 1.0])
        )
        self._gain_range: tuple[float, float] = tuple(
            config.params.get("gain_range", [0.7, 1.3])
        )
        self._original_geom_size: NDArray[np.floating] | None = None
        self._original_jnt_range: NDArray[np.floating] | None = None
        self._original_actuator_gain: NDArray[np.floating] | None = None

    def on_episode_start(self, task: Any) -> None:
        if self.intensity == 0.0:
            return

        try:
            model = task.get_mujoco_model()
            data = task.get_mujoco_data()
        except NotImplementedError:
            return

        # Save originals
        self._original_geom_size = model.geom_size.copy()
        self._original_jnt_range = model.jnt_range.copy()
        self._original_actuator_gain = model.actuator_gainprm.copy()

        # Scale link geometry
        low, high = self._link_length_range
        link_scale = 1.0 + self.intensity * (self._rng.uniform(low, high) - 1.0)
        model.geom_size[:] = self._original_geom_size * link_scale

        # Tighten joint limits
        low, high = self._joint_limit_range
        jnt_scale = 1.0 + self.intensity * (self._rng.uniform(low, high) - 1.0)
        model.jnt_range[:] = self._original_jnt_range * jnt_scale

        # Perturb actuator gains
        low, high = self._gain_range
        gain_scale = 1.0 + self.intensity * (self._rng.uniform(low, high) - 1.0)
        model.actuator_gainprm[:] = self._original_actuator_gain * gain_scale

        # Recompute MuJoCo derived constants (bounding volumes, etc.)
        mujoco.mj_setConst(model, data)

    def perturb_observation(self, observation: Observation) -> Observation:
        return observation  # Embodiment is applied at env level

    def perturb_action(self, action: np.ndarray) -> np.ndarray:
        return action  # Embodiment is applied at env level
