"""Physics shift stressor.

Modifies physical properties of the MuJoCo simulation: friction, mass, damping.
Simulates sim-to-real gaps and environment variability.

Intensity controls the magnitude of the shift:
    0.0 -> nominal physics
    1.0 -> maximum perturbation (e.g., 2x mass, 0.5x friction)
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from trace.stressor_engine.base import BaseStressor, StressorConfig


class PhysicsShiftStressor(BaseStressor):
    """Modifies simulation physics parameters."""

    def __init__(self, config: StressorConfig) -> None:
        super().__init__(config)
        self._mass_range: tuple[float, float] = tuple(
            config.params.get("mass_range", [0.5, 2.0])
        )
        self._friction_range: tuple[float, float] = tuple(
            config.params.get("friction_range", [0.3, 1.5])
        )
        self._damping_range: tuple[float, float] = tuple(
            config.params.get("damping_range", [0.5, 2.0])
        )
        # Store originals so we can restore
        self._original_mass: NDArray[np.floating] | None = None
        self._original_friction: NDArray[np.floating] | None = None
        self._original_damping: NDArray[np.floating] | None = None

    def on_episode_start(self, task: Any) -> None:
        if self.intensity == 0.0:
            return

        try:
            model = task.get_mujoco_model()
        except NotImplementedError:
            return

        # Save originals
        self._original_mass = model.body_mass.copy()
        self._original_friction = model.geom_friction.copy()
        self._original_damping = model.dof_damping.copy()

        # Apply perturbations scaled by intensity
        mass_scale = self._interpolate(1.0, self._mass_range, self.intensity)
        friction_scale = self._interpolate(1.0, self._friction_range, self.intensity)
        damping_scale = self._interpolate(1.0, self._damping_range, self.intensity)

        model.body_mass[:] = self._original_mass * mass_scale
        model.geom_friction[:] = self._original_friction * friction_scale
        model.dof_damping[:] = self._original_damping * damping_scale

    def _interpolate(
        self, nominal: float, stress_range: tuple[float, float], intensity: float
    ) -> float:
        """Pick a scale factor: at intensity 0 return nominal, at 1 sample from range."""
        low, high = stress_range
        target = self._rng.uniform(low, high)
        return nominal + intensity * (target - nominal)

    def on_episode_end(self) -> None:
        pass  # Task reset will reload physics anyway

    def perturb_observation(
        self, observation: dict[str, NDArray[np.floating]]
    ) -> dict[str, NDArray[np.floating]]:
        return observation  # Physics shift is applied at env level

    def perturb_action(self, action: NDArray[np.floating]) -> NDArray[np.floating]:
        return action  # Physics shift is applied at env level
