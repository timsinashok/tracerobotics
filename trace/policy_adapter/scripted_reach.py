"""Scripted reach policy — simple proportional controller for testing.

Maps Cartesian error (end-effector → target) into joint-space deltas
using a transpose-Jacobian-like heuristic. Won't solve IK perfectly,
but produces directed motion — enough for testing the pipeline and
measuring degradation under stress.
"""

import numpy as np
from numpy.typing import NDArray

from trace.policy_adapter.base import BasePolicy, PolicyMetadata


class ScriptedReachPolicy(BasePolicy):
    """Proportional controller that moves the end-effector toward the target."""

    def __init__(self, gain: float = 5.0, smoothing: float = 0.7) -> None:
        self._gain = gain
        self._smoothing = smoothing
        self._prev_action: NDArray[np.floating] | None = None

    def load(self, checkpoint_path: str) -> None:
        pass  # No weights to load

    def reset(self) -> None:
        self._prev_action = None

    def act(self, observation: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        ee_pos = observation["ee_pos"]
        target_pos = observation["target_pos"]

        # Cartesian error
        error = target_pos - ee_pos

        # Distribute Cartesian error across 7 joints using a simple heuristic:
        # joints 1,3,5,7 contribute to one axis, joints 2,4,6 to another.
        # This is a rough proxy for the Jacobian transpose approach.
        action = np.zeros(7, dtype=np.float32)
        action[0] = error[1] * self._gain       # joint1 (z-axis) → y-error
        action[1] = -error[2] * self._gain       # joint2 (y-axis) → z-error
        action[2] = -error[0] * self._gain       # joint3 (z-axis) → x-error
        action[3] = error[2] * self._gain * 0.5  # joint4 (y-axis) → z-error (elbow)
        action[4] = error[0] * self._gain * 0.3  # joint5 (z-axis) → x-error (wrist)
        action[5] = error[2] * self._gain * 0.3  # joint6 (y-axis) → z-error (wrist)
        action[6] = 0.0                           # joint7 (z-axis) → minimal contrib

        # Clip to [-1, 1] range
        action = np.clip(action, -1.0, 1.0)

        # Smooth with previous action
        if self._prev_action is not None:
            action = (
                self._smoothing * self._prev_action
                + (1.0 - self._smoothing) * action
            ).astype(np.float32)

        self._prev_action = action.copy()
        return action

    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            name="ScriptedReachPolicy",
            observation_space={
                "joint_pos": {"shape": (7,)},
                "joint_vel": {"shape": (7,)},
                "ee_pos": {"shape": (3,)},
                "target_pos": {"shape": (3,)},
            },
            action_space={"dim": 7, "low": -1.0, "high": 1.0},
            modalities=["proprioception"],
        )
