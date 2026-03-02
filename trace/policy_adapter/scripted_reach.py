"""Scripted reach policy — proportional controller using Jacobian transpose.

Uses MuJoCo's mj_jacSite to compute the real end-effector Jacobian,
then applies J^T * error to get joint-space corrections. These are
added to the current joint positions and mapped to the [-1, 1] action
space. Produces directed motion toward the target — enough for testing
the pipeline and measuring degradation under stress.
"""

import mujoco
import numpy as np
from numpy.typing import NDArray

from trace.policy_adapter.base import BasePolicy, PolicyMetadata
from trace.task_spec.base import Observation


class ScriptedReachPolicy(BasePolicy):
    """Proportional controller that moves the end-effector toward the target."""

    def __init__(self, gain: float = 10.0, smoothing: float = 0.1) -> None:
        self._gain = gain
        self._smoothing = smoothing
        self._prev_action: NDArray[np.floating] | None = None
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._ee_site_id: int = -1

    def set_env(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Provide MuJoCo model/data for Jacobian computation."""
        self._model = model
        self._data = data
        self._ee_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "end_effector"
        )

    def load(self, checkpoint_path: str) -> None:
        pass  # No weights to load

    def reset(self) -> None:
        self._prev_action = None

    def act(self, observation: Observation) -> np.ndarray:
        ee_pos = observation["ee_pos"]
        target_pos = observation["target_pos"]
        joint_pos = observation["joint_pos"]

        error = target_pos - ee_pos  # Cartesian error (3,)

        # Compute the real Jacobian via MuJoCo
        assert self._model is not None and self._data is not None, (
            "Call set_env() before act()"
        )
        jacp = np.zeros((3, self._model.nv))
        mujoco.mj_jacSite(self._model, self._data, jacp, None, self._ee_site_id)

        # Jacobian transpose: map Cartesian error to joint-space delta
        joint_delta = jacp.T @ error

        # Desired joint positions = current + scaled delta
        desired_joint_pos = joint_pos + self._gain * joint_delta

        # Convert desired joint positions to [-1, 1] action space
        ctrl_range = self._model.actuator_ctrlrange
        action = (
            2.0 * (desired_joint_pos - ctrl_range[:, 0])
            / (ctrl_range[:, 1] - ctrl_range[:, 0])
            - 1.0
        )
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

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
