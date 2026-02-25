"""Reach task — move end-effector to a randomized target position.

The simplest useful MuJoCo task: a 7-DOF arm must position its
end-effector within a success radius of a randomly placed target.
"""

from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from trace.task_spec.base import BaseTask, TaskConfig
from trace.task_spec.mjcf_models import PANDA_7DOF_REACH_XML

# Default workspace bounds for target randomization (x, y, z)
_TARGET_X_RANGE = (0.2, 0.6)
_TARGET_Y_RANGE = (-0.3, 0.3)
_TARGET_Z_RANGE = (0.1, 0.5)

# Physics substeps per control step (model timestep=0.002s, 25 substeps → 20Hz control)
_N_SUBSTEPS = 25

# Panda-like home configuration (all joints within valid ranges)
_HOME_QPOS = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])


class ReachTask(BaseTask):
    """Move the end-effector to a randomized target position."""

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._ee_site_id: int = -1
        self._target_body_id: int = -1
        self._rng: np.random.Generator | None = None

        # Task parameters from config
        self._success_radius: float = config.task_params.get("success_radius", 0.05)
        self._catastrophic_vel_threshold: float = config.task_params.get(
            "catastrophic_vel_threshold", 50.0
        )
        self._target_x_range: tuple[float, float] = tuple(
            config.task_params.get("target_x_range", _TARGET_X_RANGE)
        )
        self._target_y_range: tuple[float, float] = tuple(
            config.task_params.get("target_y_range", _TARGET_Y_RANGE)
        )
        self._target_z_range: tuple[float, float] = tuple(
            config.task_params.get("target_z_range", _TARGET_Z_RANGE)
        )

        # Cached initial physics params for clean reset
        self._initial_body_mass: NDArray[np.floating] | None = None
        self._initial_geom_friction: NDArray[np.floating] | None = None
        self._initial_dof_damping: NDArray[np.floating] | None = None
        self._initial_geom_size: NDArray[np.floating] | None = None
        self._initial_jnt_range: NDArray[np.floating] | None = None
        self._initial_actuator_gainprm: NDArray[np.floating] | None = None

    def initialize(self) -> None:
        self._model = mujoco.MjModel.from_xml_string(PANDA_7DOF_REACH_XML)
        self._data = mujoco.MjData(self._model)

        self._ee_site_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, "end_effector"
        )
        self._target_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )

        # Cache initial physics params so reset can restore them
        self._initial_body_mass = self._model.body_mass.copy()
        self._initial_geom_friction = self._model.geom_friction.copy()
        self._initial_dof_damping = self._model.dof_damping.copy()
        self._initial_geom_size = self._model.geom_size.copy()
        self._initial_jnt_range = self._model.jnt_range.copy()
        self._initial_actuator_gainprm = self._model.actuator_gainprm.copy()

        self._rng = np.random.default_rng(self.config.seed)
        mujoco.mj_forward(self._model, self._data)

    def reset(self, seed: int | None = None) -> dict[str, NDArray[np.floating]]:
        assert self._model is not None and self._data is not None, "Call initialize() first"

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Reset simulation state
        mujoco.mj_resetData(self._model, self._data)

        # Restore cached physics params (stressors may have modified them)
        self._model.body_mass[:] = self._initial_body_mass
        self._model.geom_friction[:] = self._initial_geom_friction
        self._model.dof_damping[:] = self._initial_dof_damping
        self._model.geom_size[:] = self._initial_geom_size
        self._model.jnt_range[:] = self._initial_jnt_range
        self._model.actuator_gainprm[:] = self._initial_actuator_gainprm

        # Set valid home configuration (zero-config violates joint4 limits)
        self._data.qpos[:self._model.nq] = _HOME_QPOS
        self._data.ctrl[:] = _HOME_QPOS

        # Randomize target position within workspace
        target_pos = np.array([
            self._rng.uniform(*self._target_x_range),
            self._rng.uniform(*self._target_y_range),
            self._rng.uniform(*self._target_z_range),
        ])
        # The target body is the first (and only) mocap body
        self._data.mocap_pos[0] = target_pos

        mujoco.mj_forward(self._model, self._data)
        return self.get_observation()

    def step(
        self, action: NDArray[np.floating]
    ) -> tuple[dict[str, NDArray[np.floating]], float, bool, dict[str, Any]]:
        assert self._model is not None and self._data is not None

        # Map [-1, 1] actions to actuator ctrl ranges
        ctrl_range = self._model.actuator_ctrlrange
        action_clipped = np.clip(action[:self._model.nu], -1.0, 1.0)
        ctrl = (
            ctrl_range[:, 0]
            + (action_clipped + 1.0) * 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
        )
        self._data.ctrl[:] = ctrl

        # Run substeps
        for _ in range(_N_SUBSTEPS):
            mujoco.mj_step(self._model, self._data)

        obs = self.get_observation()
        distance = self._ee_target_distance()
        reward = -distance
        done = self.check_success() or self.check_catastrophic_failure()
        info: dict[str, Any] = {"distance_to_target": distance}

        return obs, reward, done, info

    def check_success(self) -> bool:
        return self._ee_target_distance() < self._success_radius

    def check_catastrophic_failure(self) -> bool:
        assert self._data is not None
        return bool(np.any(np.abs(self._data.qvel) > self._catastrophic_vel_threshold))

    def get_observation(self) -> dict[str, NDArray[np.floating]]:
        assert self._model is not None and self._data is not None
        return {
            "joint_pos": self._data.qpos[:self._model.nq].astype(np.float32).copy(),
            "joint_vel": self._data.qvel[:self._model.nv].astype(np.float32).copy(),
            "ee_pos": self._data.site_xpos[self._ee_site_id].astype(np.float32).copy(),
            "target_pos": self._data.mocap_pos[0].astype(np.float32).copy(),
        }

    def get_mujoco_model(self) -> mujoco.MjModel:
        assert self._model is not None, "Call initialize() first"
        return self._model

    def get_mujoco_data(self) -> mujoco.MjData:
        assert self._data is not None, "Call initialize() first"
        return self._data

    def _ee_target_distance(self) -> float:
        assert self._data is not None
        ee = self._data.site_xpos[self._ee_site_id]
        target = self._data.mocap_pos[0]
        return float(np.linalg.norm(ee - target))
