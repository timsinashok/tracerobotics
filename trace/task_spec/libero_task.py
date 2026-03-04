"""LIBERO task wrapper — wraps LIBERO environments into Trace's BaseTask interface.

Supports all 5 LIBERO suites (libero_spatial, libero_object, libero_goal,
libero_10, libero_90) with 10-90 tasks each. Each task is a tabletop
manipulation scenario with a Franka Panda arm.
"""

import math
from typing import Any

import numpy as np
from PIL import Image

from trace.task_spec.base import BaseTask, Observation, TaskConfig


class LiberoTask(BaseTask):
    """Wraps a LIBERO environment into Trace's BaseTask interface."""

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)
        self._env: Any = None
        self._task_suite: Any = None
        self._initial_states: Any = None
        self._episode_idx: int = 0
        self._done: bool = False
        self._last_reward: float = 0.0

        # Task parameters
        params = config.task_params
        self._suite_name: str = params.get("task_suite_name", "libero_spatial")
        self._task_id: int = params.get("task_id", 0)
        self._num_steps_wait: int = params.get("num_steps_wait", 10)
        self._render_width: int = params.get("render_width", 224)
        self._render_height: int = params.get("render_height", 224)
        self._task_description: str = ""

        # Cached last observation from LIBERO env
        self._last_obs: dict[str, Any] | None = None

    def initialize(self) -> None:
        """Create the LIBERO environment from benchmark."""
        import pathlib

        from libero.libero import benchmark as libero_benchmark
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        # Load benchmark and task suite
        benchmark_dict = libero_benchmark.get_benchmark_dict()
        self._task_suite = benchmark_dict[self._suite_name]()
        task = self._task_suite.get_task(self._task_id)

        # Get task description (language instruction)
        self._task_description = task.language
        lang = self.config.task_params.get("language_instruction", "auto")
        if lang != "auto":
            self._task_description = lang

        # Create environment
        task_bddl_file = (
            pathlib.Path(get_libero_path("bddl_files"))
            / task.problem_folder
            / task.bddl_file
        )
        self._env = OffScreenRenderEnv(
            bddl_file_name=str(task_bddl_file),
            camera_heights=256,  # Native LIBERO resolution
            camera_widths=256,
        )
        self._env.seed(self.config.seed)

        # Get initial states for reproducible episodes
        self._initial_states = self._task_suite.get_task_init_states(self._task_id)

    def reset(self, seed: int | None = None) -> Observation:
        """Reset the LIBERO environment and return initial observation."""
        assert self._env is not None, "Call initialize() first"

        if seed is not None:
            self._env.seed(seed)
            self._episode_idx = seed % len(self._initial_states)
        else:
            self._episode_idx = (self._episode_idx + 1) % len(self._initial_states)

        self._done = False
        self._last_reward = 0.0

        # Reset and set initial state
        self._env.reset()
        obs = self._env.set_init_state(self._initial_states[self._episode_idx])

        # Wait for objects to settle (LIBERO drops objects at start)
        dummy_action = [0.0] * 6 + [-1.0]
        for _ in range(self._num_steps_wait):
            obs, _, _, _ = self._env.step(dummy_action)

        self._last_obs = obs
        return self.get_observation()

    def step(
        self, action: np.ndarray
    ) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Execute one action in the LIBERO environment.

        Action is 7-dim [arm_delta(6), gripper(1)] — passed directly to env.
        """
        assert self._env is not None

        action_list = action[:7].tolist()
        obs, reward, done, info = self._env.step(action_list)

        self._last_obs = obs
        self._done = bool(done)
        self._last_reward = float(reward)

        trace_obs = self.get_observation()
        return trace_obs, self._last_reward, self._done, info

    def check_success(self) -> bool:
        """Return True if the LIBERO task is completed."""
        return self._done

    def check_catastrophic_failure(self) -> bool:
        """LIBERO tasks don't define catastrophic failures."""
        return False

    def get_observation(self) -> Observation:
        """Map LIBERO observation keys to Trace format."""
        assert self._last_obs is not None

        obs: Observation = {}

        # Images: resize from 256 native to configured size (typically 224)
        if "agentview_image" in self._last_obs:
            obs["image"] = self._resize_image(self._last_obs["agentview_image"])
        if "robot0_eye_in_hand_image" in self._last_obs:
            obs["wrist_image"] = self._resize_image(
                self._last_obs["robot0_eye_in_hand_image"]
            )

        # Proprioception
        if "robot0_eef_pos" in self._last_obs:
            obs["ee_pos"] = self._last_obs["robot0_eef_pos"].astype(np.float32).copy()
        if "robot0_eef_quat" in self._last_obs:
            obs["ee_orientation"] = (
                self._last_obs["robot0_eef_quat"].astype(np.float32).copy()
            )
        if "robot0_gripper_qpos" in self._last_obs:
            obs["gripper"] = (
                self._last_obs["robot0_gripper_qpos"].astype(np.float32).copy()
            )

        return obs

    @property
    def language_instruction(self) -> str:
        """Return the LIBERO task description as the language instruction."""
        return self._task_description

    def get_mujoco_model(self) -> Any:
        """Return the underlying MuJoCo model from robosuite."""
        assert self._env is not None, "Call initialize() first"
        return self._env.sim.model._model

    def get_mujoco_data(self) -> Any:
        """Return the underlying MuJoCo data from robosuite."""
        assert self._env is not None, "Call initialize() first"
        return self._env.sim.data._data

    def close(self) -> None:
        """Clean up the LIBERO environment."""
        if self._env is not None:
            self._env.close()
            self._env = None

    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """Resize image from native 256x256 to configured resolution."""
        if img.shape[0] == self._render_height and img.shape[1] == self._render_width:
            return img.copy()
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize(
            (self._render_width, self._render_height), Image.BILINEAR
        )
        return np.array(pil_img, dtype=np.uint8)
