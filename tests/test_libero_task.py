"""Tests for LiberoTask integration.

These tests require LIBERO to be installed. Skip gracefully if not available.
"""

import numpy as np
import pytest

# Skip all tests if LIBERO is not installed
libero = pytest.importorskip("libero")

from trace.task_spec.base import TaskConfig
from trace.task_spec.libero_task import LiberoTask


@pytest.fixture
def libero_config():
    return TaskConfig(
        name="libero",
        max_episode_steps=220,
        success_threshold=0.95,
        seed=0,
        task_params={
            "task_suite_name": "libero_spatial",
            "task_id": 0,
            "num_steps_wait": 10,
            "render_width": 224,
            "render_height": 224,
            "language_instruction": "auto",
        },
    )


@pytest.fixture
def libero_task(libero_config):
    task = LiberoTask(libero_config)
    task.initialize()
    yield task
    task.close()


class TestLiberoTaskInit:
    def test_initialize(self, libero_task):
        """Task initializes without error."""
        assert libero_task._env is not None
        assert libero_task._task_suite is not None

    def test_language_instruction(self, libero_task):
        """Language instruction is loaded from LIBERO benchmark."""
        assert len(libero_task.language_instruction) > 0
        assert libero_task.language_instruction != "auto"


class TestLiberoTaskReset:
    def test_reset_returns_observation(self, libero_task):
        obs = libero_task.reset(seed=0)
        assert isinstance(obs, dict)

    def test_observation_keys(self, libero_task):
        obs = libero_task.reset(seed=0)
        assert "image" in obs
        assert "wrist_image" in obs
        assert "ee_pos" in obs
        assert "ee_orientation" in obs
        assert "gripper" in obs

    def test_image_shape(self, libero_task):
        obs = libero_task.reset(seed=0)
        assert obs["image"].shape == (224, 224, 3)
        assert obs["image"].dtype == np.uint8
        assert obs["wrist_image"].shape == (224, 224, 3)

    def test_proprioception_shapes(self, libero_task):
        obs = libero_task.reset(seed=0)
        assert obs["ee_pos"].shape == (3,)
        assert obs["ee_orientation"].shape == (4,)
        assert obs["gripper"].shape == (2,)


class TestLiberoTaskStep:
    def test_step_returns_tuple(self, libero_task):
        libero_task.reset(seed=0)
        action = np.zeros(7, dtype=np.float32)
        obs, reward, done, info = libero_task.step(action)
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_observation_format(self, libero_task):
        libero_task.reset(seed=0)
        action = np.zeros(7, dtype=np.float32)
        obs, _, _, _ = libero_task.step(action)
        assert "image" in obs
        assert obs["image"].shape == (224, 224, 3)


class TestLiberoTaskMujoco:
    def test_get_mujoco_model(self, libero_task):
        libero_task.reset(seed=0)
        model = libero_task.get_mujoco_model()
        assert model is not None

    def test_get_mujoco_data(self, libero_task):
        libero_task.reset(seed=0)
        data = libero_task.get_mujoco_data()
        assert data is not None


class TestLiberoTaskSuccess:
    def test_initial_not_success(self, libero_task):
        libero_task.reset(seed=0)
        assert not libero_task.check_success()

    def test_no_catastrophic_failure(self, libero_task):
        libero_task.reset(seed=0)
        assert not libero_task.check_catastrophic_failure()
