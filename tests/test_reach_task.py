"""Tests for the ReachTask MuJoCo environment."""

import numpy as np
import pytest

from trace.task_spec.base import TaskConfig
from trace.task_spec.reach import ReachTask


@pytest.fixture
def reach_config() -> TaskConfig:
    return TaskConfig(
        name="reach",
        max_episode_steps=200,
        seed=42,
        task_params={"success_radius": 0.05},
    )


@pytest.fixture
def reach_task(reach_config: TaskConfig) -> ReachTask:
    task = ReachTask(reach_config)
    task.initialize()
    return task


class TestInitialization:
    def test_model_dimensions(self, reach_task: ReachTask) -> None:
        model = reach_task.get_mujoco_model()
        assert model.nq == 7
        assert model.nv == 7
        assert model.nu == 7

    def test_site_and_body_ids_valid(self, reach_task: ReachTask) -> None:
        assert reach_task._ee_site_id >= 0
        assert reach_task._target_body_id >= 0

    def test_initial_physics_cached(self, reach_task: ReachTask) -> None:
        assert reach_task._initial_body_mass is not None
        assert reach_task._initial_geom_friction is not None
        assert reach_task._initial_dof_damping is not None
        assert reach_task._initial_geom_size is not None
        assert reach_task._initial_jnt_range is not None
        assert reach_task._initial_actuator_gainprm is not None


class TestReset:
    def test_observation_keys(self, reach_task: ReachTask) -> None:
        obs = reach_task.reset(seed=0)
        expected_keys = {"joint_pos", "joint_vel", "ee_pos", "target_pos"}
        assert set(obs.keys()) == expected_keys

    def test_observation_shapes(self, reach_task: ReachTask) -> None:
        obs = reach_task.reset(seed=0)
        assert obs["joint_pos"].shape == (7,)
        assert obs["joint_vel"].shape == (7,)
        assert obs["ee_pos"].shape == (3,)
        assert obs["target_pos"].shape == (3,)

    def test_observation_dtypes(self, reach_task: ReachTask) -> None:
        obs = reach_task.reset(seed=0)
        for key, val in obs.items():
            assert val.dtype == np.float32, f"{key} has wrong dtype: {val.dtype}"

    def test_reproducibility_same_seed(self, reach_task: ReachTask) -> None:
        obs1 = reach_task.reset(seed=123)
        obs2 = reach_task.reset(seed=123)
        for key in obs1:
            np.testing.assert_array_equal(obs1[key], obs2[key])

    def test_different_seeds_produce_different_targets(self, reach_task: ReachTask) -> None:
        obs1 = reach_task.reset(seed=0)
        obs2 = reach_task.reset(seed=999)
        assert not np.allclose(obs1["target_pos"], obs2["target_pos"])

    def test_target_within_workspace(self, reach_task: ReachTask) -> None:
        for seed in range(20):
            obs = reach_task.reset(seed=seed)
            t = obs["target_pos"]
            assert 0.2 <= t[0] <= 0.6, f"target x out of range: {t[0]}"
            assert -0.3 <= t[1] <= 0.3, f"target y out of range: {t[1]}"
            assert 0.1 <= t[2] <= 0.5, f"target z out of range: {t[2]}"

    def test_reset_restores_physics_after_modification(self, reach_task: ReachTask) -> None:
        model = reach_task.get_mujoco_model()
        original_mass = model.body_mass.copy()
        # Simulate a stressor modifying physics
        model.body_mass[:] *= 2.0
        reach_task.reset(seed=0)
        np.testing.assert_array_almost_equal(model.body_mass, original_mass)


class TestStep:
    def test_step_returns_correct_types(self, reach_task: ReachTask) -> None:
        reach_task.reset(seed=0)
        action = np.zeros(7, dtype=np.float32)
        obs, reward, done, info = reach_task.step(action)
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert "distance_to_target" in info

    def test_step_changes_state(self, reach_task: ReachTask) -> None:
        obs_before = reach_task.reset(seed=0)
        action = np.ones(7, dtype=np.float32)
        obs_after, _, _, _ = reach_task.step(action)
        assert not np.allclose(obs_before["joint_pos"], obs_after["joint_pos"])

    def test_reward_is_negative_distance(self, reach_task: ReachTask) -> None:
        reach_task.reset(seed=0)
        action = np.zeros(7, dtype=np.float32)
        _, reward, _, info = reach_task.step(action)
        assert reward == pytest.approx(-info["distance_to_target"])

    def test_action_clipping(self, reach_task: ReachTask) -> None:
        reach_task.reset(seed=0)
        large_action = np.full(7, 100.0, dtype=np.float32)
        # Should not raise — actions get clipped to [-1, 1]
        obs, reward, done, info = reach_task.step(large_action)
        assert np.all(np.isfinite(obs["joint_pos"]))


class TestSuccessAndFailure:
    def test_not_success_at_start(self, reach_task: ReachTask) -> None:
        reach_task.reset(seed=0)
        assert not reach_task.check_success()

    def test_not_catastrophic_at_start(self, reach_task: ReachTask) -> None:
        reach_task.reset(seed=0)
        assert not reach_task.check_catastrophic_failure()

    def test_success_when_ee_near_target(self, reach_task: ReachTask) -> None:
        reach_task.reset(seed=0)
        data = reach_task.get_mujoco_data()
        # Teleport target to current EE position
        import mujoco
        ee_pos = data.site_xpos[reach_task._ee_site_id].copy()
        data.mocap_pos[0] = ee_pos
        mujoco.mj_forward(reach_task.get_mujoco_model(), data)
        assert reach_task.check_success()


class TestStressorCompatibility:
    def test_model_properties_exist(self, reach_task: ReachTask) -> None:
        model = reach_task.get_mujoco_model()
        assert hasattr(model, "body_mass")
        assert hasattr(model, "geom_friction")
        assert hasattr(model, "dof_damping")
        assert hasattr(model, "geom_size")
        assert hasattr(model, "jnt_range")
        assert hasattr(model, "actuator_gainprm")

    def test_model_properties_are_writable(self, reach_task: ReachTask) -> None:
        model = reach_task.get_mujoco_model()
        original = model.body_mass.copy()
        model.body_mass[:] *= 1.5
        assert not np.allclose(model.body_mass, original)
        # Restore
        model.body_mass[:] = original

    def test_get_mujoco_model_returns_model(self, reach_task: ReachTask) -> None:
        import mujoco
        model = reach_task.get_mujoco_model()
        assert isinstance(model, mujoco.MjModel)

    def test_get_mujoco_data_returns_data(self, reach_task: ReachTask) -> None:
        import mujoco
        data = reach_task.get_mujoco_data()
        assert isinstance(data, mujoco.MjData)
