"""Tests for GR00T N1 policy adapter."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from trace.policy_adapter.groot_adapter import GR00TAdapter, _quat2axisangle


@pytest.fixture
def adapter() -> GR00TAdapter:
    adapter = GR00TAdapter(chunk_size=4)
    adapter.set_task_info("pick up the black bowl")
    return adapter


@pytest.fixture
def adapter_with_mock() -> GR00TAdapter:
    adapter = GR00TAdapter(chunk_size=4)
    adapter.set_task_info("pick up the black bowl")

    mock_client = MagicMock()
    # Simulate GR00T action dict response: (action_dict, info)
    action_dict = {
        "action.x": np.zeros((1, 16, 1), dtype=np.float32),
        "action.y": np.ones((1, 16, 1), dtype=np.float32) * 0.1,
        "action.z": np.zeros((1, 16, 1), dtype=np.float32),
        "action.roll": np.zeros((1, 16, 1), dtype=np.float32),
        "action.pitch": np.zeros((1, 16, 1), dtype=np.float32),
        "action.yaw": np.zeros((1, 16, 1), dtype=np.float32),
        "action.gripper": np.ones((1, 16, 1), dtype=np.float32) * 0.5,
    }
    mock_client.get_action.return_value = (action_dict, {})
    adapter._client = mock_client
    return adapter


def _make_obs() -> dict[str, np.ndarray]:
    return {
        "image": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
        "wrist_image": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
        "ee_pos": np.array([0.5, 0.0, 0.3], dtype=np.float32),
        "ee_orientation": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "gripper": np.array([0.04, 0.04], dtype=np.float32),
    }


class TestGR00TAdapter:
    def test_metadata(self, adapter: GR00TAdapter) -> None:
        meta = adapter.metadata()
        assert meta.name == "GR00TAdapter"
        assert "image" in meta.observation_space
        assert "wrist_image" in meta.observation_space
        assert meta.action_space["dim"] == 7

    def test_reset_clears_buffer(self, adapter_with_mock: GR00TAdapter) -> None:
        obs = _make_obs()
        adapter_with_mock.act(obs)
        assert len(adapter_with_mock._action_buffer) > 0
        adapter_with_mock.reset()
        assert len(adapter_with_mock._action_buffer) == 0

    def test_act_returns_correct_shape(self, adapter_with_mock: GR00TAdapter) -> None:
        obs = _make_obs()
        action = adapter_with_mock.act(obs)
        assert action.shape == (7,)
        assert action.dtype == np.float32

    def test_action_chunking(self, adapter_with_mock: GR00TAdapter) -> None:
        obs = _make_obs()
        # First call queries server, fills buffer with chunk_size=4
        action1 = adapter_with_mock.act(obs)
        assert adapter_with_mock._client.get_action.call_count == 1

        # Next 3 calls pop from buffer without server call
        for _ in range(3):
            adapter_with_mock.act(obs)
        assert adapter_with_mock._client.get_action.call_count == 1

        # 5th call triggers new server query
        adapter_with_mock.act(obs)
        assert adapter_with_mock._client.get_action.call_count == 2

    def test_build_observation_format(self, adapter: GR00TAdapter) -> None:
        obs = _make_obs()
        groot_obs = adapter._build_observation(obs)

        # Video: (1, 1, H, W, 3)
        assert groot_obs["video.image"].shape == (1, 1, 256, 256, 3)
        assert groot_obs["video.wrist_image"].shape == (1, 1, 256, 256, 3)

        # State: (1, 1, D)
        assert groot_obs["state.x"].shape == (1, 1, 1)
        assert groot_obs["state.y"].shape == (1, 1, 1)
        assert groot_obs["state.z"].shape == (1, 1, 1)
        assert groot_obs["state.roll"].shape == (1, 1, 1)
        assert groot_obs["state.pitch"].shape == (1, 1, 1)
        assert groot_obs["state.yaw"].shape == (1, 1, 1)
        assert groot_obs["state.gripper"].shape == (1, 1, 2)

        # Language
        assert groot_obs["annotation.human.action.task_description"] == "pick up the black bowl"

    def test_image_flip(self, adapter: GR00TAdapter) -> None:
        obs = _make_obs()
        obs["image"][0, 0, :] = [255, 0, 0]  # Red pixel at top-left
        groot_obs = adapter._build_observation(obs)
        # After 180° rotation, top-left should now be at bottom-right
        assert np.array_equal(groot_obs["video.image"][0, 0, -1, -1, :], [255, 0, 0])

    def test_state_axis_angle_conversion(self, adapter: GR00TAdapter) -> None:
        obs = _make_obs()
        # Identity quaternion [0,0,0,1] should give zero axis-angle
        obs["ee_orientation"] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        groot_obs = adapter._build_observation(obs)
        assert groot_obs["state.roll"][0, 0, 0] == pytest.approx(0.0, abs=1e-6)
        assert groot_obs["state.pitch"][0, 0, 0] == pytest.approx(0.0, abs=1e-6)
        assert groot_obs["state.yaw"][0, 0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_gripper_postprocess(self, adapter_with_mock: GR00TAdapter) -> None:
        obs = _make_obs()
        action = adapter_with_mock.act(obs)
        # Gripper input was 0.5 -> normalize: 2*0.5-1=0 -> invert: -0 = 0
        assert action[6] == pytest.approx(0.0, abs=1e-6)

    def test_gripper_open_postprocess(self, adapter: GR00TAdapter) -> None:
        # Gripper=1.0 (RLDS open) -> normalize: 2*1-1=1 -> invert: -1 (LIBERO open)
        raw = np.array([0, 0, 0, 0, 0, 0, 1.0], dtype=np.float32)
        processed = adapter._postprocess_action(raw)
        assert processed[6] == pytest.approx(-1.0, abs=1e-6)

    def test_gripper_close_postprocess(self, adapter: GR00TAdapter) -> None:
        # Gripper=0.0 (RLDS close) -> normalize: 2*0-1=-1 -> invert: 1 (LIBERO close)
        raw = np.array([0, 0, 0, 0, 0, 0, 0.0], dtype=np.float32)
        processed = adapter._postprocess_action(raw)
        assert processed[6] == pytest.approx(1.0, abs=1e-6)

    def test_parse_action_dict(self, adapter: GR00TAdapter) -> None:
        action_dict = {
            "action.x": np.ones((1, 8, 1), dtype=np.float32) * 0.1,
            "action.y": np.ones((1, 8, 1), dtype=np.float32) * 0.2,
            "action.z": np.ones((1, 8, 1), dtype=np.float32) * 0.3,
            "action.roll": np.zeros((1, 8, 1), dtype=np.float32),
            "action.pitch": np.zeros((1, 8, 1), dtype=np.float32),
            "action.yaw": np.zeros((1, 8, 1), dtype=np.float32),
            "action.gripper": np.ones((1, 8, 1), dtype=np.float32) * 0.5,
        }
        actions = adapter._parse_action_dict(action_dict)
        assert actions.shape == (8, 7)
        assert actions[0, 0] == pytest.approx(0.1)
        assert actions[0, 1] == pytest.approx(0.2)
        assert actions[0, 2] == pytest.approx(0.3)
        assert actions[0, 6] == pytest.approx(0.5)

    def test_no_server_returns_zeros(self, adapter: GR00TAdapter) -> None:
        adapter._client = None
        obs = _make_obs()
        action = adapter.act(obs)
        assert action.shape == (7,)
        np.testing.assert_array_equal(action, np.zeros(7))

    def test_set_task_info(self, adapter: GR00TAdapter) -> None:
        adapter.set_task_info("open the drawer")
        assert adapter._prompt == "open the drawer"

    def test_missing_obs_keys(self, adapter_with_mock: GR00TAdapter) -> None:
        # Should handle missing observation keys gracefully
        obs: dict[str, np.ndarray] = {}
        action = adapter_with_mock.act(obs)
        assert action.shape == (7,)


class TestQuat2AxisAngle:
    def test_identity(self) -> None:
        result = _quat2axisangle(np.array([0, 0, 0, 1], dtype=np.float32))
        np.testing.assert_allclose(result, [0, 0, 0], atol=1e-6)

    def test_90deg_z(self) -> None:
        # 90 degrees around Z axis
        angle = np.pi / 2
        quat = np.array([0, 0, np.sin(angle / 2), np.cos(angle / 2)], dtype=np.float32)
        result = _quat2axisangle(quat)
        expected = np.array([0, 0, angle], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-5)
