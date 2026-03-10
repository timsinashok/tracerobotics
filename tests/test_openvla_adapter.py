"""Tests for OpenVLA-OFT policy adapter."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from trace.policy_adapter.openvla_adapter import OpenVLAAdapter, _quat2axisangle


@pytest.fixture
def adapter() -> OpenVLAAdapter:
    adapter = OpenVLAAdapter(chunk_size=4)
    adapter.set_task_info("pick up the black bowl")
    return adapter


@pytest.fixture
def adapter_with_mock() -> OpenVLAAdapter:
    """Adapter with mocked model that returns fake actions."""
    adapter = OpenVLAAdapter(chunk_size=4)
    adapter.set_task_info("pick up the black bowl")
    adapter._loaded = True
    adapter._cfg = MagicMock()
    adapter._model = MagicMock()
    adapter._processor = MagicMock()
    adapter._action_head = MagicMock()
    adapter._proprio_projector = MagicMock()
    return adapter


def _make_obs() -> dict[str, np.ndarray]:
    return {
        "image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "ee_pos": np.array([0.5, 0.0, 0.3], dtype=np.float32),
        "ee_orientation": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "gripper": np.array([0.04, 0.04], dtype=np.float32),
    }


class TestOpenVLAAdapter:
    def test_metadata(self, adapter: OpenVLAAdapter) -> None:
        meta = adapter.metadata()
        assert meta.name == "OpenVLAAdapter"
        assert "image" in meta.observation_space
        assert "wrist_image" in meta.observation_space
        assert meta.action_space["dim"] == 7

    def test_reset_clears_buffer(self, adapter: OpenVLAAdapter) -> None:
        # Manually add to buffer
        adapter._action_buffer.append(np.zeros(7))
        adapter._action_buffer.append(np.zeros(7))
        adapter.reset()
        assert len(adapter._action_buffer) == 0

    def test_no_model_returns_zeros(self, adapter: OpenVLAAdapter) -> None:
        adapter._loaded = False
        obs = _make_obs()
        action = adapter.act(obs)
        assert action.shape == (7,)
        np.testing.assert_array_equal(action, np.zeros(7))

    def test_build_observation_format(self, adapter: OpenVLAAdapter) -> None:
        obs = _make_obs()
        openvla_obs = adapter._build_observation(obs)

        # Images should be present
        assert "full_image" in openvla_obs
        assert "wrist_image" in openvla_obs
        assert openvla_obs["full_image"].dtype == np.uint8
        assert openvla_obs["wrist_image"].dtype == np.uint8

        # State: 8-dim [ee_pos(3), axis_angle(3), gripper_qpos(2)]
        assert "state" in openvla_obs
        assert openvla_obs["state"].shape == (8,)
        assert openvla_obs["state"].dtype == np.float32

        # Task description
        assert openvla_obs["task_description"] == "pick up the black bowl"

    def test_image_flip(self, adapter: OpenVLAAdapter) -> None:
        obs = _make_obs()
        obs["image"][0, 0, :] = [255, 0, 0]  # Red pixel at top-left
        openvla_obs = adapter._build_observation(obs)
        # After 180° rotation + center crop, check image is transformed
        assert openvla_obs["full_image"].shape[0] < 224  # cropped

    def test_center_crop(self, adapter: OpenVLAAdapter) -> None:
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        cropped = adapter._center_crop_image(img, crop_ratio=0.9)
        expected_size = int(256 * 0.9)
        assert cropped.shape == (expected_size, expected_size, 3)

    def test_center_crop_disabled(self) -> None:
        adapter = OpenVLAAdapter(center_crop=False)
        adapter.set_task_info("test")
        obs = _make_obs()
        openvla_obs = adapter._build_observation(obs)
        # Without center crop, image keeps original size after rotation
        assert openvla_obs["full_image"].shape[0] == 224

    def test_state_axis_angle_conversion(self, adapter: OpenVLAAdapter) -> None:
        obs = _make_obs()
        obs["ee_orientation"] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        openvla_obs = adapter._build_observation(obs)
        state = openvla_obs["state"]
        # Identity quaternion -> zero axis-angle
        np.testing.assert_allclose(state[3:6], [0, 0, 0], atol=1e-6)
        # EE pos should be first 3
        np.testing.assert_allclose(state[:3], [0.5, 0.0, 0.3], atol=1e-6)
        # Gripper should be last 2
        np.testing.assert_allclose(state[6:8], [0.04, 0.04], atol=1e-6)

    def test_gripper_open_postprocess(self, adapter: OpenVLAAdapter) -> None:
        # Gripper=1.0 (RLDS open) -> normalize: 2*1-1=1 -> sign: 1 -> invert: -1 (LIBERO open)
        raw = np.array([0, 0, 0, 0, 0, 0, 1.0], dtype=np.float32)
        processed = adapter._postprocess_action(raw)
        assert processed[6] == pytest.approx(-1.0, abs=1e-6)

    def test_gripper_close_postprocess(self, adapter: OpenVLAAdapter) -> None:
        # Gripper=0.0 (RLDS close) -> normalize: 2*0-1=-1 -> sign: -1 -> invert: 1 (LIBERO close)
        raw = np.array([0, 0, 0, 0, 0, 0, 0.0], dtype=np.float32)
        processed = adapter._postprocess_action(raw)
        assert processed[6] == pytest.approx(1.0, abs=1e-6)

    def test_gripper_mid_postprocess(self, adapter: OpenVLAAdapter) -> None:
        # Gripper=0.5 -> normalize: 0 -> sign: 0 -> invert: 0
        raw = np.array([0, 0, 0, 0, 0, 0, 0.5], dtype=np.float32)
        processed = adapter._postprocess_action(raw)
        assert processed[6] == pytest.approx(0.0, abs=1e-6)

    def test_set_task_info(self, adapter: OpenVLAAdapter) -> None:
        adapter.set_task_info("open the drawer")
        assert adapter._prompt == "open the drawer"

    @patch("trace.policy_adapter.openvla_adapter.OpenVLAAdapter._load_model")
    def test_load_sets_loaded(self, mock_load: MagicMock, adapter: OpenVLAAdapter) -> None:
        adapter.load("test_checkpoint")
        assert adapter._loaded
        assert adapter._checkpoint == "test_checkpoint"

    def test_act_with_mock(self, adapter_with_mock: OpenVLAAdapter) -> None:
        obs = _make_obs()
        # Mock the get_vla_action to return fake actions
        fake_actions = [np.array([0.1, 0.2, 0.3, 0, 0, 0, 0.5], dtype=np.float32) for _ in range(8)]

        with patch(
            "trace.policy_adapter.openvla_adapter.get_vla_action",
            return_value=fake_actions,
            create=True,
        ) as mock_get_action:
            # Patch the import inside _refill_buffer
            import sys
            mock_module = MagicMock()
            mock_module.get_vla_action = MagicMock(return_value=fake_actions)
            with patch.dict(sys.modules, {"experiments.robot.openvla_utils": mock_module}):
                action = adapter_with_mock.act(obs)
                assert action.shape == (7,)
                assert action.dtype == np.float32

    def test_missing_obs_keys(self, adapter: OpenVLAAdapter) -> None:
        adapter._loaded = False
        obs: dict[str, np.ndarray] = {}
        action = adapter.act(obs)
        assert action.shape == (7,)


class TestQuat2AxisAngle:
    def test_identity(self) -> None:
        result = _quat2axisangle(np.array([0, 0, 0, 1], dtype=np.float32))
        np.testing.assert_allclose(result, [0, 0, 0], atol=1e-6)

    def test_90deg_z(self) -> None:
        angle = np.pi / 2
        quat = np.array([0, 0, np.sin(angle / 2), np.cos(angle / 2)], dtype=np.float32)
        result = _quat2axisangle(quat)
        expected = np.array([0, 0, angle], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-5)
