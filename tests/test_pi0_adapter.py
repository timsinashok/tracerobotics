"""Tests for the Pi0 policy adapter."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from trace.policy_adapter.pi0_adapter import Pi0PolicyAdapter


class MockWebsocketClient:
    """Mock openpi WebSocket client for testing without a server."""

    def __init__(self, action_dim: int = 7, chunk_size: int = 5) -> None:
        self._action_dim = action_dim
        self._chunk_size = chunk_size
        self._rng = np.random.default_rng(42)
        self.call_count = 0

    def infer(self, observation: dict) -> np.ndarray:
        self.call_count += 1
        # Return Cartesian-like actions: first 3 dims are small deltas
        actions = self._rng.uniform(-0.01, 0.01, size=(self._chunk_size, self._action_dim))
        return actions.astype(np.float32)


@pytest.fixture
def pi0_adapter() -> Pi0PolicyAdapter:
    adapter = Pi0PolicyAdapter(chunk_size=5, action_mode="joint_position")
    return adapter


@pytest.fixture
def pi0_adapter_with_mock() -> Pi0PolicyAdapter:
    adapter = Pi0PolicyAdapter(chunk_size=5, action_mode="joint_position")
    adapter._client = MockWebsocketClient(action_dim=7, chunk_size=5)
    return adapter


@pytest.fixture
def pi0_observation() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "image": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "wrist_image": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "joint_pos": rng.random(7, dtype=np.float32),
        "ee_pos": rng.random(3, dtype=np.float32),
        "target_pos": rng.random(3, dtype=np.float32),
    }


class TestPi0PolicyAdapter:
    def test_act_returns_correct_shape(self, pi0_adapter_with_mock, pi0_observation):
        action = pi0_adapter_with_mock.act(pi0_observation)
        assert action.shape == (7,)
        assert action.dtype == np.float32

    def test_act_bounded(self, pi0_adapter_with_mock, pi0_observation):
        for _ in range(20):
            action = pi0_adapter_with_mock.act(pi0_observation)
            assert np.all(action >= -1.0)
            assert np.all(action <= 1.0)

    def test_action_chunking(self, pi0_adapter_with_mock, pi0_observation):
        """One server call should produce chunk_size actions."""
        mock_client = pi0_adapter_with_mock._client
        assert mock_client.call_count == 0

        # First 5 calls should use one server inference
        for _ in range(5):
            pi0_adapter_with_mock.act(pi0_observation)
        assert mock_client.call_count == 1

        # 6th call should trigger a new server inference
        pi0_adapter_with_mock.act(pi0_observation)
        assert mock_client.call_count == 2

    def test_reset_clears_buffer(self, pi0_adapter_with_mock, pi0_observation):
        """Reset should clear the action buffer, forcing a new server call."""
        mock_client = pi0_adapter_with_mock._client

        # Fill buffer
        pi0_adapter_with_mock.act(pi0_observation)
        assert mock_client.call_count == 1

        # Reset and act again
        pi0_adapter_with_mock.reset()
        assert len(pi0_adapter_with_mock._action_buffer) == 0

        pi0_adapter_with_mock.act(pi0_observation)
        assert mock_client.call_count == 2

    def test_metadata(self, pi0_adapter):
        meta = pi0_adapter.metadata()
        assert meta.name == "Pi0PolicyAdapter"
        assert "image" in meta.observation_space
        assert meta.action_space["dim"] == 7
        assert "vision" in meta.modalities
        assert "proprioception" in meta.modalities

    def test_observation_mapping(self, pi0_adapter, pi0_observation):
        """Verify observation keys are mapped to openpi LIBERO format."""
        openpi_obs = pi0_adapter._build_observation(pi0_observation)
        assert "observation/image" in openpi_obs
        assert "observation/wrist_image" in openpi_obs
        assert "observation/state" in openpi_obs
        assert "prompt" in openpi_obs

        # Images should be rotated (not identical to input)
        assert openpi_obs["observation/image"].shape == (224, 224, 3)
        assert openpi_obs["observation/wrist_image"].shape == (224, 224, 3)

    def test_observation_image_rotation(self, pi0_adapter, pi0_observation):
        """Images should be rotated 180 degrees for LIBERO convention."""
        openpi_obs = pi0_adapter._build_observation(pi0_observation)
        original = pi0_observation["image"]
        mapped = openpi_obs["observation/image"]
        # 180-degree rotation: pixel at (0,0) goes to (-1,-1)
        np.testing.assert_array_equal(mapped[0, 0], original[-1, -1])

    def test_state_vector_shape(self, pi0_adapter, pi0_observation):
        """State vector should be 8-dimensional."""
        openpi_obs = pi0_adapter._build_observation(pi0_observation)
        state = openpi_obs["observation/state"]
        assert state.shape == (8,)
        assert state.dtype == np.float32

    def test_state_vector_contents(self, pi0_adapter, pi0_observation):
        """State vector should contain ee_pos(3) + ee_orientation(4) + gripper(1)."""
        openpi_obs = pi0_adapter._build_observation(pi0_observation)
        state = openpi_obs["observation/state"]
        # ee_pos should be in first 3 elements
        np.testing.assert_array_almost_equal(
            state[:3], pi0_observation["ee_pos"]
        )

    def test_missing_images_use_zeros(self, pi0_adapter):
        """If no images in observation, should use zero arrays."""
        obs = {"joint_pos": np.zeros(7, dtype=np.float32)}
        openpi_obs = pi0_adapter._build_observation(obs)
        assert np.all(openpi_obs["observation/image"] == 0)
        assert np.all(openpi_obs["observation/wrist_image"] == 0)

    def test_set_task_info(self, pi0_adapter):
        pi0_adapter.set_task_info("reach the target")
        obs = {"joint_pos": np.zeros(7, dtype=np.float32)}
        openpi_obs = pi0_adapter._build_observation(obs)
        assert openpi_obs["prompt"] == "reach the target"

    def test_no_server_returns_zeros(self, pi0_adapter, pi0_observation):
        """Without a server, act() should return zeros."""
        pi0_adapter._client = None
        action = pi0_adapter.act(pi0_observation)
        assert action.shape == (7,)
        np.testing.assert_array_equal(action, np.zeros(7))

    def test_load_without_openpi_client(self, pi0_adapter):
        """load() should not raise even if openpi_client is not installed."""
        pi0_adapter.load("")
        # Client should be None if import failed
        assert pi0_adapter._client is None

    def test_joint_position_action_mode(self, pi0_observation):
        """Joint position mode should directly clip raw actions."""
        adapter = Pi0PolicyAdapter(action_mode="joint_position")
        adapter._client = MockWebsocketClient()
        action = adapter.act(pi0_observation)
        assert action.shape == (7,)
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)
