"""Tests for visual stressors."""

import numpy as np

from trace.stressor_engine.base import StressorConfig
from trace.stressor_engine.visual import (
    BrightnessShiftStressor,
    ImageNoiseStressor,
    OcclusionStressor,
    ResolutionStressor,
)


def _make_image_observation(
    height: int = 64, width: int = 64, seed: int = 0
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "image": rng.integers(50, 200, size=(height, width, 3), dtype=np.uint8),
        "wrist_image": rng.integers(50, 200, size=(height, width, 3), dtype=np.uint8),
        "proprioception": rng.random(14, dtype=np.float32),
    }


class TestImageNoiseStressor:
    def test_zero_intensity_passthrough(self):
        config = StressorConfig(name="image_noise", intensity=0.0, seed=0)
        stressor = ImageNoiseStressor(config)
        stressor.on_episode_start(None)

        obs = _make_image_observation()
        result = stressor.perturb_observation(obs)
        np.testing.assert_array_equal(result["image"], obs["image"])
        np.testing.assert_array_equal(result["wrist_image"], obs["wrist_image"])

    def test_high_intensity_adds_noise(self):
        config = StressorConfig(
            name="image_noise", intensity=1.0, seed=42,
            params={"max_noise_std": 50.0},
        )
        stressor = ImageNoiseStressor(config)
        stressor.on_episode_start(None)

        obs = _make_image_observation()
        result = stressor.perturb_observation(obs)
        # Images should differ
        assert not np.array_equal(result["image"], obs["image"])
        # But still valid uint8
        assert result["image"].dtype == np.uint8
        assert result["image"].min() >= 0
        assert result["image"].max() <= 255

    def test_proprioception_passthrough(self):
        config = StressorConfig(name="image_noise", intensity=1.0, seed=0)
        stressor = ImageNoiseStressor(config)
        stressor.on_episode_start(None)

        obs = _make_image_observation()
        result = stressor.perturb_observation(obs)
        np.testing.assert_array_equal(result["proprioception"], obs["proprioception"])

    def test_action_passthrough(self):
        config = StressorConfig(name="image_noise", intensity=1.0, seed=0)
        stressor = ImageNoiseStressor(config)
        action = np.ones(7, dtype=np.float32)
        result = stressor.perturb_action(action)
        np.testing.assert_array_equal(result, action)

    def test_deterministic_with_seed(self):
        obs = _make_image_observation()
        results = []
        for _ in range(2):
            config = StressorConfig(name="image_noise", intensity=0.5, seed=123)
            stressor = ImageNoiseStressor(config)
            stressor.on_episode_start(None)
            results.append(stressor.perturb_observation(obs)["image"])
        np.testing.assert_array_equal(results[0], results[1])


class TestOcclusionStressor:
    def test_zero_intensity_passthrough(self):
        config = StressorConfig(name="occlusion", intensity=0.0, seed=0)
        stressor = OcclusionStressor(config)
        stressor.on_episode_start(None)

        obs = _make_image_observation()
        result = stressor.perturb_observation(obs)
        np.testing.assert_array_equal(result["image"], obs["image"])

    def test_low_intensity_no_crash(self):
        """Regression: intensity=0.1 with max_patch_frac=0.3 gives patch_frac=0.03 < 0.05."""
        config = StressorConfig(
            name="occlusion", intensity=0.1, seed=42,
            params={"max_patches": 5, "max_patch_frac": 0.3, "fill_value": 0},
        )
        stressor = OcclusionStressor(config)
        stressor.on_episode_start(None)

        obs = _make_image_observation()
        result = stressor.perturb_observation(obs)
        assert result["image"].shape == obs["image"].shape
        assert result["image"].dtype == np.uint8

    def test_high_intensity_occludes(self):
        config = StressorConfig(
            name="occlusion", intensity=1.0, seed=42,
            params={"max_patches": 5, "fill_value": 0},
        )
        stressor = OcclusionStressor(config)
        stressor.on_episode_start(None)

        obs = _make_image_observation()
        result = stressor.perturb_observation(obs)
        # Some pixels should now be zero (occluded)
        assert np.any(result["image"] == 0)
        # Original had pixels in [50, 200) so zeros indicate occlusion
        assert result["image"].dtype == np.uint8

    def test_proprioception_passthrough(self):
        config = StressorConfig(name="occlusion", intensity=1.0, seed=0)
        stressor = OcclusionStressor(config)
        stressor.on_episode_start(None)

        obs = _make_image_observation()
        result = stressor.perturb_observation(obs)
        np.testing.assert_array_equal(result["proprioception"], obs["proprioception"])

    def test_action_passthrough(self):
        config = StressorConfig(name="occlusion", intensity=1.0, seed=0)
        stressor = OcclusionStressor(config)
        action = np.ones(7, dtype=np.float32)
        np.testing.assert_array_equal(stressor.perturb_action(action), action)

    def test_deterministic_with_seed(self):
        obs = _make_image_observation()
        results = []
        for _ in range(2):
            config = StressorConfig(name="occlusion", intensity=0.5, seed=99)
            stressor = OcclusionStressor(config)
            stressor.on_episode_start(None)
            results.append(stressor.perturb_observation(obs)["image"])
        np.testing.assert_array_equal(results[0], results[1])


class TestBrightnessShiftStressor:
    def test_zero_intensity_passthrough(self):
        config = StressorConfig(name="brightness", intensity=0.0, seed=0)
        stressor = BrightnessShiftStressor(config)
        stressor.on_episode_start(None)

        obs = _make_image_observation()
        result = stressor.perturb_observation(obs)
        np.testing.assert_array_equal(result["image"], obs["image"])

    def test_high_intensity_shifts_brightness(self):
        config = StressorConfig(
            name="brightness", intensity=1.0, seed=42,
            params={"max_shift": 80.0},
        )
        stressor = BrightnessShiftStressor(config)
        stressor.on_episode_start(None)

        obs = _make_image_observation()
        result = stressor.perturb_observation(obs)
        assert not np.array_equal(result["image"], obs["image"])
        assert result["image"].dtype == np.uint8

    def test_shift_consistent_within_episode(self):
        """Brightness shift should be constant within an episode."""
        config = StressorConfig(name="brightness", intensity=0.5, seed=42)
        stressor = BrightnessShiftStressor(config)
        stressor.on_episode_start(None)

        obs = _make_image_observation()
        result1 = stressor.perturb_observation(obs)
        result2 = stressor.perturb_observation(obs)
        np.testing.assert_array_equal(result1["image"], result2["image"])

    def test_proprioception_passthrough(self):
        config = StressorConfig(name="brightness", intensity=1.0, seed=0)
        stressor = BrightnessShiftStressor(config)
        stressor.on_episode_start(None)

        obs = _make_image_observation()
        result = stressor.perturb_observation(obs)
        np.testing.assert_array_equal(result["proprioception"], obs["proprioception"])

    def test_action_passthrough(self):
        config = StressorConfig(name="brightness", intensity=1.0, seed=0)
        stressor = BrightnessShiftStressor(config)
        action = np.ones(7, dtype=np.float32)
        np.testing.assert_array_equal(stressor.perturb_action(action), action)

    def test_clamps_to_valid_range(self):
        """Shifted pixels should stay in [0, 255]."""
        config = StressorConfig(
            name="brightness", intensity=1.0, seed=0,
            params={"max_shift": 200.0},
        )
        stressor = BrightnessShiftStressor(config)
        stressor.on_episode_start(None)

        obs = _make_image_observation()
        result = stressor.perturb_observation(obs)
        assert result["image"].min() >= 0
        assert result["image"].max() <= 255


class TestResolutionStressor:
    def test_zero_intensity_passthrough(self):
        config = StressorConfig(name="resolution", intensity=0.0, seed=0)
        stressor = ResolutionStressor(config)
        stressor.on_episode_start(None)

        obs = _make_image_observation()
        result = stressor.perturb_observation(obs)
        np.testing.assert_array_equal(result["image"], obs["image"])

    def test_high_intensity_pixelates(self):
        config = StressorConfig(
            name="resolution", intensity=1.0, seed=0,
            params={"max_downscale_factor": 8},
        )
        stressor = ResolutionStressor(config)
        stressor.on_episode_start(None)

        obs = _make_image_observation()
        result = stressor.perturb_observation(obs)
        # Image should be pixelated (blocks of same color)
        img = result["image"]
        assert img.shape == obs["image"].shape
        # Check that adjacent pixels within a block are identical (block size = 8)
        np.testing.assert_array_equal(img[0, 0], img[0, 1])
        np.testing.assert_array_equal(img[0, 0], img[1, 0])

    def test_preserves_shape(self):
        config = StressorConfig(
            name="resolution", intensity=0.5, seed=0,
            params={"max_downscale_factor": 4},
        )
        stressor = ResolutionStressor(config)
        stressor.on_episode_start(None)

        obs = _make_image_observation(height=64, width=64)
        result = stressor.perturb_observation(obs)
        assert result["image"].shape == obs["image"].shape

    def test_proprioception_passthrough(self):
        config = StressorConfig(name="resolution", intensity=1.0, seed=0)
        stressor = ResolutionStressor(config)
        stressor.on_episode_start(None)

        obs = _make_image_observation()
        result = stressor.perturb_observation(obs)
        np.testing.assert_array_equal(result["proprioception"], obs["proprioception"])

    def test_action_passthrough(self):
        config = StressorConfig(name="resolution", intensity=1.0, seed=0)
        stressor = ResolutionStressor(config)
        action = np.ones(7, dtype=np.float32)
        np.testing.assert_array_equal(stressor.perturb_action(action), action)

    def test_deterministic_with_seed(self):
        obs = _make_image_observation()
        results = []
        for _ in range(2):
            config = StressorConfig(name="resolution", intensity=0.5, seed=77)
            stressor = ResolutionStressor(config)
            stressor.on_episode_start(None)
            results.append(stressor.perturb_observation(obs)["image"])
        np.testing.assert_array_equal(results[0], results[1])
