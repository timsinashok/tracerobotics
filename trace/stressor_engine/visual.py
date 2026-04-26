"""Visual stressors — perturbations applied to camera images.

Four stressors that degrade visual observations to test policy robustness
to sensor noise, occlusions, lighting changes, and resolution loss.
All operate on uint8 image arrays via perturb_observation().
"""

from typing import Any

import numpy as np

from trace.stressor_engine.base import StressorConfig, SustainedVisualStressor
from trace.task_spec.base import Observation

# Keys that are expected to contain image data (uint8, HxWx3)
_IMAGE_KEYS = {"image", "wrist_image"}


def _is_image(value: np.ndarray) -> bool:
    """Check if an array looks like an image (3D uint8)."""
    return value.ndim == 3 and value.dtype == np.uint8


class ImageNoiseStressor(SustainedVisualStressor):
    """Adds Gaussian noise to camera images.

    Intensity controls the noise standard deviation:
        0.0 → no noise
        1.0 → std = max_noise_std (default 50)
    """

    def __init__(self, config: StressorConfig) -> None:
        super().__init__(config)
        self._max_noise_std: float = config.params.get("max_noise_std", 50.0)

    def _corrupt_observation(self, observation: Observation) -> Observation:
        std = self.intensity * self._max_noise_std
        result = {}
        for key, value in observation.items():
            if key in _IMAGE_KEYS and _is_image(value):
                noise = self._rng.normal(0.0, std, size=value.shape)
                noisy = np.clip(value.astype(np.float32) + noise, 0, 255)
                result[key] = noisy.astype(np.uint8)
            else:
                result[key] = value
        return result


class OcclusionStressor(SustainedVisualStressor):
    """Overlays random rectangles on camera images.

    Intensity controls the number and size of occluding patches:
        0.0 → no occlusion
        1.0 → max_patches (default 5) rectangles, each up to max_patch_frac of image
    """

    def __init__(self, config: StressorConfig) -> None:
        super().__init__(config)
        self._max_patches: int = config.params.get("max_patches", 5)
        self._max_patch_frac: float = config.params.get("max_patch_frac", 0.3)
        self._fill_value: int = config.params.get("fill_value", 0)

    def _corrupt_observation(self, observation: Observation) -> Observation:
        num_patches = max(1, int(self.intensity * self._max_patches))
        patch_frac = self.intensity * self._max_patch_frac

        result = {}
        for key, value in observation.items():
            if key in _IMAGE_KEYS and _is_image(value):
                result[key] = self._apply_occlusion(
                    value, num_patches, patch_frac
                )
            else:
                result[key] = value
        return result

    def _apply_occlusion(
        self, image: np.ndarray, num_patches: int, patch_frac: float
    ) -> np.ndarray:
        h, w = image.shape[:2]
        occluded = image.copy()
        min_frac = min(0.05, patch_frac)
        for _ in range(num_patches):
            ph = max(1, int(self._rng.uniform(min_frac, patch_frac) * h))
            pw = max(1, int(self._rng.uniform(min_frac, patch_frac) * w))
            y = self._rng.integers(0, max(1, h - ph))
            x = self._rng.integers(0, max(1, w - pw))
            occluded[y : y + ph, x : x + pw] = self._fill_value
        return occluded

class BrightnessShiftStressor(SustainedVisualStressor):
    """Shifts pixel brightness (simulates exposure changes).

    Intensity controls the maximum brightness shift:
        0.0 → no shift
        1.0 → shift up to max_shift (default 80) pixels, randomly + or -
    """

    def __init__(self, config: StressorConfig) -> None:
        super().__init__(config)
        self._max_shift: float = config.params.get("max_shift", 80.0)
        self._shift: float = 0.0

    def on_episode_start(self, task: Any) -> None:
        super().on_episode_start(task)
        # Pick a random shift direction that stays fixed for the episode
        self._shift = float(
            self._rng.uniform(-1.0, 1.0) * self.intensity * self._max_shift
        )

    def _corrupt_observation(self, observation: Observation) -> Observation:
        result = {}
        for key, value in observation.items():
            if key in _IMAGE_KEYS and _is_image(value):
                shifted = np.clip(
                    value.astype(np.float32) + self._shift, 0, 255
                )
                result[key] = shifted.astype(np.uint8)
            else:
                result[key] = value
        return result


class ResolutionStressor(SustainedVisualStressor):
    """Downscale then upscale images (pixelation / resolution loss).

    Intensity controls the downscale factor:
        0.0 → no pixelation (original resolution)
        1.0 → downscale to (1/max_downscale_factor) of original, then upscale back
    """

    def __init__(self, config: StressorConfig) -> None:
        super().__init__(config)
        self._max_downscale_factor: int = config.params.get(
            "max_downscale_factor", 8
        )

    def _corrupt_observation(self, observation: Observation) -> Observation:
        # Downscale factor: 1 (no change) to max_downscale_factor
        factor = max(
            1, int(1 + self.intensity * (self._max_downscale_factor - 1))
        )
        if factor <= 1:
            return observation

        result = {}
        for key, value in observation.items():
            if key in _IMAGE_KEYS and _is_image(value):
                result[key] = self._pixelate(value, factor)
            else:
                result[key] = value
        return result

    def _pixelate(self, image: np.ndarray, factor: int) -> np.ndarray:
        """Downscale then upscale using block averaging (no OpenCV needed)."""
        h, w, c = image.shape

        # Crop to be divisible by factor
        new_h = (h // factor) * factor
        new_w = (w // factor) * factor
        cropped = image[:new_h, :new_w]

        # Block average (downscale)
        small = cropped.reshape(
            new_h // factor, factor, new_w // factor, factor, c
        ).mean(axis=(1, 3)).astype(np.uint8)

        # Nearest-neighbor upscale back to original size
        upscaled = np.repeat(np.repeat(small, factor, axis=0), factor, axis=1)

        # Pad back to original size if needed
        if upscaled.shape[0] < h or upscaled.shape[1] < w:
            padded = np.zeros_like(image)
            padded[: upscaled.shape[0], : upscaled.shape[1]] = upscaled
            return padded

        return upscaled
