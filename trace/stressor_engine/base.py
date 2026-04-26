"""Abstract base class for stressors.

A stressor is a parameterized perturbation applied during episode execution.
Stressors can modify:
- Actions (e.g., latency, noise)
- Observations (e.g., dropout, corruption)
- Physics (e.g., friction, mass)
- Embodiment (e.g., joint limits, link lengths)

Every stressor must be:
- Parameterized (controlled by a scalar or small config)
- Sweepable (can iterate over a range of intensities)
- Reproducible (deterministic given a seed)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from trace.task_spec.base import Observation


@dataclass
class StressorConfig:
    """Configuration for a stressor."""

    name: str
    intensity: float = 0.0  # 0.0 = no stress, 1.0 = maximum stress
    params: dict[str, Any] = field(default_factory=dict)
    seed: int = 0


class BaseStressor(ABC):
    """Abstract stressor that perturbs some aspect of the evaluation loop."""

    def __init__(self, config: StressorConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def intensity(self) -> float:
        return self.config.intensity

    @abstractmethod
    def on_episode_start(self, task: Any) -> None:
        """Called at the start of each episode. Use to modify env if needed."""

    @abstractmethod
    def perturb_observation(self, observation: Observation) -> Observation:
        """Modify the observation before it reaches the policy."""

    @abstractmethod
    def perturb_action(self, action: np.ndarray) -> np.ndarray:
        """Modify the action before it reaches the simulator."""

    def on_episode_end(self) -> None:
        """Called at the end of each episode. Use to restore env state."""

    def describe(self) -> dict[str, Any]:
        """Return a serializable description of this stressor's configuration."""
        return {
            "name": self.config.name,
            "intensity": self.config.intensity,
            "params": self.config.params,
        }


class SustainedVisualStressor(BaseStressor):
    """Base for visual stressors that need corruption to survive action chunking.

    Policies using action chunking only query observations every N steps.
    A single-frame corruption has a high chance of being ignored. This base
    class caches the corrupted observation for ``persist_steps`` steps so
    the policy is guaranteed to see it on its next query.

    Subclasses implement ``_corrupt_observation`` instead of ``perturb_observation``.
    """

    def __init__(self, config: StressorConfig) -> None:
        super().__init__(config)
        self._persist_steps: int = config.params.get("persist_steps", 10)
        self._cached_obs: Observation | None = None
        self._remaining_persist: int = 0

    def on_episode_start(self, task: Any) -> None:
        self._cached_obs = None
        self._remaining_persist = 0

    def perturb_observation(self, observation: Observation) -> Observation:
        if self.intensity == 0.0:
            return observation

        # If we still have a cached corruption, replay it
        if self._remaining_persist > 0 and self._cached_obs is not None:
            self._remaining_persist -= 1
            # Merge: use cached values for corrupted keys, fresh for others
            result = dict(observation)
            result.update(self._cached_obs)
            return result

        # Generate fresh corruption
        corrupted = self._corrupt_observation(observation)
        if corrupted is not observation:
            self._cached_obs = {
                k: v for k, v in corrupted.items()
                if k in observation and not np.array_equal(v, observation[k])
            }
            self._remaining_persist = self._persist_steps - 1
        return corrupted

    @abstractmethod
    def _corrupt_observation(self, observation: Observation) -> Observation:
        """Apply the visual corruption. Called once, then cached."""

    def perturb_action(self, action: np.ndarray) -> np.ndarray:
        return action
