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
from numpy.typing import NDArray


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
    def perturb_observation(
        self, observation: dict[str, NDArray[np.floating]]
    ) -> dict[str, NDArray[np.floating]]:
        """Modify the observation before it reaches the policy."""

    @abstractmethod
    def perturb_action(
        self, action: NDArray[np.floating]
    ) -> NDArray[np.floating]:
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
