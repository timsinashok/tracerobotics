"""Abstract base class for task specifications.

A task defines:
- A MuJoCo environment (model XML + initial state)
- A success condition
- Episode length and reset behavior
- Observation extraction from the simulator
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Observation type: dict mapping sensor names to numpy arrays.
# Values may be float32 (proprioception) or uint8 (images).
Observation = dict[str, np.ndarray]


@dataclass
class TaskConfig:
    """Configuration for a task."""

    name: str
    max_episode_steps: int = 500
    success_threshold: float = 0.95
    seed: int = 0
    task_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    """Result of a single episode."""

    success: bool
    total_steps: int
    total_reward: float
    time_to_success: int | None  # Step at which success was first achieved
    catastrophic_failure: bool
    step_metrics: dict[str, list[float]] = field(default_factory=dict)


class BaseTask(ABC):
    """Abstract task specification."""

    def __init__(self, config: TaskConfig) -> None:
        self.config = config

    @abstractmethod
    def initialize(self) -> None:
        """Create the MuJoCo model and data. Called once."""

    @abstractmethod
    def reset(self, seed: int | None = None) -> Observation:
        """Reset the environment and return the initial observation."""

    @abstractmethod
    def step(
        self, action: np.ndarray
    ) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Execute one action and return (obs, reward, done, info)."""

    @abstractmethod
    def check_success(self) -> bool:
        """Return True if the current state satisfies the success condition."""

    @abstractmethod
    def check_catastrophic_failure(self) -> bool:
        """Return True if the current state is an unrecoverable failure."""

    @abstractmethod
    def get_observation(self) -> Observation:
        """Extract the current observation from simulator state."""

    @property
    def language_instruction(self) -> str:
        """Natural language instruction for this task (used by VLA policies)."""
        return self.config.task_params.get("language_instruction", "")

    def get_mujoco_model(self) -> Any:
        """Return the underlying MuJoCo model (for stressors that modify physics)."""
        raise NotImplementedError("Subclass must expose the MuJoCo model for physics stressors.")

    def get_mujoco_data(self) -> Any:
        """Return the underlying MuJoCo data (for stressors that modify state)."""
        raise NotImplementedError("Subclass must expose the MuJoCo data for state stressors.")

    def post_stressor_settle(self) -> Observation:
        """Re-settle the environment after stressors modify physics.

        Called by EpisodeRunner after on_episode_start() to allow objects
        to reach equilibrium under modified physics. Returns a fresh observation.
        Default: no-op, returns current observation.
        """
        return self.get_observation()
