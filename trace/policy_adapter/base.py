"""Abstract base class for policy adapters.

A policy adapter wraps any model checkpoint into a uniform interface:
    action = policy.act(observation)

All policies must implement reset() and act().
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class PolicyMetadata:
    """Metadata describing a loaded policy."""

    name: str
    observation_space: dict[str, Any] = field(default_factory=dict)
    action_space: dict[str, Any] = field(default_factory=dict)
    modalities: list[str] = field(default_factory=lambda: ["vision", "proprioception"])


class BasePolicy(ABC):
    """Abstract policy adapter.

    Wraps a model checkpoint so the runner can call act(obs) without
    knowing the underlying framework (PyTorch, JAX, etc.).
    """

    @abstractmethod
    def load(self, checkpoint_path: str) -> None:
        """Load model weights from a checkpoint file."""

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state (e.g. recurrent hidden state)."""

    @abstractmethod
    def act(self, observation: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Return an action given an observation dict.

        Args:
            observation: Dict with keys like "image", "proprioception", etc.
                Each value is a numpy array.

        Returns:
            Continuous action vector as a numpy array.
        """

    @abstractmethod
    def metadata(self) -> PolicyMetadata:
        """Return metadata about this policy."""
