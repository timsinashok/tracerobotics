"""A random policy for testing and baseline comparisons."""

import numpy as np
from numpy.typing import NDArray

from trace.policy_adapter.base import BasePolicy, PolicyMetadata


class RandomPolicy(BasePolicy):
    """Outputs uniformly random actions. Useful as a baseline and for testing."""

    def __init__(self, action_dim: int, seed: int = 0) -> None:
        self._action_dim = action_dim
        self._rng = np.random.default_rng(seed)
        self._seed = seed

    def load(self, checkpoint_path: str) -> None:
        pass  # Nothing to load

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)

    def act(self, observation: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        return self._rng.uniform(-1.0, 1.0, size=self._action_dim).astype(np.float32)

    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            name="RandomPolicy",
            action_space={"dim": self._action_dim, "low": -1.0, "high": 1.0},
            modalities=[],
        )
