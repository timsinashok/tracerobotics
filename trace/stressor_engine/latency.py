"""Action latency stressor.

Simulates communication delay between the policy and the robot actuators.
Delays action application by buffering actions and replaying stale ones.

Intensity maps to delay in simulation steps:
    0.0 -> 0 steps delay
    1.0 -> max_delay_steps delay (default 10 steps, ~200ms at 50Hz)
"""

from collections import deque
from typing import Any

import numpy as np

from trace.stressor_engine.base import BaseStressor, StressorConfig
from trace.task_spec.base import Observation


class LatencyStressor(BaseStressor):
    """Delays action execution by buffering actions."""

    def __init__(self, config: StressorConfig) -> None:
        super().__init__(config)
        self._max_delay_steps: int = config.params.get("max_delay_steps", 10)
        self._action_buffer: deque[np.ndarray] = deque()
        self._delay_steps: int = 0

    @property
    def delay_steps(self) -> int:
        return self._delay_steps

    def on_episode_start(self, task: Any) -> None:
        self._delay_steps = int(self.intensity * self._max_delay_steps)
        self._action_buffer.clear()

    def perturb_observation(self, observation: Observation) -> Observation:
        return observation  # Latency only affects actions

    def perturb_action(self, action: np.ndarray) -> np.ndarray:
        if self._delay_steps == 0:
            return action

        self._action_buffer.append(action.copy())

        if len(self._action_buffer) > self._delay_steps:
            return self._action_buffer.popleft()

        # Not enough buffered yet — repeat current action to hold position
        # (zeros are unsafe: e.g. gripper 0 is undefined between open/close)
        return action.copy()
