"""Episode runner — executes a single episode with stressors applied."""

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trace.metrics.collectors import StepMetrics, EpisodeMetrics
from trace.policy_adapter.base import BasePolicy
from trace.stressor_engine.base import BaseStressor
from trace.task_spec.base import BaseTask, EpisodeResult

logger = logging.getLogger(__name__)


class EpisodeRunner:
    """Runs one episode: reset -> step loop -> collect results."""

    def __init__(
        self,
        task: BaseTask,
        policy: BasePolicy,
        stressors: list[BaseStressor] | None = None,
    ) -> None:
        self.task = task
        self.policy = policy
        self.stressors = stressors or []

    def run(self, seed: int | None = None) -> EpisodeResult:
        """Execute a single episode and return the result."""
        # Reset policy and environment
        self.policy.reset()
        obs = self.task.reset(seed=seed)

        # Notify stressors of episode start
        for stressor in self.stressors:
            stressor.on_episode_start(self.task)

        total_reward = 0.0
        time_to_success: int | None = None
        step_rewards: list[float] = []
        step_successes: list[float] = []

        for step in range(self.task.config.max_episode_steps):
            # Apply observation stressors
            stressed_obs = obs
            for stressor in self.stressors:
                stressed_obs = stressor.perturb_observation(stressed_obs)

            # Get action from policy
            action = self.policy.act(stressed_obs)

            # Apply action stressors
            stressed_action = action
            for stressor in self.stressors:
                stressed_action = stressor.perturb_action(stressed_action)

            # Step the environment
            obs, reward, done, info = self.task.step(stressed_action)

            total_reward += reward
            step_rewards.append(reward)

            # Check success
            success = self.task.check_success()
            step_successes.append(float(success))
            if success and time_to_success is None:
                time_to_success = step + 1

            # Check catastrophic failure
            if self.task.check_catastrophic_failure():
                for stressor in self.stressors:
                    stressor.on_episode_end()
                return EpisodeResult(
                    success=False,
                    total_steps=step + 1,
                    total_reward=total_reward,
                    time_to_success=None,
                    catastrophic_failure=True,
                    step_metrics={"reward": step_rewards, "success": step_successes},
                )

            if done:
                break

        # Notify stressors of episode end
        for stressor in self.stressors:
            stressor.on_episode_end()

        return EpisodeResult(
            success=time_to_success is not None,
            total_steps=step + 1,
            total_reward=total_reward,
            time_to_success=time_to_success,
            catastrophic_failure=False,
            step_metrics={"reward": step_rewards, "success": step_successes},
        )
