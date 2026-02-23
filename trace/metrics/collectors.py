"""Per-step and per-episode metric collection."""

from dataclasses import dataclass, field


@dataclass
class StepMetrics:
    """Metrics collected at each simulation step."""

    reward: float = 0.0
    success: bool = False
    distance_to_goal: float | None = None


@dataclass
class EpisodeMetrics:
    """Aggregated metrics for a single episode."""

    success: bool = False
    total_reward: float = 0.0
    total_steps: int = 0
    time_to_success: int | None = None
    catastrophic_failure: bool = False
    mean_reward_per_step: float = 0.0

    @classmethod
    def from_step_metrics(cls, steps: list[StepMetrics], max_steps: int) -> "EpisodeMetrics":
        """Compute episode-level metrics from a list of step metrics."""
        if not steps:
            return cls()

        total_reward = sum(s.reward for s in steps)
        successes = [i for i, s in enumerate(steps) if s.success]

        return cls(
            success=len(successes) > 0,
            total_reward=total_reward,
            total_steps=len(steps),
            time_to_success=successes[0] + 1 if successes else None,
            catastrophic_failure=False,
            mean_reward_per_step=total_reward / len(steps) if steps else 0.0,
        )
