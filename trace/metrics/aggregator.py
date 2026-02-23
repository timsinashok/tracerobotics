"""Cross-episode and cross-seed metric aggregation."""

from dataclasses import dataclass, field

import numpy as np

from trace.task_spec.base import EpisodeResult


@dataclass
class IntensityStats:
    """Aggregated stats for one stressor intensity level."""

    intensity: float
    num_episodes: int
    success_rate: float
    mean_time_to_success: float | None
    catastrophic_failure_rate: float
    mean_reward: float
    reward_std: float
    mean_steps: float


@dataclass
class SweepResult:
    """Full result of a stressor sweep."""

    stressor_name: str
    intensity_stats: list[IntensityStats]
    breakpoint_intensity: float | None = None  # Intensity where success drops below 50%

    def baseline_success_rate(self) -> float:
        """Success rate at intensity 0.0 (no stress)."""
        for s in self.intensity_stats:
            if s.intensity == 0.0:
                return s.success_rate
        return 0.0

    def max_degradation(self) -> float:
        """Largest drop in success rate from baseline."""
        baseline = self.baseline_success_rate()
        if not self.intensity_stats:
            return 0.0
        worst = min(s.success_rate for s in self.intensity_stats)
        return baseline - worst


class SweepAggregator:
    """Aggregates episode results across intensities into SweepResult."""

    @staticmethod
    def aggregate(
        stressor_name: str,
        results_by_intensity: dict[float, list[EpisodeResult]],
    ) -> SweepResult:
        stats_list: list[IntensityStats] = []

        for intensity in sorted(results_by_intensity.keys()):
            episodes = results_by_intensity[intensity]
            n = len(episodes)
            if n == 0:
                continue

            successes = [e for e in episodes if e.success]
            catastrophic = [e for e in episodes if e.catastrophic_failure]
            rewards = [e.total_reward for e in episodes]
            steps = [e.total_steps for e in episodes]
            times = [e.time_to_success for e in successes if e.time_to_success is not None]

            stats_list.append(
                IntensityStats(
                    intensity=intensity,
                    num_episodes=n,
                    success_rate=len(successes) / n,
                    mean_time_to_success=float(np.mean(times)) if times else None,
                    catastrophic_failure_rate=len(catastrophic) / n,
                    mean_reward=float(np.mean(rewards)),
                    reward_std=float(np.std(rewards)),
                    mean_steps=float(np.mean(steps)),
                )
            )

        # Find breakpoint: first intensity where success < 50%
        breakpoint = None
        for s in stats_list:
            if s.success_rate < 0.5:
                breakpoint = s.intensity
                break

        return SweepResult(
            stressor_name=stressor_name,
            intensity_stats=stats_list,
            breakpoint_intensity=breakpoint,
        )
