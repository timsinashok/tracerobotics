"""Sweep runner — runs parameter sweeps across stressors, intensities, and seeds."""

import logging
from dataclasses import dataclass, field
from typing import Any

from trace.metrics.aggregator import SweepAggregator, SweepResult
from trace.policy_adapter.base import BasePolicy
from trace.runner.episode_runner import EpisodeRunner
from trace.stressor_engine.base import BaseStressor, StressorConfig
from trace.task_spec.base import BaseTask, EpisodeResult

logger = logging.getLogger(__name__)


@dataclass
class SweepConfig:
    """Configuration for a parameter sweep."""

    stressor_type: type[BaseStressor]
    stressor_params: dict[str, Any] = field(default_factory=dict)
    intensities: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    num_episodes_per_config: int = 10


class SweepRunner:
    """Runs a full parameter sweep and aggregates results."""

    def __init__(self, task: BaseTask, policy: BasePolicy) -> None:
        self.task = task
        self.policy = policy

    def run_sweep(self, sweep_config: SweepConfig) -> SweepResult:
        """Execute the full sweep: iterate over intensities x seeds x episodes."""
        all_results: dict[float, list[EpisodeResult]] = {}

        for intensity in sweep_config.intensities:
            intensity_results: list[EpisodeResult] = []

            for seed in sweep_config.seeds:
                stressor_config = StressorConfig(
                    name=sweep_config.stressor_type.__name__,
                    intensity=intensity,
                    params=sweep_config.stressor_params,
                    seed=seed,
                )
                stressor = sweep_config.stressor_type(stressor_config)

                runner = EpisodeRunner(
                    task=self.task,
                    policy=self.policy,
                    stressors=[stressor],
                )

                for ep in range(sweep_config.num_episodes_per_config):
                    episode_seed = seed * 1000 + ep
                    result = runner.run(seed=episode_seed)
                    intensity_results.append(result)

                    logger.info(
                        "intensity=%.2f seed=%d ep=%d success=%s steps=%d",
                        intensity, seed, ep, result.success, result.total_steps,
                    )

            all_results[intensity] = intensity_results

        return SweepAggregator.aggregate(
            stressor_name=sweep_config.stressor_type.__name__,
            results_by_intensity=all_results,
        )
