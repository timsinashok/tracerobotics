"""Tests for metrics collection and aggregation."""

import numpy as np

from trace.metrics.aggregator import SweepAggregator, SweepResult
from trace.metrics.collectors import StepMetrics, EpisodeMetrics
from trace.task_spec.base import EpisodeResult


class TestEpisodeMetrics:
    def test_from_step_metrics(self):
        steps = [
            StepMetrics(reward=1.0, success=False),
            StepMetrics(reward=0.5, success=False),
            StepMetrics(reward=2.0, success=True),
        ]
        metrics = EpisodeMetrics.from_step_metrics(steps, max_steps=100)

        assert metrics.success is True
        assert metrics.total_reward == 3.5
        assert metrics.total_steps == 3
        assert metrics.time_to_success == 3

    def test_no_success(self):
        steps = [StepMetrics(reward=0.1, success=False) for _ in range(10)]
        metrics = EpisodeMetrics.from_step_metrics(steps, max_steps=100)

        assert metrics.success is False
        assert metrics.time_to_success is None


class TestSweepAggregator:
    def test_aggregate_basic(self):
        results = {
            0.0: [
                EpisodeResult(success=True, total_steps=100, total_reward=10.0,
                              time_to_success=50, catastrophic_failure=False),
                EpisodeResult(success=True, total_steps=80, total_reward=12.0,
                              time_to_success=40, catastrophic_failure=False),
            ],
            1.0: [
                EpisodeResult(success=False, total_steps=500, total_reward=1.0,
                              time_to_success=None, catastrophic_failure=True),
                EpisodeResult(success=False, total_steps=500, total_reward=0.5,
                              time_to_success=None, catastrophic_failure=False),
            ],
        }

        sweep = SweepAggregator.aggregate("TestStressor", results)

        assert sweep.stressor_name == "TestStressor"
        assert len(sweep.intensity_stats) == 2

        # Baseline
        baseline = sweep.intensity_stats[0]
        assert baseline.intensity == 0.0
        assert baseline.success_rate == 1.0

        # Stressed
        stressed = sweep.intensity_stats[1]
        assert stressed.intensity == 1.0
        assert stressed.success_rate == 0.0
        assert stressed.catastrophic_failure_rate == 0.5

    def test_breakpoint_detection(self):
        results = {
            0.0: [EpisodeResult(True, 100, 10.0, 50, False)] * 10,
            0.5: [EpisodeResult(True, 100, 8.0, 60, False)] * 6
                 + [EpisodeResult(False, 500, 1.0, None, False)] * 4,
            1.0: [EpisodeResult(False, 500, 1.0, None, False)] * 10,
        }

        sweep = SweepAggregator.aggregate("Test", results)
        assert sweep.breakpoint_intensity == 1.0  # first where < 50%

    def test_max_degradation(self):
        results = {
            0.0: [EpisodeResult(True, 100, 10.0, 50, False)] * 10,
            1.0: [EpisodeResult(False, 500, 1.0, None, False)] * 10,
        }
        sweep = SweepAggregator.aggregate("Test", results)
        assert sweep.max_degradation() == 1.0
        assert sweep.baseline_success_rate() == 1.0
