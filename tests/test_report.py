"""Tests for report generation."""

import os
import tempfile

from trace.metrics.aggregator import IntensityStats, SweepResult
from trace.policy_adapter.base import PolicyMetadata
from trace.report.generator import ReportGenerator


class TestReportGenerator:
    def test_generates_markdown_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator(output_dir=tmpdir)

            meta = PolicyMetadata(name="TestPolicy", modalities=["vision", "proprioception"])

            sweep = SweepResult(
                stressor_name="LatencyStressor",
                intensity_stats=[
                    IntensityStats(
                        intensity=0.0, num_episodes=10, success_rate=0.9,
                        mean_time_to_success=50.0, catastrophic_failure_rate=0.0,
                        mean_reward=10.0, reward_std=1.0, mean_steps=100.0,
                    ),
                    IntensityStats(
                        intensity=1.0, num_episodes=10, success_rate=0.1,
                        mean_time_to_success=None, catastrophic_failure_rate=0.3,
                        mean_reward=2.0, reward_std=0.5, mean_steps=500.0,
                    ),
                ],
                breakpoint_intensity=1.0,
            )

            filepath = gen.generate(meta, "pick_and_place", [sweep])

            assert os.path.exists(filepath)
            assert filepath.endswith(".md")

            content = open(filepath).read()
            assert "Trace Robotics" in content
            assert "TestPolicy" in content
            assert "LatencyStressor" in content
            assert "Breakpoints" in content
