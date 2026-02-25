"""End-to-end integration tests for the Trace Robotics pipeline."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from trace.config_loader import create_task, load_sweep_configs, load_task_config
from trace.metrics.aggregator import SweepResult
from trace.policy_adapter.random_policy import RandomPolicy
from trace.policy_adapter.scripted_reach import ScriptedReachPolicy
from trace.report.generator import ReportGenerator
from trace.runner.episode_runner import EpisodeRunner
from trace.runner.sweep_runner import SweepConfig, SweepRunner
from trace.stressor_engine.base import StressorConfig
from trace.stressor_engine.dropout import DropoutStressor
from trace.stressor_engine.latency import LatencyStressor
from trace.stressor_engine.physics_shift import PhysicsShiftStressor
from trace.task_spec.base import TaskConfig
from trace.task_spec.reach import ReachTask


@pytest.fixture
def reach_task() -> ReachTask:
    config = TaskConfig(name="reach", max_episode_steps=50, task_params={"success_radius": 0.05})
    task = ReachTask(config)
    task.initialize()
    return task


class TestEpisodeRunnerIntegration:
    def test_episode_with_random_policy(self, reach_task: ReachTask) -> None:
        policy = RandomPolicy(action_dim=7, seed=42)
        runner = EpisodeRunner(task=reach_task, policy=policy)
        result = runner.run(seed=0)
        assert result.total_steps > 0
        assert result.total_steps <= 50
        assert isinstance(result.total_reward, float)
        assert "reward" in result.step_metrics

    def test_episode_with_scripted_policy(self, reach_task: ReachTask) -> None:
        policy = ScriptedReachPolicy()
        policy.set_env(reach_task.get_mujoco_model(), reach_task.get_mujoco_data())
        runner = EpisodeRunner(task=reach_task, policy=policy)
        result = runner.run(seed=0)
        assert result.total_steps > 0
        assert isinstance(result.success, bool)

    def test_episode_with_latency_stressor(self, reach_task: ReachTask) -> None:
        policy = RandomPolicy(action_dim=7, seed=42)
        stressor = LatencyStressor(StressorConfig(name="latency", intensity=0.5, seed=0))
        runner = EpisodeRunner(task=reach_task, policy=policy, stressors=[stressor])
        result = runner.run(seed=0)
        assert result.total_steps > 0

    def test_episode_with_dropout_stressor(self, reach_task: ReachTask) -> None:
        policy = RandomPolicy(action_dim=7, seed=42)
        stressor = DropoutStressor(
            StressorConfig(name="dropout", intensity=0.5, seed=0, params={"mode": "zero"})
        )
        runner = EpisodeRunner(task=reach_task, policy=policy, stressors=[stressor])
        result = runner.run(seed=0)
        assert result.total_steps > 0

    def test_episode_with_physics_shift_stressor(self, reach_task: ReachTask) -> None:
        policy = RandomPolicy(action_dim=7, seed=42)
        stressor = PhysicsShiftStressor(
            StressorConfig(name="physics_shift", intensity=0.5, seed=0)
        )
        runner = EpisodeRunner(task=reach_task, policy=policy, stressors=[stressor])
        result = runner.run(seed=0)
        assert result.total_steps > 0

    def test_episode_deterministic_with_same_seed(self, reach_task: ReachTask) -> None:
        policy = RandomPolicy(action_dim=7, seed=42)
        runner = EpisodeRunner(task=reach_task, policy=policy)
        result1 = runner.run(seed=123)
        result2 = runner.run(seed=123)
        assert result1.total_reward == pytest.approx(result2.total_reward)
        assert result1.total_steps == result2.total_steps


class TestSweepRunnerIntegration:
    def test_small_sweep(self, reach_task: ReachTask) -> None:
        policy = RandomPolicy(action_dim=7, seed=42)
        sweep_config = SweepConfig(
            stressor_type=LatencyStressor,
            stressor_params={"max_delay_steps": 5},
            intensities=[0.0, 0.5],
            seeds=[0],
            num_episodes_per_config=2,
        )
        runner = SweepRunner(task=reach_task, policy=policy)
        result = runner.run_sweep(sweep_config)

        assert isinstance(result, SweepResult)
        assert result.stressor_name == "LatencyStressor"
        assert len(result.intensity_stats) == 2
        assert result.intensity_stats[0].intensity == 0.0
        assert result.intensity_stats[1].intensity == 0.5
        assert result.intensity_stats[0].num_episodes == 2
        assert 0.0 <= result.intensity_stats[0].success_rate <= 1.0


class TestFullPipeline:
    def test_task_to_sweep_to_report(self, tmp_path: Path) -> None:
        # 1. Create task from config
        task_config = load_task_config("configs/tasks/reach.yaml")
        task_config.max_episode_steps = 30  # Short episodes for speed
        task = ReachTask(task_config)
        task.initialize()

        # 2. Create policy
        policy = RandomPolicy(action_dim=7, seed=0)

        # 3. Run a small sweep
        sweep_config = SweepConfig(
            stressor_type=LatencyStressor,
            stressor_params={"max_delay_steps": 5},
            intensities=[0.0, 1.0],
            seeds=[0],
            num_episodes_per_config=2,
        )
        runner = SweepRunner(task=task, policy=policy)
        result = runner.run_sweep(sweep_config)

        # 4. Generate report
        report_gen = ReportGenerator(output_dir=str(tmp_path))
        report_path = report_gen.generate(
            policy_meta=policy.metadata(),
            task_name="reach",
            sweep_results=[result],
        )

        # 5. Verify report was created
        assert Path(report_path).exists()
        content = Path(report_path).read_text()
        assert "Robustness Report" in content
        assert "LatencyStressor" in content
        assert "RandomPolicy" in content

    def test_pipeline_with_config_loader(self, tmp_path: Path) -> None:
        # Create a minimal sweep config
        sweep_yaml = {
            "sweep": {
                "seeds": [0],
                "episodes_per_config": 2,
                "stressors": [
                    {
                        "type": "DropoutStressor",
                        "intensities": [0.0, 0.5],
                        "params": {"mode": "zero"},
                    }
                ],
            }
        }
        sweep_path = tmp_path / "test_sweep.yaml"
        sweep_path.write_text(yaml.dump(sweep_yaml))

        # Load configs
        task_config = load_task_config("configs/tasks/reach.yaml")
        task_config.max_episode_steps = 20
        task = create_task(task_config)
        sweep_configs = load_sweep_configs(str(sweep_path))

        # Run
        policy = RandomPolicy(action_dim=7, seed=0)
        sweep_runner = SweepRunner(task=task, policy=policy)
        results = [sweep_runner.run_sweep(sc) for sc in sweep_configs]

        assert len(results) == 1
        assert results[0].stressor_name == "DropoutStressor"

    def test_multiple_stressors_sweep(self, reach_task: ReachTask) -> None:
        policy = RandomPolicy(action_dim=7, seed=0)
        sweep_runner = SweepRunner(task=reach_task, policy=policy)

        configs = [
            SweepConfig(
                stressor_type=LatencyStressor,
                intensities=[0.0, 0.5],
                seeds=[0],
                num_episodes_per_config=1,
            ),
            SweepConfig(
                stressor_type=DropoutStressor,
                stressor_params={"mode": "zero"},
                intensities=[0.0, 0.5],
                seeds=[0],
                num_episodes_per_config=1,
            ),
        ]

        results = [sweep_runner.run_sweep(sc) for sc in configs]
        assert len(results) == 2
        assert results[0].stressor_name == "LatencyStressor"
        assert results[1].stressor_name == "DropoutStressor"
