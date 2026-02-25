"""Tests for the YAML configuration loader."""

import tempfile
from pathlib import Path

import pytest
import yaml

from trace.config_loader import (
    STRESSOR_REGISTRY,
    TASK_REGISTRY,
    create_task,
    load_sweep_configs,
    load_task_config,
)
from trace.stressor_engine.latency import LatencyStressor
from trace.task_spec.reach import ReachTask


class TestLoadTaskConfig:
    def test_load_reach_config(self) -> None:
        config = load_task_config("configs/tasks/reach.yaml")
        assert config.name == "reach"
        assert config.max_episode_steps == 200
        assert config.task_params["success_radius"] == 0.05

    def test_load_custom_config(self, tmp_path: Path) -> None:
        cfg = {
            "task": {
                "name": "reach",
                "max_episode_steps": 100,
                "params": {"success_radius": 0.1},
            }
        }
        yaml_path = tmp_path / "test_task.yaml"
        yaml_path.write_text(yaml.dump(cfg))
        config = load_task_config(str(yaml_path))
        assert config.name == "reach"
        assert config.max_episode_steps == 100
        assert config.task_params["success_radius"] == 0.1

    def test_defaults_applied(self, tmp_path: Path) -> None:
        cfg = {"task": {"name": "reach"}}
        yaml_path = tmp_path / "minimal.yaml"
        yaml_path.write_text(yaml.dump(cfg))
        config = load_task_config(str(yaml_path))
        assert config.max_episode_steps == 500
        assert config.success_threshold == 0.95
        assert config.seed == 0


class TestCreateTask:
    def test_create_reach_task(self) -> None:
        config = load_task_config("configs/tasks/reach.yaml")
        task = create_task(config)
        assert isinstance(task, ReachTask)
        # Should be initialized — model should be loaded
        model = task.get_mujoco_model()
        assert model.nq == 7

    def test_unknown_task_raises(self) -> None:
        from trace.task_spec.base import TaskConfig
        config = TaskConfig(name="nonexistent_task")
        with pytest.raises(ValueError, match="Unknown task"):
            create_task(config)


class TestLoadSweepConfigs:
    def test_load_default_sweep(self) -> None:
        configs = load_sweep_configs("configs/sweeps/default_sweep.yaml")
        assert len(configs) == 5  # 5 stressors in default config
        stressor_types = {c.stressor_type.__name__ for c in configs}
        assert "LatencyStressor" in stressor_types
        assert "DropoutStressor" in stressor_types
        assert "PhysicsShiftStressor" in stressor_types

    def test_sweep_config_has_correct_seeds(self) -> None:
        configs = load_sweep_configs("configs/sweeps/default_sweep.yaml")
        for config in configs:
            assert config.seeds == [0, 1, 2, 3, 4]
            assert config.num_episodes_per_config == 10

    def test_custom_sweep_config(self, tmp_path: Path) -> None:
        cfg = {
            "sweep": {
                "seeds": [0, 1],
                "episodes_per_config": 2,
                "stressors": [
                    {
                        "type": "LatencyStressor",
                        "intensities": [0.0, 0.5, 1.0],
                        "params": {"max_delay_steps": 5},
                    }
                ],
            }
        }
        yaml_path = tmp_path / "test_sweep.yaml"
        yaml_path.write_text(yaml.dump(cfg))
        configs = load_sweep_configs(str(yaml_path))
        assert len(configs) == 1
        assert configs[0].stressor_type is LatencyStressor
        assert configs[0].intensities == [0.0, 0.5, 1.0]
        assert configs[0].seeds == [0, 1]

    def test_unknown_stressor_raises(self, tmp_path: Path) -> None:
        cfg = {
            "sweep": {
                "stressors": [{"type": "FakeStressor", "intensities": [0.0]}]
            }
        }
        yaml_path = tmp_path / "bad_sweep.yaml"
        yaml_path.write_text(yaml.dump(cfg))
        with pytest.raises(ValueError, match="Unknown stressor"):
            load_sweep_configs(str(yaml_path))


class TestRegistries:
    def test_task_registry_has_reach(self) -> None:
        assert "reach" in TASK_REGISTRY
        assert TASK_REGISTRY["reach"] is ReachTask

    def test_stressor_registry_has_all_stressors(self) -> None:
        expected = {
            "LatencyStressor",
            "DropoutStressor",
            "PhysicsShiftStressor",
            "EmbodimentStressor",
            "LongHorizonDriftStressor",
        }
        assert set(STRESSOR_REGISTRY.keys()) == expected
