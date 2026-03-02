"""YAML configuration loading for tasks and sweeps."""

from pathlib import Path
from typing import Any

import yaml

from trace.runner.sweep_runner import SweepConfig
from trace.stressor_engine.base import BaseStressor
from trace.stressor_engine.dropout import DropoutStressor
from trace.stressor_engine.embodiment import EmbodimentStressor
from trace.stressor_engine.latency import LatencyStressor
from trace.stressor_engine.long_horizon import LongHorizonDriftStressor
from trace.stressor_engine.physics_shift import PhysicsShiftStressor
from trace.stressor_engine.visual import (
    BrightnessShiftStressor,
    ImageNoiseStressor,
    OcclusionStressor,
    ResolutionStressor,
)
from trace.task_spec.base import BaseTask, TaskConfig
from trace.task_spec.reach import ReachTask

# Registries mapping string names to classes
TASK_REGISTRY: dict[str, type[BaseTask]] = {
    "reach": ReachTask,
}

STRESSOR_REGISTRY: dict[str, type[BaseStressor]] = {
    "LatencyStressor": LatencyStressor,
    "DropoutStressor": DropoutStressor,
    "PhysicsShiftStressor": PhysicsShiftStressor,
    "EmbodimentStressor": EmbodimentStressor,
    "LongHorizonDriftStressor": LongHorizonDriftStressor,
    "ImageNoiseStressor": ImageNoiseStressor,
    "OcclusionStressor": OcclusionStressor,
    "BrightnessShiftStressor": BrightnessShiftStressor,
    "ResolutionStressor": ResolutionStressor,
}


def load_task_config(yaml_path: str) -> TaskConfig:
    """Load a TaskConfig from a YAML file."""
    data = yaml.safe_load(Path(yaml_path).read_text())
    task_data = data["task"]
    return TaskConfig(
        name=task_data["name"],
        max_episode_steps=task_data.get("max_episode_steps", 500),
        success_threshold=task_data.get("success_threshold", 0.95),
        seed=task_data.get("seed", 0),
        task_params=task_data.get("params", {}),
    )


def create_task(config: TaskConfig) -> BaseTask:
    """Instantiate a task from a TaskConfig using the task registry."""
    task_cls = TASK_REGISTRY.get(config.name)
    if task_cls is None:
        available = ", ".join(sorted(TASK_REGISTRY.keys()))
        raise ValueError(f"Unknown task: {config.name!r}. Available: {available}")
    task = task_cls(config)
    task.initialize()
    return task


def load_sweep_configs(yaml_path: str) -> list[SweepConfig]:
    """Load sweep configurations from a YAML file."""
    data = yaml.safe_load(Path(yaml_path).read_text())
    sweep_data = data["sweep"]

    seeds = sweep_data.get("seeds", [0, 1, 2, 3, 4])
    episodes_per_config = sweep_data.get("episodes_per_config", 10)

    configs: list[SweepConfig] = []
    for stressor_entry in sweep_data.get("stressors", []):
        type_name = stressor_entry["type"]
        stressor_cls = STRESSOR_REGISTRY.get(type_name)
        if stressor_cls is None:
            available = ", ".join(sorted(STRESSOR_REGISTRY.keys()))
            raise ValueError(
                f"Unknown stressor: {type_name!r}. Available: {available}"
            )
        configs.append(
            SweepConfig(
                stressor_type=stressor_cls,
                stressor_params=stressor_entry.get("params", {}),
                intensities=stressor_entry.get("intensities", [0.0, 0.25, 0.5, 0.75, 1.0]),
                seeds=seeds,
                num_episodes_per_config=episodes_per_config,
            )
        )

    return configs
