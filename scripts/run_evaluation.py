"""CLI entry point for running evaluations.

Usage:
    # Single task (backwards compatible):
    python -m scripts.run_evaluation --task configs/tasks/libero_spatial.yaml --policy pi0

    # Multi-task across suites:
    python -m scripts.run_evaluation \
      --task configs/tasks/libero_spatial.yaml configs/tasks/libero_object.yaml \
            configs/tasks/libero_goal.yaml configs/tasks/libero_10.yaml \
      --task-ids 0 4 9 \
      --sweep configs/sweeps/multi_task_sweep.yaml \
      --policy pi0
"""

import argparse
import datetime
import logging
import sys

from trace.config_loader import create_task, load_sweep_configs, load_task_config
from trace.policy_adapter.groot_adapter import GR00TAdapter
from trace.policy_adapter.openvla_adapter import OpenVLAAdapter
from trace.policy_adapter.pi0_adapter import Pi0PolicyAdapter
from trace.policy_adapter.pi0fast_adapter import Pi0FastAdapter
from trace.policy_adapter.random_policy import RandomPolicy
from trace.policy_adapter.scripted_reach import ScriptedReachPolicy
from trace.report.generator import ReportGenerator, TaskSweepResults
from trace.runner.sweep_runner import SweepRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

POLICY_REGISTRY: dict[str, type] = {
    "scripted_reach": ScriptedReachPolicy,
    "pi0": Pi0PolicyAdapter,
    "pi0fast": Pi0FastAdapter,
    "groot": GR00TAdapter,
    "openvla": OpenVLAAdapter,
}


def build_task_entries(args) -> list[tuple[str, int | None]]:
    """Return list of (config_path, task_id_override) pairs."""
    task_ids = args.task_ids
    entries = []
    for config_path in args.task:
        if task_ids:
            for tid in task_ids:
                entries.append((config_path, tid))
        else:
            entries.append((config_path, None))
    return entries


def create_policy(args, task, task_config):
    """Create and initialize a policy from CLI args."""
    if not (args.policy and args.policy in POLICY_REGISTRY):
        action_dim = task.get_mujoco_model().nu
        policy = RandomPolicy(action_dim=action_dim, seed=args.seed)
        logger.info("Using RandomPolicy baseline (action_dim=%d)", action_dim)
        return policy

    if args.policy == "pi0":
        action_mode = args.action_mode or (
            "passthrough" if task_config.name == "libero" else "cartesian_delta"
        )
        state_format = "axis_angle" if task_config.name == "libero" else "quaternion"
        policy = Pi0PolicyAdapter(
            host=args.pi0_host,
            port=args.pi0_port,
            chunk_size=args.chunk_size,
            action_mode=action_mode,
            state_format=state_format,
        )
        policy.set_env(task.get_mujoco_model(), task.get_mujoco_data())
        policy.set_task_info(task.language_instruction)
        policy.load("")
    elif args.policy == "pi0fast":
        policy = Pi0FastAdapter(
            model_id=args.pi0fast_model or "lerobot/pi0fast-libero",
            chunk_size=args.chunk_size if args.chunk_size != 5 else 10,
        )
        policy.set_task_info(task.language_instruction)
        policy.load("")
    elif args.policy == "groot":
        chunk_size = args.chunk_size if args.chunk_size != 5 else 8
        policy = GR00TAdapter(
            host=args.groot_host, port=args.groot_port, chunk_size=chunk_size,
        )
        policy.set_task_info(task.language_instruction)
        policy.load("")
    elif args.policy == "openvla":
        chunk_size = args.chunk_size if args.chunk_size != 5 else 8
        policy = OpenVLAAdapter(
            checkpoint=args.openvla_checkpoint,
            chunk_size=chunk_size,
            openvla_repo_path=args.openvla_repo_path,
        )
        policy.set_task_info(task.language_instruction)
        policy.load("")
    else:
        policy = POLICY_REGISTRY[args.policy]()
        if isinstance(policy, ScriptedReachPolicy):
            policy.set_env(task.get_mujoco_model(), task.get_mujoco_data())

    logger.info("Using named policy: %s", args.policy)

    if args.checkpoint:
        policy.load(args.checkpoint)
        logger.info("Loaded checkpoint: %s", args.checkpoint)

    return policy


def rebind_policy(policy, task, args) -> None:
    """Re-bind an existing policy to a new task (new env + new instruction)."""
    if isinstance(policy, Pi0PolicyAdapter):
        policy.set_env(task.get_mujoco_model(), task.get_mujoco_data())
        policy.set_task_info(task.language_instruction)
    elif isinstance(policy, (Pi0FastAdapter, GR00TAdapter, OpenVLAAdapter)):
        policy.set_task_info(task.language_instruction)
    elif isinstance(policy, ScriptedReachPolicy):
        policy.set_env(task.get_mujoco_model(), task.get_mujoco_data())


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace Robotics — Policy Evaluation Runner")
    parser.add_argument(
        "--task",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to task configuration YAML(s)",
    )
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        default=None,
        help="Task IDs to evaluate per suite (e.g., 0 4 9). If omitted, uses YAML default.",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default="configs/sweeps/default_sweep.yaml",
        help="Path to sweep configuration YAML",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to policy checkpoint (omit to use random baseline)",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        choices=list(POLICY_REGISTRY.keys()),
        help="Named policy to use (e.g., scripted_reach, pi0)",
    )
    parser.add_argument("--pi0-host", type=str, default="0.0.0.0")
    parser.add_argument("--pi0-port", type=int, default=8000)
    parser.add_argument(
        "--chunk-size", type=int, default=5,
        help="Number of actions per inference call (pi0: 5, groot: 8)",
    )
    parser.add_argument("--groot-host", type=str, default="localhost")
    parser.add_argument("--groot-port", type=int, default=5555)
    parser.add_argument(
        "--openvla-checkpoint", type=str,
        default="moojink/openvla-7b-oft-finetuned-libero-spatial",
    )
    parser.add_argument("--openvla-repo-path", type=str, default=None)
    parser.add_argument("--pi0fast-model", type=str, default=None)
    parser.add_argument("--output", type=str, default="output/reports")
    parser.add_argument(
        "--action-mode", type=str, default=None,
        choices=["cartesian_delta", "joint_position", "passthrough"],
    )
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    task_entries = build_task_entries(args)
    multi_task = len(task_entries) > 1

    logger.info("Trace Robotics Evaluation")
    logger.info("Tasks: %d entries", len(task_entries))
    logger.info("Sweep config: %s", args.sweep)
    logger.info("Output: %s", args.output)

    sweep_configs = load_sweep_configs(args.sweep)
    logger.info("Loaded %d stressor sweeps", len(sweep_configs))

    report_gen = ReportGenerator(output_dir=args.output)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create policy with the first task (it will be re-bound for subsequent tasks)
    first_config_path, first_task_id = task_entries[0]
    first_task_config = load_task_config(first_config_path)
    first_task_config.seed = args.seed
    if first_task_id is not None:
        first_task_config.task_params["task_id"] = first_task_id
    first_task = create_task(first_task_config)
    policy = create_policy(args, first_task, first_task_config)

    if multi_task:
        _run_multi_task(
            args, task_entries, sweep_configs, policy, report_gen, timestamp,
            first_task, first_task_config,
        )
    else:
        _run_single_task(
            args, first_task, first_task_config, sweep_configs, policy,
            report_gen, timestamp,
        )


def _run_single_task(args, task, task_config, sweep_configs, policy, report_gen, timestamp):
    """Original single-task evaluation path."""
    report_path = (
        f"{args.output}/report_{policy.metadata().name}_{task_config.name}_{timestamp}.md"
    )
    sweep_runner = SweepRunner(task=task, policy=policy)
    sweep_results = []

    for i, sc in enumerate(sweep_configs):
        logger.info(
            "Running sweep %d/%d: %s", i + 1, len(sweep_configs), sc.stressor_type.__name__
        )
        result = sweep_runner.run_sweep(sc)
        sweep_results.append(result)
        logger.info(
            "  baseline=%.0f%%, max_degradation=%.0f%%, breakpoint=%s",
            result.baseline_success_rate() * 100,
            result.max_degradation() * 100,
            f"{result.breakpoint_intensity:.2f}" if result.breakpoint_intensity else "none",
        )
        report_gen.generate(
            policy_meta=policy.metadata(),
            task_name=task_config.name,
            sweep_results=sweep_results,
            filepath=report_path,
        )
        logger.info("  Report updated: %s (%d/%d stressors)", report_path, i + 1, len(sweep_configs))

    logger.info("Final report: %s", report_path)


def _run_multi_task(
    args, task_entries, sweep_configs, policy, report_gen, timestamp,
    first_task, first_task_config,
):
    """Multi-task evaluation: loop over tasks, collect results, unified report."""
    report_path = (
        f"{args.output}/report_{policy.metadata().name}_libero_multi_{timestamp}.md"
    )

    all_task_results: list[TaskSweepResults] = []

    for t_idx, (config_path, task_id) in enumerate(task_entries):
        # Reuse first task if it matches, otherwise create new
        if t_idx == 0:
            task_config = first_task_config
            task = first_task
        else:
            task_config = load_task_config(config_path)
            task_config.seed = args.seed
            if task_id is not None:
                task_config.task_params["task_id"] = task_id
            task = create_task(task_config)
            rebind_policy(policy, task, args)

        suite_name = task_config.task_params.get("task_suite_name", "unknown")
        actual_task_id = task_config.task_params.get("task_id", 0)
        task_label = f"{suite_name}_task{actual_task_id}"

        logger.info(
            "=== Task %d/%d: %s (instruction: %s) ===",
            t_idx + 1, len(task_entries), task_label, task.language_instruction,
        )

        # Run all stressor sweeps for this task
        sweep_runner = SweepRunner(task=task, policy=policy)
        sweep_results = []
        for i, sc in enumerate(sweep_configs):
            logger.info(
                "  Sweep %d/%d: %s", i + 1, len(sweep_configs), sc.stressor_type.__name__
            )
            result = sweep_runner.run_sweep(sc)
            sweep_results.append(result)
            logger.info(
                "    baseline=%.0f%%, max_degradation=%.0f%%",
                result.baseline_success_rate() * 100,
                result.max_degradation() * 100,
            )

        all_task_results.append(TaskSweepResults(
            suite_name=suite_name,
            task_id=actual_task_id,
            task_label=task_label,
            language_instruction=task.language_instruction,
            sweep_results=sweep_results,
        ))

        # Progressive save after each task
        report_gen.generate_multi_task(
            policy_meta=policy.metadata(),
            task_results=all_task_results,
            filepath=report_path,
        )
        logger.info(
            "Report updated: %s (%d/%d tasks)", report_path, t_idx + 1, len(task_entries),
        )

        # Free env resources
        if t_idx > 0 and hasattr(task, "close"):
            task.close()

    logger.info("Final multi-task report: %s", report_path)


if __name__ == "__main__":
    main()
