"""CLI entry point for running evaluations.

Usage:
    python -m scripts.run_evaluation --task configs/tasks/reach.yaml --sweep configs/sweeps/default_sweep.yaml
"""

import argparse
import datetime
import logging
import sys

from trace.config_loader import create_task, load_sweep_configs, load_task_config
from trace.policy_adapter.pi0_adapter import Pi0PolicyAdapter
from trace.policy_adapter.random_policy import RandomPolicy
from trace.policy_adapter.scripted_reach import ScriptedReachPolicy
from trace.report.generator import ReportGenerator
from trace.runner.sweep_runner import SweepRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

POLICY_REGISTRY: dict[str, type] = {
    "scripted_reach": ScriptedReachPolicy,
    "pi0": Pi0PolicyAdapter,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace Robotics — Policy Evaluation Runner")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Path to task configuration YAML (e.g., configs/tasks/reach.yaml)",
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
    parser.add_argument(
        "--pi0-host",
        type=str,
        default="0.0.0.0",
        help="Pi0 openpi server host",
    )
    parser.add_argument(
        "--pi0-port",
        type=int,
        default=8000,
        help="Pi0 openpi server port",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5,
        help="Number of actions per pi0 inference call",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/reports",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--action-mode",
        type=str,
        default=None,
        choices=["cartesian_delta", "joint_position", "passthrough"],
        help="Action conversion mode (auto-detected from task if omitted)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global random seed",
    )

    args = parser.parse_args()

    logger.info("Trace Robotics Evaluation")
    logger.info("Task config: %s", args.task)
    logger.info("Sweep config: %s", args.sweep)
    logger.info("Checkpoint: %s", args.checkpoint or "(none)")
    logger.info("Output: %s", args.output)

    # Load task
    task_config = load_task_config(args.task)
    task_config.seed = args.seed
    task = create_task(task_config)
    logger.info("Task '%s' initialized", task_config.name)

    # Load policy
    if args.policy and args.policy in POLICY_REGISTRY:
        if args.policy == "pi0":
            # Auto-detect action mode and state format for LIBERO tasks
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
        else:
            policy = POLICY_REGISTRY[args.policy]()
        logger.info("Using named policy: %s", args.policy)
        # Give scripted policies access to the MuJoCo environment
        if isinstance(policy, ScriptedReachPolicy):
            policy.set_env(task.get_mujoco_model(), task.get_mujoco_data())
    else:
        action_dim = task.get_mujoco_model().nu
        policy = RandomPolicy(action_dim=action_dim, seed=args.seed)
        logger.info("Using RandomPolicy baseline (action_dim=%d)", action_dim)

    if args.checkpoint:
        policy.load(args.checkpoint)
        logger.info("Loaded checkpoint: %s", args.checkpoint)

    # Load sweep configs
    sweep_configs = load_sweep_configs(args.sweep)
    logger.info("Loaded %d stressor sweeps", len(sweep_configs))

    # Run sweeps — report is saved progressively after each stressor
    sweep_runner = SweepRunner(task=task, policy=policy)
    report_gen = ReportGenerator(output_dir=args.output)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = (
        f"{args.output}/report_{policy.metadata().name}_{task_config.name}_{timestamp}.md"
    )

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

        # Save report progressively — overwrites with all results so far
        report_gen.generate(
            policy_meta=policy.metadata(),
            task_name=task_config.name,
            sweep_results=sweep_results,
            filepath=report_path,
        )
        logger.info("  Report updated: %s (%d/%d stressors)", report_path, i + 1, len(sweep_configs))

    logger.info("Final report: %s", report_path)


if __name__ == "__main__":
    main()
