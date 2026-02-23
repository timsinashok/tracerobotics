"""CLI entry point for running evaluations.

Usage:
    python -m scripts.run_evaluation --task pick_and_place --sweep configs/sweeps/default_sweep.yaml
"""

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace Robotics — Policy Evaluation Runner")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name (e.g., pick_and_place, reach_and_grasp, drawer_open)",
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
        "--output",
        type=str,
        default="output/reports",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global random seed",
    )

    args = parser.parse_args()

    logger.info("Trace Robotics Evaluation")
    logger.info("Task: %s", args.task)
    logger.info("Sweep config: %s", args.sweep)
    logger.info("Checkpoint: %s", args.checkpoint or "(random baseline)")
    logger.info("Output: %s", args.output)

    # TODO: Wire up task loading, policy loading, sweep execution, and report generation
    # This will be implemented once concrete MuJoCo task environments are built.
    logger.info("Scaffolding ready. Implement task environments to run evaluations.")


if __name__ == "__main__":
    main()
