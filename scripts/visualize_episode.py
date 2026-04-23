"""Standalone episode visualization — runs a single episode and saves a video.

Usage:
    # Baseline (no stressor):
    python -m scripts.visualize_episode \
      --task configs/tasks/libero_spatial.yaml \
      --policy pi0 --pi0-host localhost --pi0-port 8000

    # With a specific stressor and intensity:
    python -m scripts.visualize_episode \
      --task configs/tasks/libero_spatial.yaml \
      --policy pi0 --pi0-host localhost --pi0-port 8000 \
      --stressor LatencyStressor --intensity 0.3

    # With task ID override and custom resolution:
    python -m scripts.visualize_episode \
      --task configs/tasks/libero_spatial.yaml --task-id 4 \
      --policy openvla \
      --stressor ImageNoiseStressor --intensity 0.5 \
      --resolution 512 --seed 42 --output output/videos/
"""

import argparse
import datetime
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from trace.config_loader import (
    STRESSOR_REGISTRY,
    create_task,
    load_task_config,
)
from trace.policy_adapter.base import BasePolicy
from trace.stressor_engine.base import BaseStressor, StressorConfig
from trace.task_spec.base import BaseTask, Observation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_policy(args: argparse.Namespace, task: BaseTask, task_config: Any) -> BasePolicy:
    """Create and initialize a policy from CLI args (mirrors run_evaluation.py)."""
    from trace.policy_adapter.random_policy import RandomPolicy

    if args.policy == "pi0":
        from trace.policy_adapter.pi0_adapter import Pi0PolicyAdapter

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
        from trace.policy_adapter.pi0fast_adapter import Pi0FastAdapter

        policy = Pi0FastAdapter(
            model_id=args.pi0fast_model or "lerobot/pi0fast-libero",
            chunk_size=args.chunk_size if args.chunk_size != 5 else 10,
        )
        policy.set_task_info(task.language_instruction)
        policy.load("")
    elif args.policy == "openvla":
        from trace.policy_adapter.openvla_adapter import OpenVLAAdapter

        policy = OpenVLAAdapter(
            checkpoint=args.openvla_checkpoint,
            chunk_size=args.chunk_size if args.chunk_size != 5 else 8,
            openvla_repo_path=args.openvla_repo_path,
        )
        policy.set_task_info(task.language_instruction)
        policy.load("")
    elif args.policy == "groot":
        from trace.policy_adapter.groot_adapter import GR00TAdapter

        policy = GR00TAdapter(
            host=args.groot_host, port=args.groot_port,
            chunk_size=args.chunk_size if args.chunk_size != 5 else 8,
        )
        policy.set_task_info(task.language_instruction)
        policy.load("")
    else:
        action_dim = task.get_mujoco_model().nu
        policy = RandomPolicy(action_dim=action_dim, seed=args.seed)
        logger.info("Using RandomPolicy baseline (action_dim=%d)", action_dim)

    return policy


def create_stressor(name: str, intensity: float, seed: int) -> BaseStressor:
    """Instantiate a single stressor at a given intensity."""
    stressor_cls = STRESSOR_REGISTRY.get(name)
    if stressor_cls is None:
        available = ", ".join(sorted(STRESSOR_REGISTRY.keys()))
        raise ValueError(f"Unknown stressor: {name!r}. Available: {available}")

    config = StressorConfig(name=name, intensity=intensity, seed=seed)
    return stressor_cls(config)


def render_high_res_frame(env: Any, resolution: int) -> np.ndarray:
    """Render a high-resolution frame from the LIBERO environment.

    Uses robosuite's sim.render() to get a frame at the desired resolution
    from the agentview camera, independent of the policy observation size.
    """
    frame = env.sim.render(
        camera_name="agentview",
        width=resolution,
        height=resolution,
        depth=False,
    )
    # robosuite sim.render returns vertically flipped image
    frame = np.flipud(frame).copy()
    return frame


def run_and_record(
    task: BaseTask,
    policy: BasePolicy,
    stressors: list[BaseStressor],
    seed: int,
    resolution: int,
) -> tuple[list[np.ndarray], bool, int, float]:
    """Run one episode, recording high-res frames at each step.

    Returns (frames, success, total_steps, total_reward).
    """
    policy.reset()
    obs = task.reset(seed=seed)

    for stressor in stressors:
        stressor.on_episode_start(task)

    frames: list[np.ndarray] = []
    total_reward = 0.0
    success = False

    # Capture initial frame
    frames.append(render_high_res_frame(task._env, resolution))

    for step in range(task.config.max_episode_steps):
        # Apply observation stressors
        stressed_obs: Observation = obs
        for stressor in stressors:
            stressed_obs = stressor.perturb_observation(stressed_obs)

        # Get action
        action = policy.act(stressed_obs)

        # Apply action stressors
        stressed_action = action
        for stressor in stressors:
            stressed_action = stressor.perturb_action(stressed_action)

        # Step environment
        obs, reward, done, info = task.step(stressed_action)
        total_reward += reward

        # Capture frame
        frames.append(render_high_res_frame(task._env, resolution))

        # Check success
        if task.check_success():
            success = True
            break

        if task.check_catastrophic_failure() or done:
            break

    for stressor in stressors:
        stressor.on_episode_end()

    return frames, success, step + 1, total_reward


def save_video(frames: list[np.ndarray], output_path: str, fps: int = 20) -> None:
    """Save frames as a video. Tries cv2 -> imageio -> Pillow GIF fallback."""
    try:
        import cv2

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        logger.info("Saved video (cv2): %s (%d frames, %d fps)", output_path, len(frames), fps)
        return
    except ImportError:
        pass

    try:
        import imageio.v3 as iio

        iio.imwrite(output_path, np.stack(frames), fps=fps, codec="libx264")
        logger.info("Saved video (imageio): %s (%d frames, %d fps)", output_path, len(frames), fps)
        return
    except ImportError:
        pass

    # Fallback: save as GIF using Pillow
    gif_path = output_path.replace(".mp4", ".gif")
    pil_frames = [Image.fromarray(f) for f in frames]
    duration_ms = int(1000 / fps)
    pil_frames[0].save(
        gif_path, save_all=True, append_images=pil_frames[1:],
        duration=duration_ms, loop=0,
    )
    logger.info("Saved GIF (Pillow fallback): %s (%d frames)", gif_path, len(frames))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace Robotics — Episode Visualization",
    )
    parser.add_argument(
        "--task", type=str, required=True,
        help="Path to task configuration YAML",
    )
    parser.add_argument(
        "--task-id", type=int, default=None,
        help="Override task ID from config",
    )
    parser.add_argument(
        "--policy", type=str, default=None,
        choices=["pi0", "pi0fast", "openvla", "groot"],
        help="Policy to use (omit for random baseline)",
    )
    parser.add_argument("--stressor", type=str, default=None, help="Stressor name")
    parser.add_argument("--intensity", type=float, default=0.0, help="Stressor intensity [0.0-1.0]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=512, help="Video resolution (square)")
    parser.add_argument("--fps", type=int, default=20, help="Video frame rate")
    parser.add_argument("--output", type=str, default="output/videos", help="Output directory")

    # Policy-specific args
    parser.add_argument("--pi0-host", type=str, default="localhost")
    parser.add_argument("--pi0-port", type=int, default=8000)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--action-mode", type=str, default=None)
    parser.add_argument("--openvla-checkpoint", type=str, default="moojink/openvla-7b-oft-finetuned-libero-spatial")
    parser.add_argument("--openvla-repo-path", type=str, default=None)
    parser.add_argument("--pi0fast-model", type=str, default=None)
    parser.add_argument("--groot-host", type=str, default="localhost")
    parser.add_argument("--groot-port", type=int, default=5555)

    args = parser.parse_args()

    # Load task
    task_config = load_task_config(args.task)
    task_config.seed = args.seed
    if args.task_id is not None:
        task_config.task_params["task_id"] = args.task_id

    task = create_task(task_config)
    suite_name = task_config.task_params.get("task_suite_name", "unknown")
    task_id = task_config.task_params.get("task_id", 0)

    logger.info("Task: %s (task_id=%d)", suite_name, task_id)
    logger.info("Instruction: %s", task.language_instruction)

    # Create policy
    policy = create_policy(args, task, task_config)
    policy_name = args.policy or "random"
    logger.info("Policy: %s", policy_name)

    # Create stressor (if specified)
    stressors: list[BaseStressor] = []
    stressor_label = "baseline"
    if args.stressor:
        stressor = create_stressor(args.stressor, args.intensity, args.seed)
        stressors.append(stressor)
        stressor_label = f"{args.stressor}_i{args.intensity:.2f}"
        logger.info("Stressor: %s @ intensity %.2f", args.stressor, args.intensity)

    # Run episode and record
    logger.info("Recording episode at %dx%d resolution...", args.resolution, args.resolution)
    frames, success, total_steps, total_reward = run_and_record(
        task, policy, stressors, args.seed, args.resolution,
    )

    logger.info("Episode result: success=%s, steps=%d, reward=%.3f", success, total_steps, total_reward)

    # Save video
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{policy_name}_{suite_name}_t{task_id}_{stressor_label}_s{args.seed}_{timestamp}.mp4"
    output_path = str(output_dir / filename)

    save_video(frames, output_path, fps=args.fps)

    # Print summary
    print()
    print(f"{'SUCCESS' if success else 'FAILURE'} — {total_steps} steps, reward {total_reward:.3f}")
    print(f"Video: {output_path}")

    task.close()


if __name__ == "__main__":
    main()
