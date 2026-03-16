"""Pi0-FAST policy adapter — loads lerobot/pi0fast-libero from HuggingFace.

Wraps the LeRobot PI0FastPolicy into the Trace BasePolicy interface.
Runs inference locally on GPU (no server needed).
"""

from collections import deque
from typing import Any

import numpy as np
import torch

from trace.policy_adapter.base import BasePolicy, PolicyMetadata
from trace.task_spec.base import Observation


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to axis-angle representation."""
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = np.sqrt(1.0 - w * w)
    if den < 1e-10:
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * np.arccos(w) / den).astype(np.float32)


class Pi0FastAdapter(BasePolicy):
    """Adapter for π₀-FAST models via the LeRobot library."""

    def __init__(
        self,
        model_id: str = "lerobot/pi0fast-libero",
        chunk_size: int = 10,
        device: str = "cuda",
    ) -> None:
        self._model_id = model_id
        self._chunk_size = chunk_size
        self._device = device

        self._policy: Any = None
        self._preprocess: Any = None
        self._postprocess: Any = None
        self._action_buffer: deque[np.ndarray] = deque()
        self._prompt: str = ""

    def set_task_info(self, prompt: str) -> None:
        """Set the language instruction for the task."""
        self._prompt = prompt

    def load(self, checkpoint_path: str) -> None:
        """Load the PI0Fast model from HuggingFace."""
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
        from lerobot.processor.pipeline import DataProcessorPipeline

        # Load config first and fix settings for inference
        config = PreTrainedConfig.from_pretrained(self._model_id)
        config.gradient_checkpointing = False
        config.validate_action_token_prefix = False
        config.device = self._device

        # Load model with fixed config
        self._policy = PI0FastPolicy.from_pretrained(self._model_id, config=config)
        self._policy.to(self._device)
        self._policy.eval()

        # Load the saved preprocessor/postprocessor pipelines from HF repo.
        # These include normalization stats (MEAN_STD for state/action).
        self._preprocess = DataProcessorPipeline.from_pretrained(
            self._model_id,
            config_filename="policy_preprocessor.json",
            overrides={"device_processor": {"device": self._device}},
        )
        self._postprocess = DataProcessorPipeline.from_pretrained(
            self._model_id,
            config_filename="policy_postprocessor.json",
            overrides={"device_processor": {"device": "cpu"}},
        )

    def reset(self) -> None:
        """Clear the action buffer between episodes."""
        self._action_buffer.clear()
        if self._policy is not None:
            self._policy.reset()

    def act(self, observation: Observation) -> np.ndarray:
        """Return an action, using action chunking."""
        if not self._action_buffer:
            self._refill_buffer(observation)
        return self._action_buffer.popleft()

    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            name="Pi0FastAdapter",
            observation_space={
                "image": {"shape": (224, 224, 3), "dtype": "uint8"},
                "wrist_image": {"shape": (224, 224, 3), "dtype": "uint8"},
                "joint_pos": {"shape": (7,)},
                "ee_pos": {"shape": (3,)},
                "ee_orientation": {"shape": (4,)},
                "gripper": {"shape": (1,)},
            },
            action_space={"dim": 7, "low": -1.0, "high": 1.0},
            modalities=["vision", "proprioception"],
        )

    def _refill_buffer(self, observation: Observation) -> None:
        """Run inference and fill the action buffer."""
        if self._policy is None:
            raise RuntimeError("Policy not loaded — call load() first")

        batch = self._build_batch(observation)

        if self._preprocess is not None:
            batch = self._preprocess(batch)

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            actions = self._policy.predict_action_chunk(batch)

        # Postprocess: unnormalize actions. The postprocessor expects a dict,
        # so wrap the tensor and unwrap after.
        if self._postprocess is not None:
            try:
                action_dict = {"action": actions}
                result = self._postprocess(action_dict)
                actions = result["action"] if isinstance(result, dict) else result
            except (ValueError, TypeError):
                pass  # skip postprocessing if format doesn't match

        # actions shape: (batch, n_action_steps, action_dim) or similar
        actions_np = actions.cpu().numpy() if isinstance(actions, torch.Tensor) else np.asarray(actions)
        if actions_np.ndim == 3:
            actions_np = actions_np[0]  # remove batch dim

        for i in range(min(len(actions_np), self._chunk_size)):
            self._action_buffer.append(actions_np[i, :7].astype(np.float32))

    def _build_batch(self, observation: Observation) -> dict[str, Any]:
        """Convert Trace observation to LeRobot raw input format."""
        # Images: CHW float [0,1] — rotated 180° for LIBERO convention
        if "image" in observation:
            base_img = np.rot90(observation["image"], k=2).copy()
        else:
            base_img = np.zeros((224, 224, 3), dtype=np.uint8)

        if "wrist_image" in observation:
            wrist_img = np.rot90(observation["wrist_image"], k=2).copy()
        else:
            wrist_img = np.zeros((224, 224, 3), dtype=np.uint8)

        base_tensor = torch.from_numpy(base_img).permute(2, 0, 1).float() / 255.0
        wrist_tensor = torch.from_numpy(wrist_img).permute(2, 0, 1).float() / 255.0

        # State vector: [ee_pos(3), axis_angle(3), gripper(2)] = 8 dims
        ee_pos = observation.get("ee_pos", np.zeros(3, dtype=np.float32))
        ee_quat = observation.get(
            "ee_orientation",
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        axis_angle = _quat2axisangle(ee_quat.copy())
        gripper = observation.get("gripper", np.zeros(2, dtype=np.float32))
        state = np.concatenate([ee_pos, axis_angle, gripper]).astype(np.float32)
        state_tensor = torch.from_numpy(state)

        return {
            "observation.images.image": base_tensor,
            "observation.images.image2": wrist_tensor,
            "observation.state": state_tensor,
            "task": self._prompt,
        }
