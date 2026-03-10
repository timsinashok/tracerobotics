"""OpenVLA-OFT policy adapter — loads model directly on GPU.

Wraps Stanford's OpenVLA-OFT model into the Trace BasePolicy interface.
Unlike pi0 and GR00T, OpenVLA does not use a server-client architecture —
the model is loaded in-process via HuggingFace transformers.

OpenVLA LIBERO observation format:
    full_image:  (224, 224, 3) uint8 — center-cropped, 180° rotated
    wrist_image: (224, 224, 3) uint8 — same
    state:       (8,) float32 — [ee_pos(3), axis_angle(3), gripper_qpos(2)]

OpenVLA LIBERO action format:
    (7,) float32 — [x, y, z, roll, pitch, yaw, gripper] Cartesian delta
    Gripper: 0=close, 1=open (RLDS convention) — needs normalize + invert for LIBERO

Dependencies:
    - openvla-oft repo (cloned and on sys.path)
    - prismatic, transformers, torch
"""

from collections import deque
from typing import Any

import numpy as np

from trace.policy_adapter.base import BasePolicy, PolicyMetadata
from trace.task_spec.base import Observation


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to axis-angle representation."""
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = np.sqrt(1.0 - w * w)
    if den < 1e-10:
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * np.arccos(w) / den).astype(np.float32)


class OpenVLAAdapter(BasePolicy):
    """Adapter for OpenVLA-OFT models loaded directly on GPU.

    Unlike pi0/GR00T adapters, this loads the model in-process.
    Requires ~16GB VRAM for inference.
    """

    def __init__(
        self,
        checkpoint: str = "moojink/openvla-7b-oft-finetuned-libero-spatial",
        chunk_size: int = 8,
        unnorm_key: str = "libero_spatial_no_noops",
        center_crop: bool = True,
        device: str = "cuda:0",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        openvla_repo_path: str | None = None,
    ) -> None:
        self._checkpoint = checkpoint
        self._chunk_size = chunk_size
        self._unnorm_key = unnorm_key
        self._center_crop = center_crop
        self._device = device
        self._load_in_8bit = load_in_8bit
        self._load_in_4bit = load_in_4bit
        self._openvla_repo_path = openvla_repo_path

        self._model: Any = None
        self._processor: Any = None
        self._action_head: Any = None
        self._proprio_projector: Any = None
        self._cfg: Any = None
        self._action_buffer: deque[np.ndarray] = deque()
        self._prompt: str = ""
        self._loaded = False

    def set_task_info(self, prompt: str) -> None:
        """Set the language instruction for the task."""
        self._prompt = prompt

    def load(self, checkpoint_path: str) -> None:
        """Load the OpenVLA model and components.

        If checkpoint_path is provided and non-empty, it overrides the
        checkpoint set in __init__.
        """
        if checkpoint_path:
            self._checkpoint = checkpoint_path

        try:
            self._load_model()
            self._loaded = True
        except ImportError as e:
            print(f"[OpenVLAAdapter] Could not load model: {e}")
            self._loaded = False

    def _load_model(self) -> None:
        """Import openvla-oft modules and load model components."""
        import sys

        # Add openvla-oft repo to path if specified
        if self._openvla_repo_path and self._openvla_repo_path not in sys.path:
            sys.path.insert(0, self._openvla_repo_path)

        from experiments.robot.libero.run_libero_eval import GenerateConfig
        from experiments.robot.openvla_utils import (
            get_action_head,
            get_processor,
            get_proprio_projector,
            get_vla,
        )

        self._cfg = GenerateConfig(
            pretrained_checkpoint=self._checkpoint,
            use_l1_regression=True,
            use_diffusion=False,
            use_film=False,
            num_images_in_input=2,
            use_proprio=True,
            load_in_8bit=self._load_in_8bit,
            load_in_4bit=self._load_in_4bit,
            center_crop=self._center_crop,
            num_open_loop_steps=self._chunk_size,
            unnorm_key=self._unnorm_key,
        )

        self._model = get_vla(self._cfg)
        self._processor = get_processor(self._cfg)
        self._action_head = get_action_head(
            self._cfg, llm_dim=self._model.llm_dim
        )
        self._proprio_projector = get_proprio_projector(
            self._cfg,
            llm_dim=self._model.llm_dim,
            proprio_dim=8,  # LIBERO: [ee_pos(3), axis_angle(3), gripper_qpos(2)]
        )

    def reset(self) -> None:
        """Clear the action buffer between episodes."""
        self._action_buffer.clear()

    def act(self, observation: Observation) -> np.ndarray:
        """Return an action, querying the model when the buffer is empty."""
        if not self._action_buffer:
            self._refill_buffer(observation)
        return self._action_buffer.popleft()

    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            name="OpenVLAAdapter",
            observation_space={
                "image": {"shape": (224, 224, 3), "dtype": "uint8"},
                "wrist_image": {"shape": (224, 224, 3), "dtype": "uint8"},
                "ee_pos": {"shape": (3,)},
                "ee_orientation": {"shape": (4,)},
                "gripper": {"shape": (2,)},
            },
            action_space={"dim": 7, "low": -1.0, "high": 1.0},
            modalities=["vision", "proprioception"],
        )

    def _refill_buffer(self, observation: Observation) -> None:
        """Query the model for a chunk of actions and fill the buffer."""
        if not self._loaded:
            for _ in range(self._chunk_size):
                self._action_buffer.append(np.zeros(7, dtype=np.float32))
            return

        openvla_obs = self._build_observation(observation)

        try:
            from experiments.robot.openvla_utils import get_vla_action

            actions = get_vla_action(
                cfg=self._cfg,
                vla=self._model,
                processor=self._processor,
                obs=openvla_obs,
                task_label=self._prompt,
                action_head=self._action_head,
                proprio_projector=self._proprio_projector,
            )

            # actions is a list of np.ndarray, each (7,)
            if not isinstance(actions, list):
                actions = [actions]

            for i in range(min(len(actions), self._chunk_size)):
                action = self._postprocess_action(np.asarray(actions[i], dtype=np.float32))
                self._action_buffer.append(action)
        except Exception as e:
            print(f"[OpenVLAAdapter] Inference error: {e}")
            for _ in range(self._chunk_size):
                self._action_buffer.append(np.zeros(7, dtype=np.float32))

    def _build_observation(self, observation: Observation) -> dict[str, Any]:
        """Map Trace observation keys to OpenVLA format.

        OpenVLA expects:
            full_image:  (H, W, 3) uint8 — 180° rotated, center-cropped to 224x224
            wrist_image: (H, W, 3) uint8 — same
            state:       (8,) float32 — [ee_pos(3), axis_angle(3), gripper_qpos(2)]
            task_description: str
        """
        openvla_obs: dict[str, Any] = {}

        # Images: rotate 180°, then resize/center-crop handled by OpenVLA processor
        if "image" in observation:
            img = np.rot90(observation["image"], k=2).copy()
        else:
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        if self._center_crop:
            img = self._center_crop_image(img)
        openvla_obs["full_image"] = img

        if "wrist_image" in observation:
            wrist = np.rot90(observation["wrist_image"], k=2).copy()
        else:
            wrist = np.zeros((224, 224, 3), dtype=np.uint8)

        if self._center_crop:
            wrist = self._center_crop_image(wrist)
        openvla_obs["wrist_image"] = wrist

        # State: [ee_pos(3), axis_angle(3), gripper_qpos(2)] = 8-dim
        ee_pos = observation.get("ee_pos", np.zeros(3, dtype=np.float32))
        ee_quat = observation.get(
            "ee_orientation",
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        axis_angle = _quat2axisangle(ee_quat.copy())
        gripper = observation.get("gripper", np.zeros(2, dtype=np.float32))
        state = np.concatenate([ee_pos, axis_angle, gripper]).astype(np.float32)
        openvla_obs["state"] = state

        # Task description
        openvla_obs["task_description"] = self._prompt

        return openvla_obs

    def _center_crop_image(self, img: np.ndarray, crop_ratio: float = 0.9) -> np.ndarray:
        """Center crop to crop_ratio of original size (matching OpenVLA training augmentation)."""
        h, w = img.shape[:2]
        new_h = int(h * crop_ratio)
        new_w = int(w * crop_ratio)
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        return img[top : top + new_h, left : left + new_w]

    def _postprocess_action(self, raw_action: np.ndarray) -> np.ndarray:
        """Post-process a single action for LIBERO.

        OpenVLA's gripper follows RLDS convention (0=close, 1=open).
        LIBERO expects: normalized to [-1, 1], binarized, then inverted.
        """
        action = raw_action[:7].copy().astype(np.float32)

        # Normalize gripper: [0, 1] -> [-1, 1]
        action[6] = 2.0 * action[6] - 1.0
        # Binarize
        action[6] = np.sign(action[6])
        # Invert: LIBERO convention (-1=open, +1=close) -> flip
        action[6] = -action[6]

        return action
