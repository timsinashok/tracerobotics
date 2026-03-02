"""Pi0 policy adapter — connects to an openpi server via WebSocket.

Wraps Physical Intelligence's pi0 model (served by openpi) into the
Trace BasePolicy interface. Handles observation mapping from Trace format
to LIBERO convention, action chunking, and Cartesian-to-joint-space
conversion via Jacobian transpose.
"""

from collections import deque
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from trace.policy_adapter.base import BasePolicy, PolicyMetadata
from trace.task_spec.base import Observation


class Pi0PolicyAdapter(BasePolicy):
    """Adapter for pi0 models served by openpi WebSocket server."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        chunk_size: int = 5,
        action_mode: str = "cartesian_delta",
        gain: float = 10.0,
    ) -> None:
        self._host = host
        self._port = port
        self._chunk_size = chunk_size
        self._action_mode = action_mode
        self._gain = gain

        self._client: Any = None
        self._action_buffer: deque[np.ndarray] = deque()
        self._prompt: str = ""

        # MuJoCo references (set via set_env)
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._ee_site_id: int = -1

    def set_env(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Provide MuJoCo model/data for Jacobian-based action conversion."""
        self._model = model
        self._data = data
        self._ee_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "end_effector"
        )

    def set_task_info(self, prompt: str) -> None:
        """Set the language instruction for the task."""
        self._prompt = prompt

    def load(self, checkpoint_path: str) -> None:
        """Connect to the openpi WebSocket server.

        The checkpoint_path is unused — the server manages the model.
        Connection is deferred until first act() call if the client
        library is not available.
        """
        try:
            from openpi_client import websocket_client_policy

            self._client = websocket_client_policy.WebsocketClientPolicy(
                host=self._host,
                port=self._port,
            )
        except ImportError:
            self._client = None

    def reset(self) -> None:
        """Clear the action buffer between episodes."""
        self._action_buffer.clear()

    def act(self, observation: Observation) -> np.ndarray:
        """Return an action, querying the server when the buffer is empty.

        Uses action chunking: one server call returns chunk_size actions.
        Subsequent act() calls pop from the buffer until it's empty,
        then a new server call is made.
        """
        if not self._action_buffer:
            self._refill_buffer(observation)
        return self._action_buffer.popleft()

    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            name="Pi0PolicyAdapter",
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
        """Query the server for a chunk of actions and fill the buffer."""
        openpi_obs = self._build_observation(observation)

        if self._client is not None:
            raw_actions = self._client.infer(openpi_obs)
            # raw_actions is (chunk_size, action_dim) or (action_dim,)
            raw_actions = np.asarray(raw_actions, dtype=np.float32)
            if raw_actions.ndim == 1:
                raw_actions = raw_actions[np.newaxis, :]
            # Take up to chunk_size actions
            for i in range(min(len(raw_actions), self._chunk_size)):
                action = self._convert_action(raw_actions[i], observation)
                self._action_buffer.append(action)
        else:
            # Fallback: no server connected, return zeros
            action_dim = 7
            if self._model is not None:
                action_dim = self._model.nu
            for _ in range(self._chunk_size):
                self._action_buffer.append(
                    np.zeros(action_dim, dtype=np.float32)
                )

    def _build_observation(self, observation: Observation) -> dict[str, Any]:
        """Map Trace observation keys to openpi LIBERO convention."""
        openpi_obs: dict[str, Any] = {}

        # Image: rotate 180 degrees for LIBERO convention
        if "image" in observation:
            img = observation["image"]
            openpi_obs["observation/image"] = np.rot90(img, k=2).copy()
        else:
            openpi_obs["observation/image"] = np.zeros(
                (224, 224, 3), dtype=np.uint8
            )

        # Wrist image: rotate 180 degrees for LIBERO convention
        if "wrist_image" in observation:
            wrist = observation["wrist_image"]
            openpi_obs["observation/wrist_image"] = np.rot90(wrist, k=2).copy()
        else:
            openpi_obs["observation/wrist_image"] = np.zeros(
                (224, 224, 3), dtype=np.uint8
            )

        # State: 8-dim vector [joint_pos(7) truncated or ee_pos(3) + ee_orientation(4) + gripper(1)]
        state = self._build_state_vector(observation)
        openpi_obs["observation/state"] = state

        # Language prompt
        openpi_obs["prompt"] = self._prompt

        return openpi_obs

    def _build_state_vector(self, observation: Observation) -> NDArray[np.floating]:
        """Build the 8-dim state vector for openpi.

        Layout: [ee_pos(3), ee_orientation(4), gripper(1)] = 8 dims.
        Falls back to zeros for missing keys.
        """
        ee_pos = observation.get(
            "ee_pos", np.zeros(3, dtype=np.float32)
        )
        ee_orientation = observation.get(
            "ee_orientation", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )
        gripper = observation.get(
            "gripper", np.zeros(1, dtype=np.float32)
        )
        state = np.concatenate([ee_pos, ee_orientation, gripper]).astype(np.float32)
        return state

    def _convert_action(
        self, raw_action: np.ndarray, observation: Observation
    ) -> np.ndarray:
        """Convert pi0's raw action output to joint-space [-1, 1].

        For action_mode="cartesian_delta" (default):
            Uses Jacobian transpose to map Cartesian delta → joint delta,
            same math as ScriptedReachPolicy.

        For action_mode="joint_position":
            Directly maps to [-1, 1] using actuator ctrl ranges.
        """
        if self._action_mode == "joint_position":
            return self._convert_joint_position(raw_action)
        else:
            return self._convert_cartesian_delta(raw_action, observation)

    def _convert_cartesian_delta(
        self, raw_action: np.ndarray, observation: Observation
    ) -> np.ndarray:
        """Convert Cartesian delta action to joint-space [-1, 1] via Jacobian transpose."""
        if self._model is None or self._data is None:
            # No MuJoCo env: return clipped raw action
            return np.clip(raw_action[:7], -1.0, 1.0).astype(np.float32)

        # Extract position delta (first 3 dims of raw action)
        cart_delta = raw_action[:3].astype(np.float64)

        # Compute Jacobian
        jacp = np.zeros((3, self._model.nv))
        mujoco.mj_jacSite(self._model, self._data, jacp, None, self._ee_site_id)

        # Jacobian transpose: Cartesian delta → joint-space delta
        joint_delta = jacp.T @ cart_delta

        # Desired joint positions
        joint_pos = observation.get(
            "joint_pos",
            self._data.qpos[:self._model.nq].astype(np.float32),
        )
        desired = joint_pos + self._gain * joint_delta

        # Map to [-1, 1] action space
        ctrl_range = self._model.actuator_ctrlrange
        action = (
            2.0 * (desired - ctrl_range[:, 0])
            / (ctrl_range[:, 1] - ctrl_range[:, 0])
            - 1.0
        )
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def _convert_joint_position(self, raw_action: np.ndarray) -> np.ndarray:
        """Directly interpret raw action as joint positions in [-1, 1]."""
        action_dim = 7
        if self._model is not None:
            action_dim = self._model.nu
        action = raw_action[:action_dim]
        return np.clip(action, -1.0, 1.0).astype(np.float32)
