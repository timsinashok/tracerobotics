"""GR00T N1 policy adapter — connects to a GR00T server via ZeroMQ.

Wraps NVIDIA's GR00T N1 model (served by PolicyServer) into the
Trace BasePolicy interface. Handles observation mapping from Trace format
to GR00T's nested {video, state, language} convention, action chunking,
and gripper normalization.

GR00T LIBERO observation format:
    video:  {image: (1, T, H, W, 3) uint8, wrist_image: (1, T, H, W, 3) uint8}
    state:  {x: (1,T,1), y: (1,T,1), z: (1,T,1), roll: (1,T,1),
             pitch: (1,T,1), yaw: (1,T,1), gripper: (1,T,2)}
    language: {task: [["instruction"]]}

GR00T LIBERO action format:
    {action.x: (1,T,1), action.y: (1,T,1), ..., action.gripper: (1,T,1)}
    7-dim Cartesian delta: [x, y, z, roll, pitch, yaw, gripper]
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


class GR00TAdapter(BasePolicy):
    """Adapter for GR00T N1 models served by ZeroMQ PolicyServer."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        chunk_size: int = 8,
        timeout_ms: int = 60000,
    ) -> None:
        self._host = host
        self._port = port
        self._chunk_size = chunk_size
        self._timeout_ms = timeout_ms

        self._client: Any = None
        self._action_buffer: deque[np.ndarray] = deque()
        self._prompt: str = ""

    def set_task_info(self, prompt: str) -> None:
        """Set the language instruction for the task."""
        self._prompt = prompt

    def load(self, checkpoint_path: str) -> None:
        """Connect to the GR00T ZeroMQ server.

        The checkpoint_path is unused — the server manages the model.
        Connection is deferred if the gr00t library is not available;
        in that case we use a lightweight ZeroMQ client.
        """
        try:
            from gr00t.policy.server_client import PolicyClient

            self._client = PolicyClient(
                host=self._host,
                port=self._port,
                timeout_ms=self._timeout_ms,
                strict=False,
            )
        except ImportError:
            # Fallback: try raw ZeroMQ connection
            try:
                self._client = _ZmqFallbackClient(
                    host=self._host,
                    port=self._port,
                    timeout_ms=self._timeout_ms,
                )
            except ImportError:
                self._client = None

    def reset(self) -> None:
        """Clear the action buffer between episodes."""
        self._action_buffer.clear()
        if self._client is not None and hasattr(self._client, "reset"):
            try:
                self._client.reset()
            except Exception:
                pass

    def act(self, observation: Observation) -> np.ndarray:
        """Return an action, querying the server when the buffer is empty."""
        if not self._action_buffer:
            self._refill_buffer(observation)
        return self._action_buffer.popleft()

    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            name="GR00TAdapter",
            observation_space={
                "image": {"shape": (256, 256, 3), "dtype": "uint8"},
                "wrist_image": {"shape": (256, 256, 3), "dtype": "uint8"},
                "ee_pos": {"shape": (3,)},
                "ee_orientation": {"shape": (4,)},
                "gripper": {"shape": (2,)},
            },
            action_space={"dim": 7, "low": -1.0, "high": 1.0},
            modalities=["vision", "proprioception"],
        )

    def _refill_buffer(self, observation: Observation) -> None:
        """Query the server for a chunk of actions and fill the buffer."""
        groot_obs = self._build_observation(observation)

        if self._client is not None:
            try:
                result = self._client.get_action(groot_obs)
                # get_action returns (action_dict, info)
                if isinstance(result, tuple):
                    action_dict, _ = result
                else:
                    action_dict = result

                raw_actions = self._parse_action_dict(action_dict)
                for i in range(min(len(raw_actions), self._chunk_size)):
                    action = self._postprocess_action(raw_actions[i])
                    self._action_buffer.append(action)
            except Exception:
                # Server error — fill with zeros
                for _ in range(self._chunk_size):
                    self._action_buffer.append(np.zeros(7, dtype=np.float32))
        else:
            for _ in range(self._chunk_size):
                self._action_buffer.append(np.zeros(7, dtype=np.float32))

    def _build_observation(self, observation: Observation) -> dict[str, Any]:
        """Map Trace observation keys to GR00T's nested format.

        GR00T expects:
            video.image: (B, T, H, W, 3) uint8
            video.wrist_image: (B, T, H, W, 3) uint8
            state.{x,y,z,roll,pitch,yaw}: (B, T, 1) float32
            state.gripper: (B, T, 2) float32
            annotation.human.action.task_description: str
        """
        groot_obs: dict[str, Any] = {}

        # Images: flip 180° for LIBERO convention, add batch+time dims
        if "image" in observation:
            img = np.rot90(observation["image"], k=2).copy()
        else:
            img = np.zeros((256, 256, 3), dtype=np.uint8)
        groot_obs["video.image"] = img[np.newaxis, np.newaxis]  # (1, 1, H, W, 3)

        if "wrist_image" in observation:
            wrist = np.rot90(observation["wrist_image"], k=2).copy()
        else:
            wrist = np.zeros((256, 256, 3), dtype=np.uint8)
        groot_obs["video.wrist_image"] = wrist[np.newaxis, np.newaxis]

        # State: EE pose as axis-angle + gripper
        ee_pos = observation.get("ee_pos", np.zeros(3, dtype=np.float32))
        ee_quat = observation.get(
            "ee_orientation",
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        axis_angle = _quat2axisangle(ee_quat.copy())
        gripper = observation.get("gripper", np.zeros(2, dtype=np.float32))

        # Each state component is (B, T, D)
        groot_obs["state.x"] = np.array([[[ee_pos[0]]]], dtype=np.float32)
        groot_obs["state.y"] = np.array([[[ee_pos[1]]]], dtype=np.float32)
        groot_obs["state.z"] = np.array([[[ee_pos[2]]]], dtype=np.float32)
        groot_obs["state.roll"] = np.array([[[axis_angle[0]]]], dtype=np.float32)
        groot_obs["state.pitch"] = np.array([[[axis_angle[1]]]], dtype=np.float32)
        groot_obs["state.yaw"] = np.array([[[axis_angle[2]]]], dtype=np.float32)
        groot_obs["state.gripper"] = gripper.reshape(1, 1, -1).astype(np.float32)

        # Language
        groot_obs["annotation.human.action.task_description"] = self._prompt

        return groot_obs

    def _parse_action_dict(self, action_dict: dict[str, Any]) -> np.ndarray:
        """Parse GR00T's action dict into (T, 7) array.

        GR00T returns: {action.x: (B,T,1), action.y: ..., action.gripper: (B,T,1)}
        We concatenate into [x, y, z, roll, pitch, yaw, gripper] per timestep.
        """
        keys = [
            "action.x", "action.y", "action.z",
            "action.roll", "action.pitch", "action.yaw",
            "action.gripper",
        ]

        components = []
        for key in keys:
            if key in action_dict:
                val = np.asarray(action_dict[key], dtype=np.float32)
                # Remove batch dim if present: (B, T, D) -> (T, D)
                if val.ndim == 3:
                    val = val[0]
                components.append(val)
            else:
                # Fallback: check if action is a flat array
                break

        if len(components) == 7:
            # (T, 1) each -> (T, 7)
            return np.concatenate(components, axis=-1)

        # Fallback: action might be a flat numpy array
        for key in ["actions", "action"]:
            if key in action_dict:
                arr = np.asarray(action_dict[key], dtype=np.float32)
                if arr.ndim == 3:
                    arr = arr[0]  # Remove batch dim
                if arr.ndim == 1:
                    arr = arr[np.newaxis, :]
                return arr

        # Last resort: try to stack whatever we got
        if components:
            return np.concatenate(components, axis=-1)

        return np.zeros((1, 7), dtype=np.float32)

    def _postprocess_action(self, raw_action: np.ndarray) -> np.ndarray:
        """Post-process a single action for LIBERO.

        GR00T's gripper output follows RLDS convention (0=close, 1=open).
        LIBERO expects: normalized to [-1, 1], then inverted (-1=open, +1=close).
        """
        action = raw_action[:7].copy().astype(np.float32)

        # Normalize gripper: [0, 1] -> [-1, 1]
        action[6] = 2.0 * action[6] - 1.0
        # Invert gripper: LIBERO convention
        action[6] = -action[6]

        return action


class _ZmqFallbackClient:
    """Lightweight ZeroMQ client when gr00t package is not installed.

    Implements just enough of the PolicyClient interface to talk to
    the GR00T server without pulling in the full dependency tree.
    """

    def __init__(self, host: str, port: int, timeout_ms: int = 15000) -> None:
        import io

        import msgpack
        import zmq

        self._zmq = zmq
        self._msgpack = msgpack
        self._io = io
        self._np = np

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{host}:{port}")
        self._timeout_ms = timeout_ms

    def _encode(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            buf = self._io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
        return obj

    def _decode(self, obj: Any) -> Any:
        if isinstance(obj, dict) and "__ndarray_class__" in obj:
            return np.load(self._io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    def _serialize(self, data: Any) -> bytes:
        return self._msgpack.packb(data, default=self._encode)

    def _deserialize(self, data: bytes) -> Any:
        return self._msgpack.unpackb(data, object_hook=self._decode)

    def get_action(self, observation: dict) -> tuple[dict, dict]:
        request = {
            "endpoint": "get_action",
            "data": {"observation": observation},
        }
        self._socket.send(self._serialize(request))
        message = self._socket.recv()
        response = self._deserialize(message)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        if isinstance(response, (list, tuple)):
            return response[0], response[1] if len(response) > 1 else {}
        return response, {}

    def reset(self, options: dict | None = None) -> dict:
        request = {
            "endpoint": "reset",
            "data": {"options": options},
        }
        self._socket.send(self._serialize(request))
        return self._deserialize(self._socket.recv())

    def ping(self) -> bool:
        try:
            request = {"endpoint": "ping"}
            self._socket.send(self._serialize(request))
            self._socket.recv()
            return True
        except Exception:
            return False
