# LIBERO Task Integration — Development Plan

**Date:** 2026-03-04
**Author:** Trace Robotics Team
**Status:** Ready for implementation

---

## 1. Motivation

Pi0.5-LIBERO scores **0% on our custom reach task** because it was never trained on "reach the target" — it was trained on LIBERO manipulation tasks (pick/place objects, open drawers, etc.). To get meaningful robustness reports from Trace, we must evaluate pi0 on tasks it actually understands.

LIBERO provides **130 tabletop manipulation tasks** across 5 suites, all in MuJoCo with a Franka Panda arm. Pi0.5-LIBERO achieves **92–99% success** on these. By wrapping LIBERO environments into Trace's `BaseTask`, we get a single `LiberoTask` that works for ALL vision-language-action models — not just pi0.

**Goal:** Run `python -m scripts.run_evaluation --task configs/tasks/libero_spatial.yaml --policy pi0` and get a real robustness report showing where pi0 degrades under stress.

---

## 2. LIBERO Environment Interface

Reference implementation: `openpi/examples/libero/main.py`

```python
# Create env
env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256)
env.seed(seed)
env.reset()
obs = env.set_init_state(initial_state)

# Step
obs, reward, done, info = env.step(action)  # action: 7-dim [arm_delta(6), gripper(1)]

# Observations
obs["agentview_image"]           # (256, 256, 3) uint8
obs["robot0_eye_in_hand_image"]  # (256, 256, 3) uint8
obs["robot0_eef_pos"]            # (3,) end-effector position
obs["robot0_eef_quat"]           # (4,) quaternion
obs["robot0_gripper_qpos"]       # (2,) gripper joint positions
```

### Pi0 Action Format for LIBERO
- **7-dim:** `[arm_delta_x, y, z, rx, ry, rz, gripper]`
- Goes **directly** to `env.step()` — no Jacobian conversion needed
- Current Pi0 adapter does unnecessary Cartesian→joint conversion that must be bypassed

### Pi0 State Vector for LIBERO
- **8-dim:** `[ee_pos(3), axis_angle(3), gripper_qpos(2)]`
- Orientation must be **axis-angle** (not quaternion) — requires `_quat2axisangle()` conversion
- This differs from the current adapter which sends `[ee_pos(3), ee_quat(4), gripper(1)]`

---

## 3. Dependencies

```bash
# In trace-pi0 conda environment (Python 3.11)
pip install robosuite==1.4.1 bddl==1.0.1 "gym==0.25.2"
pip install -e /scratch/at5282/trace/openpi/third_party/libero

# Verify
python -c "from libero.libero import benchmark; print('OK')"
```

**Compatibility notes:**
- `robosuite==1.4.1` uses `mujoco==3.2.3` (modern bindings, NOT mujoco-py)
- Risk: `gym` vs `gymnasium` version conflict (LIBERO needs `gym==0.25.2`)
- LIBERO is installed from openpi's vendored third_party copy

---

## 4. Implementation Steps

### Step 1: Create `trace/task_spec/libero_task.py`

New file implementing `BaseTask`. This is the core integration point.

```python
"""LIBERO task wrapper — wraps LIBERO environments into Trace's BaseTask interface.

Supports all 5 LIBERO suites (libero_spatial, libero_object, libero_goal,
libero_10, libero_90) with 10-90 tasks each. Each task is a tabletop
manipulation scenario with a Franka Panda arm.
"""

import math
from typing import Any

import numpy as np
from PIL import Image

from trace.task_spec.base import BaseTask, Observation, TaskConfig


class LiberoTask(BaseTask):
    """Wraps a LIBERO environment into Trace's BaseTask interface."""

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)
        self._env = None
        self._task_suite = None
        self._initial_states = None
        self._episode_idx = 0
        self._done = False
        self._last_reward = 0.0

        # Task parameters
        params = config.task_params
        self._suite_name: str = params.get("task_suite_name", "libero_spatial")
        self._task_id: int = params.get("task_id", 0)
        self._num_steps_wait: int = params.get("num_steps_wait", 10)
        self._render_width: int = params.get("render_width", 224)
        self._render_height: int = params.get("render_height", 224)
        self._task_description: str = ""

        # Cached last observation from LIBERO env
        self._last_obs: dict[str, Any] | None = None

    def initialize(self) -> None:
        """Create the LIBERO environment from benchmark."""
        from libero.libero import benchmark as libero_benchmark
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        import pathlib

        # Load benchmark and task suite
        benchmark_dict = libero_benchmark.get_benchmark_dict()
        self._task_suite = benchmark_dict[self._suite_name]()
        task = self._task_suite.get_task(self._task_id)

        # Get task description (language instruction)
        self._task_description = task.language
        if self.config.task_params.get("language_instruction", "auto") == "auto":
            # Use LIBERO's built-in task description
            pass
        else:
            self._task_description = self.config.task_params["language_instruction"]

        # Create environment
        task_bddl_file = (
            pathlib.Path(get_libero_path("bddl_files"))
            / task.problem_folder
            / task.bddl_file
        )
        self._env = OffScreenRenderEnv(
            bddl_file_name=str(task_bddl_file),
            camera_heights=256,  # Native LIBERO resolution
            camera_widths=256,
        )
        self._env.seed(self.config.seed)

        # Get initial states for reproducible episodes
        self._initial_states = self._task_suite.get_task_init_states(self._task_id)

    def reset(self, seed: int | None = None) -> Observation:
        """Reset the LIBERO environment and return initial observation."""
        assert self._env is not None, "Call initialize() first"

        if seed is not None:
            self._env.seed(seed)
            self._episode_idx = seed % len(self._initial_states)
        else:
            self._episode_idx = (self._episode_idx + 1) % len(self._initial_states)

        self._done = False
        self._last_reward = 0.0

        # Reset and set initial state
        self._env.reset()
        obs = self._env.set_init_state(self._initial_states[self._episode_idx])

        # Wait for objects to settle (LIBERO drops objects at start)
        dummy_action = [0.0] * 6 + [-1.0]
        for _ in range(self._num_steps_wait):
            obs, _, _, _ = self._env.step(dummy_action)

        self._last_obs = obs
        return self.get_observation()

    def step(
        self, action: np.ndarray
    ) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Execute one action in the LIBERO environment.

        Action is 7-dim [arm_delta(6), gripper(1)] — passed directly to env.
        No Jacobian conversion needed.
        """
        assert self._env is not None

        # Convert to list for LIBERO env
        action_list = action[:7].tolist()
        obs, reward, done, info = self._env.step(action_list)

        self._last_obs = obs
        self._done = bool(done)
        self._last_reward = float(reward)

        trace_obs = self.get_observation()
        return trace_obs, self._last_reward, self._done, info

    def check_success(self) -> bool:
        """Return True if the LIBERO task is completed."""
        return self._done

    def check_catastrophic_failure(self) -> bool:
        """LIBERO tasks don't have catastrophic failures — return False."""
        return False

    def get_observation(self) -> Observation:
        """Map LIBERO observation keys to Trace format."""
        assert self._last_obs is not None

        obs: Observation = {}

        # Images: resize from 256 native to configured size (typically 224)
        if "agentview_image" in self._last_obs:
            obs["image"] = self._resize_image(self._last_obs["agentview_image"])
        if "robot0_eye_in_hand_image" in self._last_obs:
            obs["wrist_image"] = self._resize_image(
                self._last_obs["robot0_eye_in_hand_image"]
            )

        # Proprioception
        if "robot0_eef_pos" in self._last_obs:
            obs["ee_pos"] = self._last_obs["robot0_eef_pos"].astype(np.float32).copy()
        if "robot0_eef_quat" in self._last_obs:
            obs["ee_orientation"] = (
                self._last_obs["robot0_eef_quat"].astype(np.float32).copy()
            )
        if "robot0_gripper_qpos" in self._last_obs:
            obs["gripper"] = (
                self._last_obs["robot0_gripper_qpos"].astype(np.float32).copy()
            )

        return obs

    @property
    def language_instruction(self) -> str:
        """Return the LIBERO task description as the language instruction."""
        return self._task_description

    def get_mujoco_model(self) -> Any:
        """Return the underlying MuJoCo model from robosuite."""
        assert self._env is not None, "Call initialize() first"
        return self._env.sim.model._model

    def get_mujoco_data(self) -> Any:
        """Return the underlying MuJoCo data from robosuite."""
        assert self._env is not None, "Call initialize() first"
        return self._env.sim.data._data

    def close(self) -> None:
        """Clean up the LIBERO environment."""
        if self._env is not None:
            self._env.close()
            self._env = None

    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """Resize image from native 256x256 to configured resolution."""
        if img.shape[0] == self._render_height and img.shape[1] == self._render_width:
            return img.copy()
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize(
            (self._render_width, self._render_height), Image.BILINEAR
        )
        return np.array(pil_img, dtype=np.uint8)
```

**Key design decisions:**
- `initialize()` defers LIBERO imports (not all nodes have LIBERO installed)
- `reset()` cycles through LIBERO's pre-recorded initial states for reproducibility
- `step()` passes raw 7-dim actions directly — no conversion
- `get_observation()` maps LIBERO obs keys to Trace convention
- `get_mujoco_model()`/`get_mujoco_data()` expose robosuite's underlying MuJoCo objects for physics stressors
- Images are resized from 256 native to 224 (pi0 input size)

### Step 2: Modify `trace/policy_adapter/pi0_adapter.py`

Two changes needed:

#### 2a. Add passthrough action mode

The current `_convert_action()` always does Jacobian conversion, which is wrong for LIBERO (LIBERO expects raw Cartesian deltas). Add a `"passthrough"` mode:

```python
def _convert_action(self, raw_action, observation):
    if self._action_mode == "passthrough":
        # Raw 7-dim goes directly to LIBERO env — no conversion
        return raw_action[:7].astype(np.float32)
    elif self._action_mode == "joint_position":
        return self._convert_joint_position(raw_action)
    else:  # cartesian_delta
        return self._convert_cartesian_delta(raw_action, observation)
```

#### 2b. Fix state vector for LIBERO

The current `_build_state_vector()` sends `[ee_pos(3), ee_quat(4), gripper(1)]` = 8 dims with quaternion. LIBERO pi0 expects `[ee_pos(3), axis_angle(3), gripper_qpos(2)]` = 8 dims with axis-angle.

Add a `state_format` config option:

```python
def __init__(self, ..., state_format: str = "quaternion"):
    self._state_format = state_format

def _build_state_vector(self, observation):
    ee_pos = observation.get("ee_pos", np.zeros(3, dtype=np.float32))

    if self._state_format == "axis_angle":
        # LIBERO format: [ee_pos(3), axis_angle(3), gripper_qpos(2)]
        ee_quat = observation.get(
            "ee_orientation", np.array([1, 0, 0, 0], dtype=np.float32)
        )
        axis_angle = _quat2axisangle(ee_quat.copy())
        gripper = observation.get("gripper", np.zeros(2, dtype=np.float32))
        return np.concatenate([ee_pos, axis_angle, gripper]).astype(np.float32)
    else:
        # Default: [ee_pos(3), ee_quat(4), gripper(1)]
        ee_quat = observation.get(
            "ee_orientation", np.array([1, 0, 0, 0], dtype=np.float32)
        )
        gripper = observation.get("gripper", np.zeros(1, dtype=np.float32))
        return np.concatenate([ee_pos, ee_quat, gripper]).astype(np.float32)
```

Add the `_quat2axisangle()` helper (from `openpi/examples/libero/main.py`):

```python
def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to axis-angle representation."""
    # Clip w component
    w = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - w * w)
    if den < 1e-10:
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(w)) / den
```

#### 2c. Make `set_env()` optional for passthrough mode

In passthrough mode, the adapter doesn't need MuJoCo references for Jacobian computation. Make `set_env()` gracefully skip the site lookup when `action_mode == "passthrough"`:

```python
def set_env(self, model, data):
    self._model = model
    self._data = data
    if self._action_mode != "passthrough":
        self._ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
```

### Step 3: Modify `trace/config_loader.py`

Register `LiberoTask` in the task registry:

```python
from trace.task_spec.libero_task import LiberoTask

TASK_REGISTRY: dict[str, type[BaseTask]] = {
    "reach": ReachTask,
    "libero": LiberoTask,
}
```

### Step 4: Modify `scripts/run_evaluation.py`

Update the pi0 policy creation to support passthrough mode and axis-angle state format when used with LIBERO tasks:

```python
if args.policy == "pi0":
    # Determine action mode and state format based on task
    action_mode = args.action_mode or ("passthrough" if task_config.name == "libero" else "cartesian_delta")
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
```

Add `--action-mode` CLI argument:

```python
parser.add_argument(
    "--action-mode",
    type=str,
    default=None,
    choices=["cartesian_delta", "joint_position", "passthrough"],
    help="Action conversion mode (auto-detected from task if omitted)",
)
```

### Step 5: Create task configs in `configs/tasks/`

#### `configs/tasks/libero_spatial.yaml`
```yaml
task:
  name: libero
  max_episode_steps: 220
  success_threshold: 0.95
  params:
    task_suite_name: libero_spatial
    task_id: 0
    num_steps_wait: 10
    language_instruction: auto
    render_width: 224
    render_height: 224
```

#### `configs/tasks/libero_object.yaml`
```yaml
task:
  name: libero
  max_episode_steps: 280
  success_threshold: 0.95
  params:
    task_suite_name: libero_object
    task_id: 0
    num_steps_wait: 10
    language_instruction: auto
    render_width: 224
    render_height: 224
```

#### `configs/tasks/libero_goal.yaml`
```yaml
task:
  name: libero
  max_episode_steps: 300
  success_threshold: 0.95
  params:
    task_suite_name: libero_goal
    task_id: 0
    num_steps_wait: 10
    language_instruction: auto
    render_width: 224
    render_height: 224
```

#### `configs/tasks/libero_10.yaml`
```yaml
task:
  name: libero
  max_episode_steps: 520
  success_threshold: 0.95
  params:
    task_suite_name: libero_10
    task_id: 0
    num_steps_wait: 10
    language_instruction: auto
    render_width: 224
    render_height: 224
```

**Max steps rationale** (from openpi LIBERO example):
- `libero_spatial`: 220 (longest training demo = 193 steps)
- `libero_object`: 280 (longest training demo = 254 steps)
- `libero_goal`: 300 (longest training demo = 270 steps)
- `libero_10`: 520 (longest training demo = 505 steps)

### Step 6: Update `scripts/test_pi0.sbatch`

Add LIBERO-specific test levels after the existing ones:

```bash
# =============================================================================
# Level 3b: LIBERO single-episode evaluation
# =============================================================================
echo ""
echo "============================================"
echo "LEVEL 3b: LIBERO single-episode evaluation"
echo "============================================"

python -c "
from trace.config_loader import create_task, load_task_config
from trace.policy_adapter.pi0_adapter import Pi0PolicyAdapter
from trace.runner.episode_runner import EpisodeRunner

config = load_task_config('configs/tasks/libero_spatial.yaml')
config.seed = 0
task = create_task(config)
print(f'Task initialized: {config.name}')
print(f'Language instruction: {task.language_instruction}')

policy = Pi0PolicyAdapter(
    host='localhost', port=$PI0_PORT, chunk_size=5,
    action_mode='passthrough', state_format='axis_angle',
)
policy.set_env(task.get_mujoco_model(), task.get_mujoco_data())
policy.set_task_info(task.language_instruction)
policy.load('')

runner = EpisodeRunner(task, policy, stressors=[])
result = runner.run(seed=0)
print(f'Success:              {result.success}')
print(f'Steps:                {result.total_steps}')
print(f'Reward:               {result.total_reward:.3f}')
print(f'Catastrophic failure: {result.catastrophic_failure}')
print('LEVEL 3b PASSED')
" 2>&1
LEVEL3B_EXIT=$?

if [ $LEVEL3B_EXIT -ne 0 ]; then
    echo "LEVEL 3b FAILED (exit $LEVEL3B_EXIT) — aborting."
    exit 1
fi

# =============================================================================
# Level 4b: LIBERO mini sweep with visual stressors
# =============================================================================
echo ""
echo "============================================"
echo "LEVEL 4b: LIBERO mini sweep"
echo "============================================"

python -m scripts.run_evaluation \
  --task configs/tasks/libero_spatial.yaml \
  --sweep configs/sweeps/quick_test.yaml \
  --policy pi0 \
  --action-mode passthrough \
  --pi0-host localhost --pi0-port $PI0_PORT \
  --chunk-size 5 \
  --seed 0 \
  --output output/reports 2>&1
LEVEL4B_EXIT=$?

if [ $LEVEL4B_EXIT -ne 0 ]; then
    echo "LEVEL 4b FAILED (exit $LEVEL4B_EXIT) — aborting."
    exit 1
fi
echo "LEVEL 4b PASSED"
```

### Step 7: Write tests — `tests/test_libero_task.py`

```python
"""Tests for LiberoTask integration.

These tests require LIBERO to be installed. Skip gracefully if not available.
"""

import numpy as np
import pytest

# Skip all tests if LIBERO is not installed
libero = pytest.importorskip("libero")

from trace.task_spec.base import TaskConfig
from trace.task_spec.libero_task import LiberoTask


@pytest.fixture
def libero_config():
    return TaskConfig(
        name="libero",
        max_episode_steps=220,
        success_threshold=0.95,
        seed=0,
        task_params={
            "task_suite_name": "libero_spatial",
            "task_id": 0,
            "num_steps_wait": 10,
            "render_width": 224,
            "render_height": 224,
            "language_instruction": "auto",
        },
    )


@pytest.fixture
def libero_task(libero_config):
    task = LiberoTask(libero_config)
    task.initialize()
    yield task
    task.close()


class TestLiberoTaskInit:
    def test_initialize(self, libero_task):
        """Task initializes without error."""
        assert libero_task._env is not None
        assert libero_task._task_suite is not None

    def test_language_instruction(self, libero_task):
        """Language instruction is loaded from LIBERO benchmark."""
        assert len(libero_task.language_instruction) > 0
        assert libero_task.language_instruction != "auto"


class TestLiberoTaskReset:
    def test_reset_returns_observation(self, libero_task):
        obs = libero_task.reset(seed=0)
        assert isinstance(obs, dict)

    def test_observation_keys(self, libero_task):
        obs = libero_task.reset(seed=0)
        assert "image" in obs
        assert "wrist_image" in obs
        assert "ee_pos" in obs
        assert "ee_orientation" in obs
        assert "gripper" in obs

    def test_image_shape(self, libero_task):
        obs = libero_task.reset(seed=0)
        assert obs["image"].shape == (224, 224, 3)
        assert obs["image"].dtype == np.uint8
        assert obs["wrist_image"].shape == (224, 224, 3)

    def test_proprioception_shapes(self, libero_task):
        obs = libero_task.reset(seed=0)
        assert obs["ee_pos"].shape == (3,)
        assert obs["ee_orientation"].shape == (4,)
        assert obs["gripper"].shape == (2,)


class TestLiberoTaskStep:
    def test_step_returns_tuple(self, libero_task):
        libero_task.reset(seed=0)
        action = np.zeros(7, dtype=np.float32)
        obs, reward, done, info = libero_task.step(action)
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_observation_format(self, libero_task):
        libero_task.reset(seed=0)
        action = np.zeros(7, dtype=np.float32)
        obs, _, _, _ = libero_task.step(action)
        assert "image" in obs
        assert obs["image"].shape == (224, 224, 3)


class TestLiberoTaskMujoco:
    def test_get_mujoco_model(self, libero_task):
        libero_task.reset(seed=0)
        model = libero_task.get_mujoco_model()
        assert model is not None

    def test_get_mujoco_data(self, libero_task):
        libero_task.reset(seed=0)
        data = libero_task.get_mujoco_data()
        assert data is not None


class TestLiberoTaskSuccess:
    def test_initial_not_success(self, libero_task):
        libero_task.reset(seed=0)
        assert not libero_task.check_success()

    def test_no_catastrophic_failure(self, libero_task):
        libero_task.reset(seed=0)
        assert not libero_task.check_catastrophic_failure()
```

---

## 5. Files Summary

| File | Action | Description |
|------|--------|-------------|
| `trace/task_spec/libero_task.py` | **Create** | LiberoTask implementation wrapping LIBERO envs |
| `trace/policy_adapter/pi0_adapter.py` | **Modify** | Add passthrough action mode, axis-angle state format, `_quat2axisangle()` |
| `trace/config_loader.py` | **Modify** | Register LiberoTask in `TASK_REGISTRY` |
| `scripts/run_evaluation.py` | **Modify** | Add `--action-mode` flag, auto-detect LIBERO settings |
| `configs/tasks/libero_spatial.yaml` | **Create** | LIBERO spatial suite config (10 tasks) |
| `configs/tasks/libero_object.yaml` | **Create** | LIBERO object suite config (10 tasks) |
| `configs/tasks/libero_goal.yaml` | **Create** | LIBERO goal suite config (10 tasks) |
| `configs/tasks/libero_10.yaml` | **Create** | LIBERO-10 suite config (10 tasks) |
| `tests/test_libero_task.py` | **Create** | Unit tests for LiberoTask |
| `scripts/test_pi0.sbatch` | **Modify** | Add Level 3b/4b LIBERO test levels |

---

## 6. Stressor Compatibility

| Stressor | Works? | Notes |
|----------|--------|-------|
| LatencyStressor | **Yes** | Buffers actions, no env access needed |
| DropoutStressor | **Yes** | Modifies obs dict, works on any task |
| ImageNoiseStressor | **Yes** | Modifies images in obs |
| OcclusionStressor | **Yes** | Modifies images in obs |
| BrightnessShiftStressor | **Yes** | Modifies images in obs |
| ResolutionStressor | **Yes** | Modifies images in obs |
| LongHorizonDriftStressor | **Yes** | Adds noise to obs/actions |
| PhysicsShiftStressor | **Partial** | Needs `get_mujoco_model()` — depends on `env.sim.model._model` access |
| EmbodimentStressor | **Partial** | Same — needs MuJoCo model access |

**7 of 9 stressors work immediately.** Physics stressors depend on `env.sim.model._model` being a proper `mujoco.MjModel`, which robosuite 1.4.1 should expose.

---

## 7. Verification Steps

1. **Install deps:**
   ```bash
   conda activate trace-pi0
   pip install robosuite==1.4.1 bddl==1.0.1 "gym==0.25.2"
   pip install -e /scratch/at5282/trace/openpi/third_party/libero
   python -c "from libero.libero import benchmark; print('OK')"
   ```

2. **Unit tests:**
   ```bash
   cd /scratch/at5282/trace/tracerobotics
   python -m pytest tests/test_libero_task.py -v
   ```

3. **Single episode (needs GPU + pi0 server):**
   ```bash
   sbatch scripts/test_pi0.sbatch  # Run Level 3b
   ```
   Expected: non-zero success on LIBERO spatial task 0

4. **Mini sweep:**
   ```bash
   python -m scripts.run_evaluation \
     --task configs/tasks/libero_spatial.yaml \
     --sweep configs/sweeps/quick_test.yaml \
     --policy pi0 --action-mode passthrough
   ```
   Expected: baseline ~95%+ success, visible degradation under visual stressors

5. **Verify report:**
   Check `output/reports/` for meaningful degradation curves showing where pi0 breaks

---

## 8. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `gym==0.25.2` conflicts with other packages | Medium | Install in isolated trace-pi0 env |
| `env.sim.model._model` doesn't expose proper MuJoCo model | Low | Robosuite 1.4.1 uses `mujoco` bindings; fallback: skip physics stressors |
| LIBERO initial states have different episode counts per task | Low | Modulo index: `seed % len(initial_states)` |
| Image rotation: openpi flips images 180° for LIBERO | Medium | Pi0 adapter handles rotation in `_build_observation()`, not in LiberoTask |
| Action chunking + wait steps: off-by-one in step counting | Low | Wait steps happen in `reset()`, not counted in episode steps |

---

## 9. Future Extensions

- **Task sweep:** Loop over all 10 task_ids in a suite for comprehensive evaluation
- **libero_90:** Support the full 90-task suite (needs longer eval time)
- **Multi-task report:** Aggregate results across tasks within a suite
- **Video recording:** Save episode replays like openpi's example does
