# LIBERO Task Integration

**Date:** 2026-03-04
**Status:** Implemented, unit tests passing, sbatch submitted (job 14300051)

---

## What We Built

Wrapped LIBERO's 130 tabletop manipulation environments into Trace's `BaseTask` interface so we can stress-test pi0.5-LIBERO on tasks it was actually trained on (instead of our custom reach task where it scores 0%).

---

## Files Created

| File | Description |
|------|-------------|
| `trace/task_spec/libero_task.py` | `LiberoTask(BaseTask)` — wraps LIBERO `OffScreenRenderEnv` into Trace. Defers LIBERO imports, maps obs keys, resizes images 256→224, exposes MuJoCo model via `env.sim.model._model` |
| `configs/tasks/libero_spatial.yaml` | Spatial suite, 220 max steps |
| `configs/tasks/libero_object.yaml` | Object suite, 280 max steps |
| `configs/tasks/libero_goal.yaml` | Goal suite, 300 max steps |
| `configs/tasks/libero_10.yaml` | LIBERO-10 suite, 520 max steps |
| `tests/test_libero_task.py` | 12 unit tests — init, reset, step, obs format, MuJoCo access, success detection |
| `scripts/test_libero_pi0.sbatch` | Dedicated LIBERO+pi0 sbatch pipeline (4 levels) |

## Files Modified

| File | Changes |
|------|---------|
| `trace/policy_adapter/pi0_adapter.py` | Added `passthrough` action mode (skips Jacobian), `axis_angle` state format, `_quat2axisangle()` helper, `state_format` constructor param. `set_env()` skips site lookup in passthrough mode |
| `trace/config_loader.py` | Registered `"libero": LiberoTask` in `TASK_REGISTRY` |
| `scripts/run_evaluation.py` | Added `--action-mode` CLI flag. Auto-detects passthrough + axis-angle when task is `libero` |
| `scripts/test_pi0.sbatch` | Removed LIBERO levels (moved to dedicated `test_libero_pi0.sbatch`) |

---

## Key Design Decisions

**Action passthrough:** LIBERO expects raw 7-dim `[arm_delta(6), gripper(1)]` directly in `env.step()`. The existing Pi0 adapter did Jacobian transpose conversion (Cartesian→joint), which is wrong for LIBERO. The `passthrough` mode bypasses this entirely.

**State format:** Pi0-LIBERO expects `[ee_pos(3), axis_angle(3), gripper_qpos(2)]` = 8 dims. Our adapter previously sent `[ee_pos(3), quat(4), gripper(1)]` = 8 dims. The `axis_angle` state format handles this with `_quat2axisangle()` copied from robosuite.

**Observation mapping:**
```
LIBERO key                    → Trace key        → Shape
agentview_image               → image            → (224, 224, 3) uint8
robot0_eye_in_hand_image      → wrist_image      → (224, 224, 3) uint8
robot0_eef_pos                → ee_pos           → (3,) float32
robot0_eef_quat               → ee_orientation   → (4,) float32
robot0_gripper_qpos           → gripper          → (2,) float32
```

**Deferred imports:** LIBERO/robosuite imports happen in `initialize()`, not at module level. This lets the rest of Trace work on nodes without LIBERO installed.

**Wait steps in reset:** LIBERO drops objects at episode start and needs ~10 dummy steps for them to settle. These happen inside `reset()` and are not counted as episode steps.

---

## Dependency Setup

The `--no-deps` flag was required because `robosuite` pulls in `evdev` which needs `gcc` (unavailable on HPC compute nodes).

```bash
conda activate trace-pi0

# Core packages (no-deps to avoid evdev build failure)
pip install robosuite==1.4.1 bddl==1.0.1 "gym==0.25.2" --no-deps

# Runtime deps that robosuite/bddl/gym/libero actually need
pip install future easydict h5py cloudpickle gym-notices pygame opencv-python numba scipy termcolor networkx

# PyTorch (needed by LIBERO benchmark for torch.load of init states)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# LIBERO itself (editable from openpi third_party)
pip install -e /scratch/at5282/trace/openpi/third_party/libero --no-deps
```

### Fixes Required During Setup

1. **Missing `__init__.py`:** `openpi/third_party/libero/libero/__init__.py` did not exist. Without it, `setuptools.find_packages()` returned nothing and the pip wheel was empty (5KB). Created the file → wheel became 132KB.

2. **LIBERO config prompt:** First import triggers an interactive prompt for dataset path. Created `~/.libero/config.yaml` manually:
   ```yaml
   benchmark_root: /scratch/at5282/trace/openpi/third_party/libero/libero/libero
   bddl_files: /scratch/at5282/trace/openpi/third_party/libero/libero/libero/bddl_files
   init_states: /scratch/at5282/trace/openpi/third_party/libero/libero/libero/init_files
   datasets: /scratch/at5282/trace/openpi/third_party/libero/libero/datasets
   assets: /scratch/at5282/trace/openpi/third_party/libero/libero/libero/assets
   ```

3. **PyTorch `weights_only` error:** PyTorch 2.10 defaults `torch.load(weights_only=True)` which rejects numpy arrays in LIBERO's init state files. Patched `openpi/third_party/libero/libero/libero/benchmark/__init__.py` line 164:
   ```python
   # Before:
   init_states = torch.load(init_states_path)
   # After:
   init_states = torch.load(init_states_path, weights_only=False)
   ```

---

## Test Results

**Unit tests:** 12/12 passed (45s on login node, no GPU needed)

```
tests/test_libero_task.py::TestLiberoTaskInit::test_initialize PASSED
tests/test_libero_task.py::TestLiberoTaskInit::test_language_instruction PASSED
tests/test_libero_task.py::TestLiberoTaskReset::test_reset_returns_observation PASSED
tests/test_libero_task.py::TestLiberoTaskReset::test_observation_keys PASSED
tests/test_libero_task.py::TestLiberoTaskReset::test_image_shape PASSED
tests/test_libero_task.py::TestLiberoTaskReset::test_proprioception_shapes PASSED
tests/test_libero_task.py::TestLiberoTaskStep::test_step_returns_tuple PASSED
tests/test_libero_task.py::TestLiberoTaskStep::test_step_observation_format PASSED
tests/test_libero_task.py::TestLiberoTaskMujoco::test_get_mujoco_model PASSED
tests/test_libero_task.py::TestLiberoTaskMujoco::test_get_mujoco_data PASSED
tests/test_libero_task.py::TestLiberoTaskSuccess::test_initial_not_success PASSED
tests/test_libero_task.py::TestLiberoTaskSuccess::test_no_catastrophic_failure PASSED
```

**Sbatch job:** Submitted as job 14300051 via `scripts/test_libero_pi0.sbatch`. Awaiting GPU results.

---

## Sbatch Pipeline (`test_libero_pi0.sbatch`)

| Level | What | Needs Server |
|-------|------|:---:|
| 1 | `pytest tests/test_libero_task.py` — unit tests | No |
| 2 | Single LIBERO episode with pi0 — verify end-to-end | Yes |
| 3 | Mini sweep (`quick_test.yaml`) — latency stressor, 3 intensities | Yes |
| 4 | Full robustness sweep (`default_sweep.yaml`) — all stressors | Yes |

Monitor: `tail -f output/logs/libero-pi0-14300051.out`

---

## Stressor Compatibility

| Stressor | Works? | Notes |
|----------|:------:|-------|
| LatencyStressor | Yes | Buffers actions, no env access needed |
| DropoutStressor | Yes | Modifies obs dict |
| ImageNoiseStressor | Yes | Modifies images in obs |
| OcclusionStressor | Yes | Modifies images in obs |
| BrightnessShiftStressor | Yes | Modifies images in obs |
| ResolutionStressor | Yes | Modifies images in obs |
| LongHorizonDriftStressor | Yes | Adds noise to obs/actions |
| PhysicsShiftStressor | Partial | Needs `get_mujoco_model()` — depends on `env.sim.model._model` |
| EmbodimentStressor | Partial | Same — needs MuJoCo model access |

7 of 9 work immediately.

---

## Usage

```bash
# Single evaluation
python -m scripts.run_evaluation \
  --task configs/tasks/libero_spatial.yaml \
  --sweep configs/sweeps/quick_test.yaml \
  --policy pi0 \
  --pi0-host localhost --pi0-port 8000 \
  --output output/reports

# Sbatch (starts server + runs all levels)
sbatch scripts/test_libero_pi0.sbatch
```

The `--action-mode` flag is auto-detected as `passthrough` for LIBERO tasks but can be overridden manually.

---

## Git History

```
3e6f79f Add LIBERO task integration development plan
ad91f3a Add LIBERO task configs, unit tests, and sbatch test levels 3b/4b
7b584a7 Register LiberoTask in config loader and add --action-mode to CLI
e58ccac Add passthrough action mode and axis-angle state format to Pi0 adapter
1627d1a Add LiberoTask wrapping LIBERO environments into BaseTask interface
```

---

## Future Work

- Sweep over all 10 `task_id`s within a suite for comprehensive evaluation
- Support `libero_90` (90-task suite, needs longer eval time)
- Multi-task aggregated reports across a suite
- Video recording of episodes (like openpi's example)
