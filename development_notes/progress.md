# Trace Robotics — Development Progress

**Last updated:** 2026-03-04

---

## Project Summary

Trace Robotics is a stress-testing platform for robot foundation policies. Input: a policy checkpoint. Output: a robustness report showing where the policy degrades under stress (latency, sensor dropout, physics perturbations, visual noise, etc.).

**Current milestone:** Pi0.5-LIBERO running end-to-end on actual LIBERO manipulation tasks with stress testing. First successful evaluation on A100 GPU — pi0 solves tasks at baseline, degrades under latency stress.

---

## Phase 1: Core Framework [COMPLETE — Feb 2026]

Built the entire evaluation infrastructure from scratch.

**What was built:**
- `BasePolicy`, `BaseTask`, `BaseStressor` — abstract interfaces
- `EpisodeRunner` — runs one episode with stressor chaining (obs → perturb → policy → perturb → step)
- `SweepRunner` — grid over intensity × seed × episode
- `SweepAggregator` — statistical aggregation with breakpoint detection
- `ReportGenerator` — markdown robustness reports
- `config_loader.py` — YAML parsing with task/stressor registries
- CLI: `python -m scripts.run_evaluation --task ... --sweep ... --policy ...`

**Tasks:** `ReachTask` — 7-DOF Panda arm in MuJoCo, randomized targets, inline MJCF (no external assets)

**Policies:** `RandomPolicy` (baseline), `ScriptedReachPolicy` (Jacobian-transpose proportional controller)

**5 Stressors:** Latency, Dropout, PhysicsShift, Embodiment, LongHorizonDrift

**Validation:** 1,750 episodes with ScriptedReachPolicy — pipeline works end-to-end, first robustness report generated.

| Stressor | Baseline | Max Intensity | Breakpoint | Verdict |
|----------|----------|---------------|------------|---------|
| Latency | 100% | 66% | none | Robust |
| Dropout | 100% | 0% | 0.50 | Very fragile |
| PhysicsShift | 100% | 98% | none | Robust |
| Embodiment | 100% | 28% | 0.50 | Fragile |
| LongHorizonDrift | 100% | 38% | 1.00 | Fragile |

**Tests:** 60+ passing

---

## Phase 2: Camera & Visual Observations [COMPLETE — Mar 2 2026]

Added image-based observations to support vision-language-action models.

- Added third-person camera (diagonal overhead) and wrist camera (eye-in-hand) to Panda MJCF
- Configurable offscreen rendering via `render_cameras` dict — opt-in, zero overhead when disabled
- `Observation = dict[str, np.ndarray]` type alias across 13 files
- Updated DropoutStressor and LongHorizonDriftStressor to handle uint8 image arrays
- Added `close()` for renderer cleanup

**Tests:** 69 passing (+9 camera rendering tests)

---

## Phase 3: Language Prompt Support [COMPLETE — Mar 2 2026]

- Added `language_instruction` property to `BaseTask` (reads from `task_params`)
- Design decision: prompt is task metadata, not sensor data — keeps `Observation` type clean

**Tests:** 72 passing (+3)

---

## Phase 4: Pi0 Policy Adapter [COMPLETE — Mar 3 2026]

Connected to Physical Intelligence's pi0.5 model via openpi WebSocket server.

- `Pi0PolicyAdapter` — WebSocket client to openpi inference server
- LIBERO observation mapping: 180° image rotation, 8-dim state vector
- Action chunk buffering (5 actions/call, configurable via `chunk_size`)
- Jacobian-transpose Cartesian→joint conversion for ReachTask
- Graceful fallback to zero actions when openpi_client unavailable
- `set_env()` / `set_task_info()` for MuJoCo model and language instruction
- CLI args: `--pi0-host`, `--pi0-port`, `--chunk-size`
- `reach_pi0.yaml` task config with cameras at 224×224

**Tests:** 108 passing (+36: 14 adapter + 22 visual stressors)

---

## Phase 5: Visual Stressors [COMPLETE — Mar 3 2026]

4 image-based stressors in `trace/stressor_engine/visual.py`:

| Stressor | What It Does |
|----------|-------------|
| ImageNoiseStressor | Gaussian noise on camera images |
| OcclusionStressor | Random rectangular patches over image regions |
| BrightnessShiftStressor | Exposure/contrast perturbation |
| ResolutionStressor | Downscale + upscale to simulate low-res cameras |

All operate on `image`/`wrist_image` obs keys, zero-intensity passthrough, deterministic with seed. Registered in STRESSOR_REGISTRY and default sweep config.

---

## Phase 6: HPC Setup & Pi0 Server [COMPLETE — Mar 3-4 2026]

Got pi0 running on NYU HPC (Jubail cluster).

- `trace-pi0` conda env (Python 3.11) with openpi, JAX, PyTorch
- openpi server serving `pi05_libero` checkpoint on A100 GPU (~31 GB VRAM)
- First inference triggers JAX JIT warmup (~60-120s), subsequent calls ~73ms
- Fixed WebSocket `ping_timeout` (20s default → 300s) to survive JIT warmup
- `scripts/test_pi0.sbatch` — 5-level incremental test pipeline (unit tests → smoke test → single episode → mini sweep → full sweep)
- SSL cert fix for GCS checkpoint download on HPC nodes

See `development_notes/testingpi0.md` for detailed HPC setup guide.

---

## Phase 7: LIBERO Task Integration [COMPLETE — Mar 4 2026]

Pi0 scored 0% on our ReachTask because it was never trained on "reach the target". Integrated LIBERO's 130 manipulation tasks so we can evaluate pi0 on tasks it actually understands.

### What was built

**`trace/task_spec/libero_task.py`** — `LiberoTask(BaseTask)`:
- Wraps LIBERO's `OffScreenRenderEnv` into Trace's interface
- Supports all 5 suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`
- Deferred imports (LIBERO not needed on every node)
- Obs mapping: `agentview_image` → `image`, `robot0_eef_pos` → `ee_pos`, etc.
- Image resize 256→224, wait steps in `reset()` for object settling
- MuJoCo model access via `env.sim.model._model` for physics stressors

**Pi0 adapter changes** (`pi0_adapter.py`):
- `passthrough` action mode — raw 7-dim actions bypass Jacobian conversion
- `axis_angle` state format — `[ee_pos(3), axis_angle(3), gripper_qpos(2)]` instead of quaternion
- `_quat2axisangle()` helper from robosuite
- `set_env()` skips site lookup in passthrough mode

**Config & CLI:**
- `"libero": LiberoTask` registered in `TASK_REGISTRY`
- `--action-mode` CLI flag (auto-detects `passthrough` for LIBERO)
- 4 task configs: `libero_spatial.yaml` (220 steps), `libero_object.yaml` (280), `libero_goal.yaml` (300), `libero_10.yaml` (520)

**`scripts/test_libero_pi0.sbatch`** — dedicated LIBERO + pi0 pipeline:
1. Unit tests (no GPU)
2. Single LIBERO episode with pi0
3. Mini sweep (quick_test.yaml)
4. Full robustness sweep (default_sweep.yaml)

### Dependency setup issues & fixes

1. **Missing `__init__.py`** in `openpi/third_party/libero/libero/` — caused empty pip wheel (5KB → 132KB after fix)
2. **LIBERO config prompt** — first import asks for dataset path interactively. Created `~/.libero/config.yaml` manually pointing to vendored data
3. **PyTorch `weights_only`** — PyTorch 2.10 rejects numpy arrays in `torch.load()`. Patched LIBERO's `benchmark/__init__.py` to use `weights_only=False`
4. **WebSocket ping timeout** — first pi0 inference takes 60-120s for JAX JIT. Patched `openpi_client/websocket_client_policy.py` to use `ping_timeout=300`
5. **Transitive deps** — installed with `--no-deps` (evdev needs gcc, unavailable on HPC), then manually installed: `future`, `easydict`, `h5py`, `cloudpickle`, `gym-notices`, `pygame`, `opencv-python`, `numba`, `scipy`, `termcolor`, `networkx`

### First LIBERO evaluation results

**Job 14300072** on A100 80GB (cn257):

- 12/12 unit tests passed
- Single episode: **SUCCESS** in 69 steps, reward 1.0
- Mini sweep (LatencyStressor, 3 intensities):
  - intensity=0.00: 2/2 success (73, 106 steps)
  - intensity=0.50: 0/2 success (both hit 220 step limit)
  - intensity=1.00: awaiting results

Pi0 solves LIBERO spatial tasks at baseline (~95%+) and degrades meaningfully under latency stress — exactly the degradation curves we wanted.

**Tests:** 12 LIBERO-specific tests passing (+ 108 existing = 120 total)

---

## Stressor Compatibility with LIBERO

| Stressor | Works? | Notes |
|----------|:------:|-------|
| LatencyStressor | Yes | Verified — causes degradation |
| DropoutStressor | Yes | Modifies obs dict |
| ImageNoiseStressor | Yes | Modifies images |
| OcclusionStressor | Yes | Modifies images |
| BrightnessShiftStressor | Yes | Modifies images |
| ResolutionStressor | Yes | Modifies images |
| LongHorizonDriftStressor | Yes | Adds noise to obs/actions |
| PhysicsShiftStressor | Partial | Needs `env.sim.model._model` |
| EmbodimentStressor | Partial | Same |

7 of 9 stressors work immediately. Physics stressors need verification that robosuite exposes proper `mujoco.MjModel`.

---

## Files Reference

```
trace/
  policy_adapter/
    base.py              — BasePolicy ABC, PolicyMetadata
    random_policy.py     — RandomPolicy (baseline)
    scripted_reach.py    — ScriptedReachPolicy (Jacobian controller)
    pi0_adapter.py       — Pi0PolicyAdapter (WebSocket, passthrough, axis-angle)
  task_spec/
    base.py              — BaseTask ABC, TaskConfig, EpisodeResult
    reach.py             — ReachTask (7-DOF Panda, inline MJCF)
    libero_task.py       — LiberoTask (wraps LIBERO envs)
    mjcf_models.py       — Inline MJCF XML
  stressor_engine/
    base.py              — BaseStressor ABC
    latency.py           — LatencyStressor
    dropout.py           — DropoutStressor
    physics_shift.py     — PhysicsShiftStressor
    embodiment.py        — EmbodimentStressor
    long_horizon.py      — LongHorizonDriftStressor
    visual.py            — ImageNoise, Occlusion, BrightnessShift, Resolution
  runner/
    episode_runner.py    — EpisodeRunner
    sweep_runner.py      — SweepRunner + SweepConfig
  metrics/
    collectors.py        — StepMetrics, EpisodeMetrics
    aggregator.py        — SweepAggregator, IntensityStats, SweepResult
  report/
    generator.py         — ReportGenerator (markdown)
  config_loader.py       — YAML parsing, task/stressor registries
configs/
  tasks/
    reach.yaml           — ReachTask (proprioception only)
    reach_pi0.yaml       — ReachTask with cameras at 224x224
    libero_spatial.yaml  — LIBERO spatial suite (220 steps)
    libero_object.yaml   — LIBERO object suite (280 steps)
    libero_goal.yaml     — LIBERO goal suite (300 steps)
    libero_10.yaml       — LIBERO-10 suite (520 steps)
  sweeps/
    default_sweep.yaml   — Full sweep (9 stressors)
    quick_test.yaml      — Mini sweep (1 stressor, 3 intensities)
scripts/
  run_evaluation.py      — CLI entry point
  test_pi0.sbatch        — Reach + pi0 sbatch pipeline
  test_libero_pi0.sbatch — LIBERO + pi0 sbatch pipeline
tests/                   — 120 tests
development_notes/
  progress.md            — This file
  1-pi0_evaluation_progress.md  — Detailed phase-by-phase notes
  Liberotask.md          — LIBERO integration details
  testingpi0.md          — HPC setup guide
```

---

## Git History

```
b2c50ee Add dedicated LIBERO + pi0 sbatch pipeline
9bd1c38 Move LIBERO plan to development notes with implementation details
3e6f79f Add LIBERO task integration development plan
ad91f3a Add LIBERO task configs, unit tests, and sbatch test levels 3b/4b
7b584a7 Register LiberoTask in config loader and add --action-mode to CLI
e58ccac Add passthrough action mode and axis-angle state format to Pi0 adapter
1627d1a Add LiberoTask wrapping LIBERO environments into BaseTask interface
9dfe933 basics checks done
154d203 evaluation setup on hpc for pi0 policy
00bf1e6 updated the testing process from hpc
a323020 Add Pi0 HPC testing guide with setup and evaluation steps
21ea3d3 Update development notes: Phases 4 & 5 complete
cf1ce26 Add pi0 policy to evaluation CLI with server connection args
6572cb6 Add Pi0 policy adapter with LIBERO observation mapping
129b18f Add visual stressors: image noise, occlusion, brightness shift, resolution
2123432 Phase 3: Add language instruction support to BaseTask
fa265e9 Update development notes: Phase 2 complete
890ebb1 Add camera rendering tests and update config/fixtures
eb0b0d8 Handle uint8 image observations in dropout and drift stressors
2784255 Add configurable camera rendering to ReachTask
28abd10 Add third-person and wrist cameras to Panda MJCF model
6067de6 Add Observation type alias, broaden type signatures for image support
d39b79d added development notes, for pi0 adaptation
4633e3f evaluated reach policy again
d3f7343 claude fixed issues with reach policy, mapping, jacobian and action as delta
f2f4256 evaluated scripted Reach policy
d914ccf ran one complete evaluation
392a32e Update .gitignore to remove ignored report directories
42b006f adding one task for validation
176b708 added testing documentation and scripts
cc7ff07 initial claude development based on plan
3b69cc7 added plan
2f19910 Revise README for Trace Robotics overview and details
e85d343 first commit
```

---

## What's Next

- **Analyze full sweep results** — job 14300072 running on A100, check `output/reports/`
- **Run all 9 stressors** on LIBERO spatial — get full degradation curves
- **Sweep across task_ids** — evaluate all 10 tasks in libero_spatial
- **Try other suites** — libero_object, libero_goal, libero_10
- **Verify physics stressors** — test PhysicsShiftStressor and EmbodimentStressor with robosuite's MuJoCo model
- **Video recording** — save episode replays for qualitative analysis
- **Multi-task reports** — aggregate results across tasks within a suite
