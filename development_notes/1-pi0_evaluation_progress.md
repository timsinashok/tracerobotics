# Pi0 Evaluation Progress

**Goal:** Evaluate Physical Intelligence's pi0 policy using the Trace Robotics stress-testing framework.
**Status:** Phase 1 complete, Phase 2 complete, Phase 3 next (language prompts)
**Last updated:** 2026-03-02

---

## Project Overview

Trace Robotics is a stress-testing platform for robot foundation policies.
Input: a policy checkpoint. Output: a robustness report showing failure modes under stress.

We are working toward running pi0 (Physical Intelligence's 3B-parameter Vision-Language-Action model)
through our full stressor suite and generating a robustness report.

---

## Current State

### What's Built and Working

**Core Framework (complete)**
- `BasePolicy`, `BaseTask`, `BaseStressor` — clean ABC interfaces
- Episode runner with stressor chaining: obs -> perturb -> policy -> perturb -> step
- Sweep runner: intensity x seed x episode grid, with statistical aggregation
- YAML config loader with task/stressor registries
- Markdown report generator with per-stressor tables and breakpoint detection
- CLI: `python -m scripts.run_evaluation --task ... --sweep ... --policy ...`
- 60+ deterministic tests passing

**5 Stressors (all parameterized, sweepable, seeded)**

| Stressor             | What It Does                                      | Status  |
|----------------------|---------------------------------------------------|---------|
| LatencyStressor      | Buffers actions to simulate communication delay    | Working |
| DropoutStressor      | Corrupts observations (zero/noise/freeze modes)    | Working |
| PhysicsShiftStressor | Perturbs mass, friction, damping                   | Working |
| EmbodimentStressor   | Perturbs link geometry, joint limits, actuator gains| Working |
| LongHorizonDriftStressor | Growing observation + action noise over time   | Working |

**1 Task: ReachTask**
- 7-DOF Panda arm in MuJoCo (inline MJCF, self-contained, no external assets)
- Randomized targets in configurable workspace bounds
- Position control, 20Hz control rate (25 MuJoCo substeps at 0.002s)
- Physics param caching/restoration across resets (critical for stressor correctness)
- Configurable camera rendering: third-person + wrist cameras, opt-in via `render_cameras`
- Images are uint8 (H, W, 3), rendered via `mujoco.Renderer`

**2 Test Policies**
- `RandomPolicy` — uniform random baseline
- `ScriptedReachPolicy` — Jacobian-transpose proportional controller

### Validated Evaluation Run

ScriptedReachPolicy on ReachTask, 1,750 episodes (5 stressors x 7 intensities x 5 seeds x 10 episodes):

| Stressor             | Baseline | Max Intensity | Breakpoint | Verdict       |
|----------------------|----------|---------------|------------|---------------|
| LatencyStressor      | 100%     | 66%           | none       | Robust        |
| DropoutStressor      | 100%     | 0%            | 0.50       | Very fragile  |
| PhysicsShiftStressor | 100%     | 98%           | none       | Robust        |
| EmbodimentStressor   | 100%     | 28%           | 0.50       | Fragile       |
| LongHorizonDriftStressor | 100% | 38%           | 1.00       | Fragile       |

Full report: `output/reports/report_ScriptedReachPolicy_reach_20260226_014716.md`

This validates that the pipeline works end-to-end: task -> sweep -> aggregation -> report.

---

## What Pi0 Needs (Gap Analysis)

Pi0 is a Vision-Language-Action flow model. It expects inputs and produces outputs that differ
significantly from our current proprioception-only setup.

### Pi0 Input/Output Format

**Inputs:**
- 1-2 RGB images (224x224, uint8) — third-person camera + optional wrist camera
- Proprioceptive state — joint positions (7) + gripper position (1)
- Language prompt — natural language task instruction (e.g., "reach the red target")

**Outputs:**
- Action chunk of shape (50, action_dim) — 50 future actions at once
- Convention: execute 16-25 actions from the chunk, then re-query the model
- Action space: typically Cartesian (3D position delta + 3D rotation delta + gripper)

### Gaps Between Current Framework and Pi0

| Gap | Current State | Pi0 Needs | Severity | Status |
|-----|---------------|-----------|----------|--------|
| Image observations | Configurable camera rendering (third-person + wrist) | RGB images (224x224) | Critical | **DONE** |
| Language prompts | No language support in obs dict | Task instruction string | Critical | Phase 3 |
| Action chunking | 1 action per step | 50-action chunks, execute N | Critical | Phase 4 |
| Inference backend | Pure numpy policies | JAX/PyTorch model or WebSocket server | Critical | Phase 4 |
| Observation types | `Observation = dict[str, np.ndarray]` (uint8 + float32) | Needs uint8 images + strings | Moderate | **DONE** |
| Action space | Joint-space position [-1, 1] | Cartesian end-effector deltas | Moderate | Phase 4 |
| Visual stressors | Dropout/drift handle uint8; no image-specific stressors yet | Need image noise, occlusion, etc. | Important | Phase 5 |

### Pi0 Open-Source Status

Pi0 is fully open-source via [github.com/Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi):
- Model weights on GCS and HuggingFace
- JAX and PyTorch implementations
- Remote inference server via WebSocket (recommended for us)
- `pi05_libero` checkpoint — already trained on MuJoCo simulation (LIBERO benchmark)
- Fine-tuning with LoRA (22.5 GB VRAM) or full (70+ GB)

---

## Development Plan — Incremental TODOs

### Phase 1: Framework Foundation [COMPLETE]
- [x] Design and implement BasePolicy, BaseTask, BaseStressor ABCs
- [x] Implement ReachTask with inline MJCF 7-DOF Panda arm
- [x] Implement 5 stressors (latency, dropout, physics_shift, embodiment, long_horizon)
- [x] Build EpisodeRunner with stressor chaining
- [x] Build SweepRunner with intensity x seed x episode grid
- [x] Build SweepAggregator with breakpoint detection
- [x] Build ReportGenerator (markdown output)
- [x] Build config_loader with YAML parsing and registries
- [x] Implement RandomPolicy baseline
- [x] Implement ScriptedReachPolicy (Jacobian-transpose controller)
- [x] Wire CLI entry point (scripts/run_evaluation.py)
- [x] Write test suite (60+ tests)
- [x] Run and validate full evaluation on ScriptedReachPolicy

### Phase 2: Camera and Visual Observations [COMPLETE]
- [x] Add `Observation = dict[str, np.ndarray]` type alias across all interfaces (13 files)
- [x] Add third-person camera to ReachTask MJCF XML (diagonal overhead view)
- [x] Add wrist-mount camera to ReachTask MJCF XML (eye-in-hand on link7)
- [x] Add overhead light for camera rendering
- [x] Add configurable offscreen rendering to ReachTask via `render_cameras` dict
- [x] Rendering is opt-in: empty dict = no cameras, zero overhead
- [x] `render_cameras` maps obs keys to MJCF camera names (e.g., `{"image": "third_person"}`)
- [x] Add `render_width` / `render_height` as configurable task params (default 128)
- [x] Update DropoutStressor to handle uint8 images (integer noise in 0-255 range)
- [x] Update LongHorizonDriftStressor to skip drift noise for uint8 arrays
- [x] Add `close()` method to ReachTask for renderer cleanup
- [x] Write 9 camera rendering tests (shape, dtype, value range, step rendering, fallback)
- [x] 69 tests passing (60 original + 9 new)

### Phase 3: Language Prompt Support
- [ ] Add `language_instruction` field to TaskConfig
- [ ] Include prompt in observation dict from tasks (or handle in adapter)
- [ ] Update ReachTask config with a default instruction (e.g., "reach the target")
- [ ] Decide: prompt lives in task vs. adapter (adapter is simpler for v1)

### Phase 4: Pi0 Policy Adapter
- [ ] Add `openpi-client` as a project dependency
- [ ] Write Pi0PolicyAdapter class:
  - [ ] WebSocket client connection to openpi inference server
  - [ ] Observation key remapping (Trace keys -> openpi keys)
  - [ ] Action chunk buffering (receive 50, return 1 per act() call)
  - [ ] Configurable chunk execution length (default 16)
  - [ ] Reset clears action buffer
- [ ] Handle action space conversion:
  - [ ] Determine if pi0 checkpoint outputs joint-space or Cartesian actions
  - [ ] If Cartesian: implement IK or operational-space conversion in adapter
  - [ ] If joint-space (LIBERO checkpoint): direct pass-through
- [ ] Register Pi0PolicyAdapter in POLICY_REGISTRY
- [ ] Write adapter unit tests with mock WebSocket server
- [ ] Document GPU server setup (openpi serve_policy.py)

### Phase 5: Visual Stressors
- [ ] ImageNoiseStressor — Gaussian noise, salt-and-pepper on camera images
- [ ] OcclusionStressor — random rectangular patches over image regions
- [ ] BrightnessShiftStressor — exposure/contrast perturbation
- [ ] ResolutionStressor — downscale and upscale to simulate low-res cameras
- [ ] Update stressor base to handle image observations in perturb_observation()
- [ ] Write tests for each visual stressor

### Phase 6: Run Pi0 Evaluation
- [ ] Set up GPU inference server with openpi + pi0 checkpoint
- [ ] Choose checkpoint: pi05_libero (MuJoCo-trained) or fine-tuned
- [ ] Run baseline evaluation (no stressors) to verify pi0 works in our pipeline
- [ ] Run full stressor sweep (proprioceptive + visual stressors)
- [ ] Generate robustness report
- [ ] Analyze results — identify pi0's failure modes and breakpoints

### Phase 7: Richer Tasks (stretch)
- [ ] Implement ReachAndGraspTask (reach + gripper close)
- [ ] Implement PickAndPlaceTask (pick object, move to target)
- [ ] Implement DrawerOpenTask (articulated object manipulation)
- [ ] Add MJCF models with objects, drawers, etc.
- [ ] Update configs and registries
- [ ] Re-run pi0 evaluation on harder tasks

---

## Architecture Notes

### Recommended Pi0 Adapter Design

```python
class Pi0PolicyAdapter(BasePolicy):
    """Adapter for Physical Intelligence pi0 via remote inference."""

    def __init__(self, host: str, port: int, prompt: str, chunk_size: int = 16):
        self._host = host
        self._port = port
        self._prompt = prompt
        self._chunk_size = chunk_size
        self._action_buffer: list[NDArray] = []
        self._client = None

    def load(self, checkpoint_path: str) -> None:
        # checkpoint_path unused for remote inference
        # Connection happens here
        from openpi_client import WebsocketClientPolicy
        self._client = WebsocketClientPolicy(self._host, self._port)

    def reset(self) -> None:
        self._action_buffer = []

    def act(self, observation: dict[str, Any]) -> NDArray:
        if not self._action_buffer:
            pi0_obs = {
                "observation/image": observation["image"],            # (224,224,3) uint8
                "observation/wrist_image": observation.get("wrist_image", np.zeros((224,224,3), dtype=np.uint8)),
                "observation/state": observation["joint_pos"],        # (7,) or (8,)
                "prompt": self._prompt,
            }
            result = self._client.infer(pi0_obs)
            chunk = result["actions"]                                 # (50, action_dim)
            self._action_buffer = list(chunk[:self._chunk_size])

        return self._action_buffer.pop(0)

    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            name="Pi0Policy",
            observation_space={"image": (224, 224, 3), "joint_pos": (7,)},
            action_space=(7,),
            modalities=["vision", "proprioception", "language"],
        )
```

### Key Design Decisions

1. **Remote inference (WebSocket)** — Decouples GPU from evaluation host. Run openpi server
   separately. Simplest path for v1.
2. **Action chunking handled in adapter** — The runner doesn't change. The adapter buffers
   internally and returns 1 action per act() call.
3. **Observation remapping in adapter** — Tasks produce Trace-format obs, adapter translates
   to openpi format. Clean separation.
4. **Start with pi05_libero checkpoint** — Already trained on MuJoCo simulation (LIBERO).
   Easiest integration path. Can fine-tune later.

---

## Key Risks

| Risk | Mitigation |
|------|------------|
| GPU availability (22+ GB VRAM needed) | Use remote inference; can run server on cloud GPU |
| Action space mismatch (Cartesian vs joint) | LIBERO checkpoint likely uses joint-space; verify first |
| pi0 may not generalize to our ReachTask | Start with LIBERO tasks that pi0 was trained on, then test transfer |
| Inference latency (~73ms/chunk) | Action chunking amortizes cost; 16 steps per inference call |
| Image rendering slows simulation | Profile early; offscreen rendering is typically fast in MuJoCo |

---

## Files Reference

```
trace/
  policy_adapter/
    base.py              — BasePolicy ABC, PolicyMetadata
    random_policy.py     — RandomPolicy (baseline)
    scripted_reach.py    — ScriptedReachPolicy (Jacobian controller)
    [pi0_adapter.py]     — TODO: Pi0PolicyAdapter
  task_spec/
    base.py              — BaseTask ABC, TaskConfig, EpisodeResult
    reach.py             — ReachTask (7-DOF Panda)
    mjcf_models.py       — Inline MJCF XML
  stressor_engine/
    base.py              — BaseStressor ABC, StressorConfig
    latency.py           — LatencyStressor
    dropout.py           — DropoutStressor
    physics_shift.py     — PhysicsShiftStressor
    embodiment.py        — EmbodimentStressor
    long_horizon.py      — LongHorizonDriftStressor
    [image_noise.py]     — TODO: visual stressors
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
  tasks/reach.yaml       — ReachTask config (working)
  sweeps/default_sweep.yaml — Default sweep config (working)
scripts/
  run_evaluation.py      — CLI entry point
tests/                   — 69 tests
development_notes/
  1-pi0_evaluation_progress.md  — This file
```

---

## Changelog

### 2026-03-02 — Phase 2: Camera and Visual Observations
- Added `Observation = dict[str, np.ndarray]` type alias across 13 files
- Added third-person camera (diagonal overhead) and wrist camera (eye-in-hand) to MJCF XML
- Added overhead light for rendering
- Implemented configurable offscreen rendering in ReachTask (`render_cameras` dict)
- Rendering is opt-in: no cameras configured = zero overhead (backward compatible)
- Updated DropoutStressor and LongHorizonDriftStressor to handle uint8 image arrays
- Added `close()` method to ReachTask for renderer cleanup
- Wrote 9 new camera rendering tests (69 total passing)
- Updated reach.yaml config with commented-out camera params
- Decision: camera-to-obs-key mapping via dict (e.g., `{"image": "third_person"}`)
- Decision: renderer lives in ReachTask directly (no separate utility — only one task exists)

### 2026-02-28 — Initial Progress Report
- Documented full project state after Phase 1 completion
- Performed gap analysis for pi0 integration
- Created incremental development plan (Phases 2-7)
- Identified key risks and mitigations

### 2026-02-26 — First Full Evaluation
- Ran ScriptedReachPolicy through all 5 stressors (1,750 episodes)
- Generated first robustness report
- Validated end-to-end pipeline

### 2026-02-26 — ScriptedReachPolicy + Bug Fixes
- Implemented ScriptedReachPolicy (Jacobian-transpose controller)
- Fixed physics param persistence bug (caching in initialize, restore on reset)
- Fixed action-as-delta mapping issue

### Earlier — Framework Build
- Built core framework: interfaces, stressors, runner, metrics, report
- Implemented ReachTask with inline MJCF
- Built config system and CLI
- Wrote 60+ tests
