# Trace Robotics — Stressor Reference

All stressors inherit from `BaseStressor` (`trace/stressor_engine/base.py`) and implement three hooks:

- `on_episode_start(task)` — modify the environment before the episode begins
- `perturb_observation(obs)` — alter observations before they reach the policy
- `perturb_action(action)` — alter actions before they reach the simulator

Every stressor is parameterized by a scalar **intensity** in `[0.0, 1.0]`:
- `0.0` = no perturbation (passthrough)
- `1.0` = maximum perturbation

All stressors are seeded and deterministic.

---

## 1. LatencyStressor

**File:** `trace/stressor_engine/latency.py`

Simulates communication delay between the policy and robot actuators. Buffers outgoing actions and replays stale ones, forcing the robot to act on delayed commands.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_delay_steps` | 10 | Maximum buffer depth at intensity=1.0 (~200ms at 50Hz) |

**How it works:**
- At `intensity=0.5` with `max_delay_steps=10`, the delay is 5 steps
- New actions are pushed into a FIFO buffer; the oldest buffered action is sent to the robot
- Until the buffer fills, zero actions are sent (robot holds still)

**Affects:** Actions only. Observations pass through unchanged.

**Pi0-LIBERO results:** Breakpoint at intensity 0.50 — success drops from 98% to 0%. Pi0 is very sensitive to action delay.

---

## 2. DropoutStressor

**File:** `trace/stressor_engine/dropout.py`

Simulates sensor failures: camera blackout, noisy readings, or missing data. Each observation channel is independently dropped with probability equal to the intensity.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `"zero"` | Dropout replacement: `"zero"` (zeros), `"noise"` (random), `"freeze"` (last good reading) |
| `noise_scale` | 0.1 | Noise std when `mode="noise"` |

**How it works:**
- Each key in the observation dict is independently tested: with probability `intensity`, it's replaced
- `zero` mode: replaced with zeros (total blackout)
- `noise` mode: replaced with random noise (garbled sensor)
- `freeze` mode: replaced with the last clean observation (stale data)
- Handles both float arrays and uint8 images

**Affects:** Observations only. Actions pass through unchanged.

**Pi0-LIBERO results:** Breakpoint at intensity 0.70 — success drops from 100% to 0%. Policy tolerates occasional dropout but collapses when most readings are missing.

---

## 3. PhysicsShiftStressor

**File:** `trace/stressor_engine/physics_shift.py`

Modifies physical properties of the MuJoCo simulation to simulate sim-to-real transfer gaps. Perturbs mass, friction, and damping at episode start.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mass_range` | `[0.5, 2.0]` | Scale factor range for body masses |
| `friction_range` | `[0.3, 1.5]` | Scale factor range for geom friction |
| `damping_range` | `[0.5, 2.0]` | Scale factor range for joint damping |

**How it works:**
- At episode start, retrieves `mujoco.MjModel` from the task
- Saves original values, then scales mass/friction/damping by a factor interpolated between 1.0 (nominal) and a random value from the range based on intensity
- Physics reset happens naturally on task reset

**Affects:** Environment physics. Observations and actions pass through unchanged.

**Pi0-LIBERO results:** Robust — 100% baseline to 86% at max intensity, no breakpoint. Pi0 handles moderate physics variation well.

**LIBERO compatibility:** Requires `task.get_mujoco_model()` to return a proper `mujoco.MjModel`. Works via `env.sim.model._model` in robosuite-based envs.

---

## 4. EmbodimentStressor

**File:** `trace/stressor_engine/embodiment.py`

Simulates deployment on a different robot than the one used for training. Perturbs link geometry, joint limits, and actuator gains.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `link_length_range` | `[0.9, 1.1]` | Scale factor range for geom sizes |
| `joint_limit_range` | `[0.85, 1.0]` | Scale factor range for joint ranges (tightens limits) |
| `gain_range` | `[0.7, 1.3]` | Scale factor range for actuator gains |

**How it works:**
- At episode start, scales `geom_size` (link geometry), `jnt_range` (joint limits), and `actuator_gainprm` (motor gains)
- The policy must compensate for a kinematically and dynamically different robot

**Affects:** Environment embodiment. Observations and actions pass through unchanged.

**Pi0-LIBERO results:** Robust — 100% baseline to 98% at max intensity. Pi0 generalizes well to small embodiment changes on LIBERO tasks.

**LIBERO compatibility:** Same as PhysicsShiftStressor — requires `mujoco.MjModel` access.

---

## 5. LongHorizonDriftStressor

**File:** `trace/stressor_engine/long_horizon.py`

Simulates gradual degradation over time: sensor drift, accumulating noise, and slow calibration loss. Perturbation magnitude grows linearly with step count.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `obs_noise_growth` | 0.01 | Observation noise growth rate per step |
| `action_noise_growth` | 0.005 | Action noise growth rate per step |

**How it works:**
- `drift_factor = intensity * current_step`
- At each step, Gaussian noise with `std = growth_rate * drift_factor` is added to observations and actions
- Early steps are clean; later steps become increasingly noisy
- Skips uint8 image arrays (use visual stressors for image drift)

**Affects:** Both observations and actions, with increasing severity.

**Pi0-LIBERO results:** Mostly robust — 100% baseline to 78% at max intensity, no breakpoint. Tasks complete before drift accumulates enough to cause failure.

---

## 6. ImageNoiseStressor

**File:** `trace/stressor_engine/visual.py`

Adds Gaussian noise to camera images, simulating noisy CMOS sensors or electromagnetic interference.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_noise_std` | 50.0 | Noise std (in pixel values 0-255) at intensity=1.0 |

**How it works:**
- Applies Gaussian noise to `image` and `wrist_image` keys
- Noise std = `intensity * max_noise_std`
- Output is clipped to [0, 255] uint8

**Affects:** Image observations only. Non-image obs and actions pass through.

**Pi0-LIBERO results:** Fully robust — 100% at all intensities. Pi0's vision encoder is highly noise-tolerant.

---

## 7. OcclusionStressor

**File:** `trace/stressor_engine/visual.py`

Overlays random black rectangles on camera images, simulating partial camera obstruction, objects in the foreground, or dirt on the lens.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_patches` | 5 | Maximum number of rectangles at intensity=1.0 |
| `max_patch_frac` | 0.3 | Maximum patch size as fraction of image dimension |
| `fill_value` | 0 | Pixel value for occluding patches (0 = black) |

**How it works:**
- Number of patches = `intensity * max_patches` (at least 1)
- Each patch size is random between 5% and `intensity * max_patch_frac` of image dims
- Patches are placed at random positions

**Affects:** Image observations only.

**Pi0-LIBERO results:** Fully robust — 98% at all intensities. Pi0 handles partial occlusion well, likely due to training data augmentation.

---

## 8. BrightnessShiftStressor

**File:** `trace/stressor_engine/visual.py`

Shifts pixel brightness uniformly across camera images, simulating exposure changes, lighting variation, or auto-gain drift.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_shift` | 80.0 | Maximum pixel brightness shift at intensity=1.0 |

**How it works:**
- At episode start, picks a random shift direction (brighter or darker)
- `shift = uniform(-1, 1) * intensity * max_shift`
- Same shift is applied to all frames in the episode (consistent lighting change)
- Output clipped to [0, 255]

**Affects:** Image observations only.

**Pi0-LIBERO results:** Fully robust — 98% at all intensities. Pi0 handles brightness variation without degradation.

---

## 9. ResolutionStressor

**File:** `trace/stressor_engine/visual.py`

Downscales then upscales camera images (pixelation), simulating low-resolution cameras or bandwidth-constrained video streams.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_downscale_factor` | 8 | Downscale factor at intensity=1.0 (8x = 224px → 28px → 224px) |

**How it works:**
- `factor = 1 + intensity * (max_downscale_factor - 1)`
- Block-average downscale (no OpenCV dependency), then nearest-neighbor upscale
- At intensity=1.0 with 224x224 images: downscales to 28x28 then upscales back — heavy pixelation

**Affects:** Image observations only.

**Pi0-LIBERO results:** Breakpoint at intensity 1.00 — 100% baseline to 2%. Extreme pixelation (28x28 effective resolution) destroys spatial information. Moderate pixelation is tolerated.

---

## Summary Table

| # | Stressor | Target | Pi0-LIBERO Baseline | Pi0-LIBERO Breakpoint | Verdict |
|---|----------|--------|---------------------|-----------------------|---------|
| 1 | LatencyStressor | Actions | 98% | 0.50 | Fragile |
| 2 | DropoutStressor | Observations | 100% | 0.70 | Fragile |
| 3 | PhysicsShiftStressor | Environment | 100% | none | Robust |
| 4 | EmbodimentStressor | Environment | 100% | none | Robust |
| 5 | LongHorizonDriftStressor | Obs + Actions | 100% | none | Robust |
| 6 | ImageNoiseStressor | Images | 100% | none | Robust |
| 7 | OcclusionStressor | Images | 98% | none | Robust |
| 8 | BrightnessShiftStressor | Images | 98% | none | Robust |
| 9 | ResolutionStressor | Images | 100% | 1.00 | Fragile |

**Key insight:** Pi0 is most vulnerable to **action-level** perturbations (latency) and **information-destroying** perturbations (dropout, extreme resolution loss). It is remarkably robust to additive noise, occlusion, brightness changes, and moderate physics variation.

---

## Stressor Pipeline

Stressors are chained in the `EpisodeRunner` loop:

```
observation = task.get_observation()
observation = stressor.perturb_observation(observation)    # obs corruption
action = policy.act(observation)
action = stressor.perturb_action(action)                   # action corruption
task.step(action)
```

Multiple stressors can stack (applied sequentially), and `on_episode_start()` runs before the first step to modify environment-level properties.

## Adding a New Stressor

1. Create `trace/stressor_engine/my_stressor.py`
2. Inherit from `BaseStressor`, implement `on_episode_start`, `perturb_observation`, `perturb_action`
3. Register in `STRESSOR_REGISTRY` in `trace/config_loader.py`
4. Add to sweep config YAML in `configs/sweeps/`
5. Add tests in `tests/`
