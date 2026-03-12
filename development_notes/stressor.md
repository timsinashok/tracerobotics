# Trace Robotics — Stressor Reference

All stressors inherit from `BaseStressor` (`trace/stressor_engine/base.py`) and implement three hooks:

- `on_episode_start(task)` — modify the environment before the episode begins
- `perturb_observation(obs)` — alter observations before they reach the policy
- `perturb_action(action)` — alter actions before they reach the simulator

Every stressor is parameterized by a scalar **intensity** in `[0.0, 1.0]`:
- `0.0` = no perturbation (passthrough)
- `1.0` = maximum perturbation

All stressors are seeded and deterministic.

**Evaluation environment:** LIBERO-Spatial benchmark (tabletop manipulation, Franka Panda arm, 224x224 camera images, 50Hz control loop, 220 max steps per episode, 10 tasks).

**Models evaluated:**
- **Pi0.5** (Physical Intelligence) — 3B parameter VLA, served via WebSocket, action chunking 50 predicted / 5 executed
- **OpenVLA-OFT** (Stanford) — 7B parameter VLA, loaded in-process on GPU, action chunking 8 predicted / 8 executed

**Sweep parameters:** 7 intensity levels (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0), 5 seeds, 10 episodes per config = **350 episodes per stressor per model**.

---

## 1. LatencyStressor

**File:** `trace/stressor_engine/latency.py`

Simulates communication delay between the policy and robot actuators. Buffers outgoing actions and replays stale ones, forcing the robot to act on delayed commands.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_delay_steps` | 10 | Maximum buffer depth at intensity=1.0 |

**How it works:**
- `delay_steps = round(intensity * max_delay_steps)`
- New actions are pushed into a FIFO buffer; the oldest buffered action is sent to the robot
- Until the buffer fills, zero actions are sent (robot holds still)
- At 50Hz control: each step = 20ms

**Real-world delay mapping:**

| Intensity | Delay Steps | Real-World Delay | Context |
|-----------|------------|-----------------|---------|
| 0.00 | 0 | 0ms | Ideal (no delay) |
| 0.10 | 1 | 20ms | LAN / co-located GPU |
| 0.20 | 2 | 40ms | Fast cloud inference |
| 0.30 | 3 | 60ms | Typical edge compute |
| 0.50 | 5 | 100ms | Normal deployment range (80-150ms is common) |
| 0.70 | 7 | 140ms | WiFi / congested network |
| 1.00 | 10 | 200ms | Remote cloud / poor network |

**Affects:** Actions only. Observations pass through unchanged.

**Results:**

| Intensity | Pi0.5 | OpenVLA | Real Delay |
|-----------|-------|---------|-----------|
| 0.00 | 100% | 100% | 0ms |
| 0.10 | 100% | 100% | 20ms |
| 0.20 | 98% | 86% | 40ms |
| 0.30 | 80% | 34% | 60ms |
| 0.50 | 14% | 0% | 100ms |
| 0.70 | 0% | 0% | 140ms |
| 1.00 | 0% | 0% | 200ms |

**Key insight:** Both models collapse between 60-100ms. This is within normal production deployment latency (80-150ms). OpenVLA is even more latency-sensitive than Pi0.5 — it drops to 34% at 60ms where Pi0.5 is still at 80%. Neither model can operate reliably with latency common in real robot stacks.

**Pi0.5 breakpoint:** intensity 0.50 (100ms)
**OpenVLA breakpoint:** intensity 0.30 (60ms)

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
- At intensity=0.30: each observation has a 30% chance of being blanked per timestep

**Real-world analog:** WiFi camera dropouts, USB disconnections, sensor bus contention, intermittent hardware failures.

**Affects:** Observations only. Actions pass through unchanged.

**Results:**

| Intensity | Pi0.5 | OpenVLA | Meaning |
|-----------|-------|---------|---------|
| 0.00 | 98% | 100% | No dropout |
| 0.10 | 96% | 72% | 10% of readings dropped |
| 0.20 | 98% | 56% | 20% of readings dropped |
| 0.30 | 90% | 28% | 30% of readings dropped |
| 0.50 | 50% | 10% | Half of readings dropped |
| 0.70 | 4% | 0% | 70% of readings dropped |
| 1.00 | 0% | 0% | All readings dropped |

**Key insight:** OpenVLA is dramatically more sensitive to dropout than Pi0.5. At 30% dropout, Pi0.5 maintains 90% success while OpenVLA drops to 28%. Pi0.5's action chunking (predicts 50 actions, executes 5) likely provides a buffer — it can coast through missing observations using its longer action plan. OpenVLA executes all 8 predicted actions, so it has no coasting buffer.

**Pi0.5 breakpoint:** intensity 0.70
**OpenVLA breakpoint:** intensity 0.30

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
- At intensity=0.5: masses could be scaled ~0.75x-1.5x, friction ~0.65x-1.25x
- At intensity=1.0: full range applies — masses 0.5x-2.0x, friction 0.3x-1.5x, damping 0.5x-2.0x
- Physics reset happens naturally on task reset

**Real-world analog:** Moving from simulation to a real robot (sim-to-real gap), different table surfaces, different object weights, wear and tear on joints.

**Affects:** Environment physics. Observations and actions pass through unchanged.

**Results:**

| Intensity | Pi0.5 | OpenVLA |
|-----------|-------|---------|
| 0.00 | 100% | 100% |
| 0.10 | 100% | 100% |
| 0.20 | 100% | 100% |
| 0.30 | 100% | 100% |
| 0.50 | 98% | 100% |
| 0.70 | 100% | 100% |
| 1.00 | 88% | 86% |

**Key insight:** Both models are remarkably robust to physics variation. Even at maximum intensity (2x mass, 0.3x friction, 2x damping), success stays above 86%. This suggests both models have learned dynamics-agnostic manipulation strategies, or that LIBERO tasks don't require precise force control.

**Pi0.5 breakpoint:** none (robust)
**OpenVLA breakpoint:** none (robust)

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
- At intensity=1.0: links could be 0.9x-1.1x size, joints tightened to 85% range, gains 0.7x-1.3x
- The policy must compensate for a kinematically and dynamically different robot

**Real-world analog:** Deploying on a slightly different robot model, manufacturing tolerance variation, worn gearboxes, arm with different reach.

**Affects:** Environment embodiment. Observations and actions pass through unchanged.

**Results:**

| Intensity | Pi0.5 | OpenVLA |
|-----------|-------|---------|
| 0.00 | 100% | 100% |
| 0.10 | 100% | 100% |
| 0.20 | 100% | 100% |
| 0.30 | 100% | 100% |
| 0.50 | 100% | 100% |
| 0.70 | 100% | 100% |
| 1.00 | 100% | 100% |

**Key insight:** Both models are perfectly robust to the embodiment perturbation ranges we tested. 100% success across all intensities for both models. The ranges tested (±10% link length, ±15% joint limits, ±30% gains) may be too conservative — future work should test wider ranges. Alternatively, tabletop pick-and-place may simply not require precise kinematics.

**Pi0.5 breakpoint:** none (robust)
**OpenVLA breakpoint:** none (robust)

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
- At step 100 with intensity=0.5: obs noise std = 0.01 * 50 = 0.5, action noise std = 0.005 * 50 = 0.25
- At step 200 with intensity=1.0: obs noise std = 0.01 * 200 = 2.0, action noise std = 0.005 * 200 = 1.0
- Skips uint8 image arrays (use visual stressors for image drift)

**Real-world analog:** Thermal drift in sensors, IMU bias accumulation, slow calibration loss, battery voltage drop affecting actuator precision.

**Affects:** Both observations and actions, with increasing severity.

**Results:**

| Intensity | Pi0.5 | OpenVLA |
|-----------|-------|---------|
| 0.00 | 100% | 100% |
| 0.10 | 100% | 86% |
| 0.20 | 100% | 56% |
| 0.30 | 100% | 38% |
| 0.50 | 100% | 18% |
| 0.70 | 88% | 14% |
| 1.00 | 82% | 12% |

**Key insight:** Massive divergence between models. Pi0.5 is highly robust (82% at max intensity) while OpenVLA collapses to 12%. This is likely because Pi0.5 completes tasks faster (~80 steps) so drift has less time to accumulate, and its action chunking provides temporal smoothing. OpenVLA takes similar steps (~75) but its per-step action execution means drift noise is applied to every action without smoothing.

**Pi0.5 breakpoint:** none (robust)
**OpenVLA breakpoint:** intensity 0.30

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
- At intensity=0.5: noise std = 25 pixel values (~10% of dynamic range)
- At intensity=1.0: noise std = 50 pixel values (~20% of dynamic range)
- Output is clipped to [0, 255] uint8

**Real-world analog:** Low-light camera noise, electromagnetic interference, cheap sensors, compression artifacts.

**Affects:** Image observations only. Non-image obs and actions pass through.

**Results:**

| Intensity | Pi0.5 | OpenVLA |
|-----------|-------|---------|
| 0.00 | 100% | 100% |
| 0.10 | 100% | 100% |
| 0.20 | 100% | 100% |
| 0.30 | 100% | 100% |
| 0.50 | 100% | 100% |
| 0.70 | 100% | 100% |
| 1.00 | 98% | 100% |

**Key insight:** Both models are fully robust to image noise up to std=50 (significant visual corruption). Modern vision encoders (SigLIP in OpenVLA, likely similar in Pi0.5) handle additive Gaussian noise extremely well. This is expected — ImageNet-pretrained encoders are known to be noise-tolerant.

**Pi0.5 breakpoint:** none (robust)
**OpenVLA breakpoint:** none (robust)

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
- Number of patches = `round(intensity * max_patches)` (at least 1 when intensity > 0)
- Each patch size is random between 5% and `intensity * max_patch_frac` of image dims
- At intensity=1.0: up to 5 patches, each up to 30% of image width/height — can occlude ~50-70% of the image
- Patches are placed at random positions, different per frame

**Real-world analog:** Objects passing in front of camera, dirt/smudges on lens, partial camera blockage, shadows.

**Affects:** Image observations only.

**Results:**

| Intensity | Pi0.5 | OpenVLA |
|-----------|-------|---------|
| 0.00 | 100% | 100% |
| 0.10 | 100% | 100% |
| 0.20 | 100% | 100% |
| 0.30 | 100% | 100% |
| 0.50 | 100% | 100% |
| 0.70 | 98% | 100% |
| 1.00 | 94% | 98% |

**Key insight:** Both models handle occlusion remarkably well. Even with ~50-70% of the image blocked, success stays above 94%. Vision encoders pretrained on diverse internet data have likely learned to extract spatial information from partial views. Random rectangular occlusion may also be less destructive than targeted occlusion of task-relevant regions.

**Pi0.5 breakpoint:** none (robust)
**OpenVLA breakpoint:** none (robust)

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
- At intensity=1.0: shift up to ±80 pixel values (~31% of dynamic range)
- Same shift is applied to all frames in the episode (consistent lighting change)
- Output clipped to [0, 255]

**Real-world analog:** Different lighting conditions (warehouse vs outdoor), time-of-day variation, auto-exposure changes, moving from lab to deployment site.

**Affects:** Image observations only.

**Results:**

| Intensity | Pi0.5 | OpenVLA |
|-----------|-------|---------|
| 0.00 | 100% | 100% |
| 0.10 | 100% | 100% |
| 0.20 | 100% | 98% |
| 0.30 | 100% | 100% |
| 0.50 | 98% | 100% |
| 0.70 | 100% | 98% |
| 1.00 | 100% | 100% |

**Key insight:** Both models are fully robust to brightness shifts up to ±80 pixel values. Vision encoders normalize input, and brightness is one of the most common data augmentations used in pretraining. No surprise here — this is a solved problem for modern vision models.

**Pi0.5 breakpoint:** none (robust)
**OpenVLA breakpoint:** none (robust)

---

## 9. ResolutionStressor

**File:** `trace/stressor_engine/visual.py`

Downscales then upscales camera images (pixelation), simulating low-resolution cameras or bandwidth-constrained video streams.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_downscale_factor` | 8 | Downscale factor at intensity=1.0 |

**How it works:**
- `factor = 1 + intensity * (max_downscale_factor - 1)`
- Block-average downscale (no OpenCV dependency), then nearest-neighbor upscale
- At intensity=0.5: factor ≈ 4.5x → 224px becomes ~50px effective resolution
- At intensity=1.0: factor = 8x → 224px becomes 28px effective resolution (heavy pixelation)
- Image dimensions are preserved (downscale then upscale back) so the policy sees the same tensor shape

**Effective resolution at each intensity:**

| Intensity | Downscale Factor | Effective Resolution | Equivalent Camera |
|-----------|-----------------|---------------------|-------------------|
| 0.00 | 1.0x | 224x224 | Standard HD crop |
| 0.10 | 1.7x | ~132x132 | Good quality |
| 0.20 | 2.4x | ~93x93 | Medium quality |
| 0.30 | 3.1x | ~72x72 | Low quality webcam |
| 0.50 | 4.5x | ~50x50 | Very low quality |
| 0.70 | 5.9x | ~38x38 | Thumbnail-level |
| 1.00 | 8.0x | 28x28 | MNIST-level pixelation |

**Real-world analog:** Low-cost cameras, bandwidth-constrained video streaming (compressed feeds over WiFi), downsampled video for faster transmission.

**Affects:** Image observations only.

**Results:**

| Intensity | Pi0.5 | OpenVLA | Effective Res |
|-----------|-------|---------|--------------|
| 0.00 | 100% | 100% | 224px |
| 0.10 | 100% | 100% | ~132px |
| 0.20 | 100% | 100% | ~93px |
| 0.30 | 100% | 100% | ~72px |
| 0.50 | 72% | 100% | ~50px |
| 0.70 | 78% | 98% | ~38px |
| 1.00 | 4% | 36% | 28px |

**Key insight:** OpenVLA is significantly more resolution-robust than Pi0.5. OpenVLA maintains 100% at ~50px and 98% at ~38px, while Pi0.5 drops to 72% and 78% respectively. At maximum pixelation (28px), OpenVLA still achieves 36% vs Pi0.5's 4%. This may reflect OpenVLA's vision encoder (SigLIP) handling coarse spatial features better, or its center-cropping augmentation during training providing robustness to reduced spatial detail.

**Pi0.5 breakpoint:** intensity 1.00 (28px effective)
**OpenVLA breakpoint:** intensity 1.00 (28px effective)

---

## Multi-Model Comparison Table

| # | Stressor | Target | Pi0.5 Breakpoint | OpenVLA Breakpoint | More Robust Model |
|---|----------|--------|-----------------|-------------------|-------------------|
| 1 | LatencyStressor | Actions | 0.50 (100ms) | 0.30 (60ms) | Pi0.5 |
| 2 | DropoutStressor | Observations | 0.70 (70% drop) | 0.30 (30% drop) | Pi0.5 |
| 3 | PhysicsShiftStressor | Environment | none | none | Tie |
| 4 | EmbodimentStressor | Environment | none | none | Tie |
| 5 | LongHorizonDriftStressor | Obs + Actions | none | 0.30 | Pi0.5 |
| 6 | ImageNoiseStressor | Images | none | none | Tie |
| 7 | OcclusionStressor | Images | none | none | Tie |
| 8 | BrightnessShiftStressor | Images | none | none | Tie |
| 9 | ResolutionStressor | Images | 1.00 (28px) | 1.00 (28px) | OpenVLA |

**Fragile stressors (at least one model breaks):** Latency, Dropout, LongHorizonDrift, Resolution
**Robust stressors (both models survive):** PhysicsShift, Embodiment, ImageNoise, Occlusion, Brightness

---

## Key Takeaways

1. **Latency is the #1 failure mode** — both models collapse at delays common in production robot stacks (60-100ms). This is the strongest finding: it's systemic, not model-specific.

2. **Pi0.5 is more robust overall** — survives longer under latency, dropout, and drift. Its action chunking (predict 50, execute 5) likely provides temporal smoothing and coasting ability through perturbations.

3. **OpenVLA has better vision robustness** — outperforms Pi0.5 on resolution degradation. SigLIP vision encoder handles coarse spatial features better.

4. **Vision perturbations are mostly solved** — image noise, occlusion, brightness shifts don't affect either model. Modern pretrained vision encoders handle these well out of the box.

5. **Action/temporal perturbations are the real threat** — latency, dropout, and drift are the deployment-relevant failure modes. These are harder to train away because they involve the control loop, not just perception.

6. **Architecture affects robustness profile** — Pi0.5 (larger action chunks = temporal buffer) vs OpenVLA (execute every prediction = no buffer). The same stressor affects architecturally different models differently.

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
