# Trace Robotics — Multi-Task Robustness Report

**Policy:** Pi0PolicyAdapter  
**Suites:** libero_10  
**Task IDs:** [0, 2, 4, 7, 9]  
**Total tasks evaluated:** 5  
**Modalities:** vision, proprioception  
**Generated:** 2026-05-05 10:30  
**Control frequency:** 50Hz (20ms per step)

---

## Cross-Suite Summary

| Stressor | libero_10 |
|---|---|
| LatencyStressor | 99% (bp=0.54) |
| DropoutStressor | 99% (bp=0.36) |
| PhysicsShiftStressor | 96% (bp=0.10) |
| EmbodimentStressor | 100% (bp=0.10) |
| LongHorizonDriftStressor | 99% (bp=0.42) |
| ImageNoiseStressor | 97% (bp=0.80) |
| OcclusionStressor | 99% (robust) |
| BrightnessShiftStressor | 99% (robust) |
| ResolutionStressor | 97% (bp=0.38) |

## libero_10

### Task 0: *put both the alphabet soup and the tomato sauce in the basket*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.50 (100ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.50 (50% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 0%, robust
- **EmbodimentStressor**: 100% baseline, max deg 13%, robust
- **LongHorizonDriftStressor**: 100% baseline, max deg 100%, bp=0.30 (obs std 0.3 @step100)
- **ImageNoiseStressor**: 100% baseline, max deg 93%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 100% baseline, max deg 33%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 33%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.30 (74px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 274 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 302 |
| 0.20 | 40ms (2 steps) | 100% | 0% | 1.00 | 356 |
| 0.30 | 60ms (3 steps) | 87% | 0% | 0.87 | 389 |
| 0.50 | 100ms (5 steps) | 7% | 0% | 0.07 | 518 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 520 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 520 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 274 |
| 0.10 | 10% drop prob | 93% | 0% | 0.93 | 324 |
| 0.20 | 20% drop prob | 87% | 0% | 0.87 | 362 |
| 0.30 | 30% drop prob | 53% | 0% | 0.53 | 472 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 520 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 520 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 520 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 274 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 100% | 0% | 1.00 | 1 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 100% | 0% | 1.00 | 1 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 100% | 0% | 1.00 | 2 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 100% | 0% | 1.00 | 2 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 100% | 0% | 1.00 | 1 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 100% | 0% | 1.00 | 1 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 271 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 100% | 0% | 1.00 | 4 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 100% | 0% | 1.00 | 4 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 100% | 0% | 1.00 | 39 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 100% | 0% | 1.00 | 2 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 93% | 0% | 0.93 | 71 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 87% | 0% | 0.87 | 120 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 268 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 271 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 67% | 0% | 0.67 | 376 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 40% | 0% | 0.40 | 447 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 0% | 0% | 0.00 | 520 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 0% | 0% | 0.00 | 520 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 0% | 0% | 0.00 | 520 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 278 |
| 0.10 | std=5/255 (2%) | 80% | 0% | 0.80 | 380 |
| 0.20 | std=10/255 (4%) | 73% | 0% | 0.73 | 396 |
| 0.30 | std=15/255 (6%) | 73% | 0% | 0.73 | 405 |
| 0.50 | std=25/255 (10%) | 80% | 0% | 0.80 | 400 |
| 0.70 | std=35/255 (14%) | 80% | 0% | 0.80 | 416 |
| 1.00 | std=50/255 (20%) | 7% | 0% | 0.07 | 513 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 272 |
| 0.10 | 1 patches, up to 3% each | 80% | 0% | 0.80 | 388 |
| 0.20 | 1 patches, up to 6% each | 73% | 0% | 0.73 | 401 |
| 0.30 | 1 patches, up to 9% each | 80% | 0% | 0.80 | 390 |
| 0.50 | 2 patches, up to 15% each | 80% | 0% | 0.80 | 403 |
| 0.70 | 3 patches, up to 21% each | 73% | 0% | 0.73 | 410 |
| 1.00 | 5 patches, up to 30% each | 67% | 0% | 0.67 | 419 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 271 |
| 0.10 | +/-8/255 (3%) | 87% | 0% | 0.87 | 382 |
| 0.20 | +/-16/255 (6%) | 87% | 0% | 0.87 | 377 |
| 0.30 | +/-24/255 (9%) | 80% | 0% | 0.80 | 385 |
| 0.50 | +/-40/255 (16%) | 87% | 0% | 0.87 | 381 |
| 0.70 | +/-56/255 (22%) | 87% | 0% | 0.87 | 380 |
| 1.00 | +/-80/255 (31%) | 67% | 0% | 0.67 | 411 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 276 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 277 |
| 0.20 | 112px effective (2x downscale) | 60% | 0% | 0.60 | 442 |
| 0.30 | 74px effective (3x downscale) | 27% | 0% | 0.27 | 504 |
| 0.50 | 56px effective (4x downscale) | 0% | 0% | 0.00 | 520 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 520 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 520 |

### Task 2: *turn on the stove and put the moka pot on it*

- **LatencyStressor**: 100% baseline, max deg 87%, bp=0.50 (100ms)
- **DropoutStressor**: 93% baseline, max deg 93%, bp=0.20 (20% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 100%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 100% baseline, max deg 100%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 100% baseline, max deg 100%, bp=0.50 (obs std 0.5 @step100)
- **ImageNoiseStressor**: 100% baseline, max deg 40%, robust
- **OcclusionStressor**: 93% baseline, max deg 27%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 33%, robust
- **ResolutionStressor**: 93% baseline, max deg 93%, bp=0.30 (74px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 238 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 250 |
| 0.20 | 40ms (2 steps) | 80% | 0% | 0.80 | 321 |
| 0.30 | 60ms (3 steps) | 80% | 0% | 0.80 | 351 |
| 0.50 | 100ms (5 steps) | 27% | 0% | 0.27 | 469 |
| 0.70 | 140ms (7 steps) | 40% | 0% | 0.40 | 464 |
| 1.00 | 200ms (10 steps) | 13% | 0% | 0.13 | 508 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 93% | 0% | 0.93 | 255 |
| 0.10 | 10% drop prob | 73% | 0% | 0.73 | 326 |
| 0.20 | 20% drop prob | 40% | 0% | 0.40 | 415 |
| 0.30 | 30% drop prob | 7% | 0% | 0.07 | 500 |
| 0.50 | 50% drop prob | 7% | 0% | 0.07 | 506 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 520 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 520 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 251 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 520 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 520 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 520 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 520 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 520 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 520 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 241 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 520 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 520 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 520 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 520 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 520 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 520 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 248 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 250 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 87% | 0% | 0.87 | 289 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 67% | 0% | 0.67 | 358 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 40% | 0% | 0.40 | 414 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 7% | 0% | 0.07 | 506 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 0% | 0% | 0.00 | 520 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 238 |
| 0.10 | std=5/255 (2%) | 93% | 0% | 0.93 | 308 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 289 |
| 0.30 | std=15/255 (6%) | 100% | 0% | 1.00 | 285 |
| 0.50 | std=25/255 (10%) | 93% | 0% | 0.93 | 294 |
| 0.70 | std=35/255 (14%) | 93% | 0% | 0.93 | 290 |
| 1.00 | std=50/255 (20%) | 60% | 0% | 0.60 | 380 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 93% | 0% | 0.93 | 261 |
| 0.10 | 1 patches, up to 3% each | 73% | 0% | 0.73 | 363 |
| 0.20 | 1 patches, up to 6% each | 93% | 0% | 0.93 | 298 |
| 0.30 | 1 patches, up to 9% each | 87% | 0% | 0.87 | 336 |
| 0.50 | 2 patches, up to 15% each | 80% | 0% | 0.80 | 330 |
| 0.70 | 3 patches, up to 21% each | 80% | 0% | 0.80 | 334 |
| 1.00 | 5 patches, up to 30% each | 67% | 0% | 0.67 | 363 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 246 |
| 0.10 | +/-8/255 (3%) | 93% | 0% | 0.93 | 293 |
| 0.20 | +/-16/255 (6%) | 93% | 0% | 0.93 | 305 |
| 0.30 | +/-24/255 (9%) | 80% | 0% | 0.80 | 350 |
| 0.50 | +/-40/255 (16%) | 67% | 0% | 0.67 | 364 |
| 0.70 | +/-56/255 (22%) | 93% | 0% | 0.93 | 302 |
| 1.00 | +/-80/255 (31%) | 73% | 0% | 0.73 | 347 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 93% | 0% | 0.93 | 258 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 246 |
| 0.20 | 112px effective (2x downscale) | 67% | 0% | 0.67 | 405 |
| 0.30 | 74px effective (3x downscale) | 0% | 0% | 0.00 | 520 |
| 0.50 | 56px effective (4x downscale) | 7% | 0% | 0.07 | 510 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 520 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 520 |

### Task 4: *put the white mug on the left plate and put the yellow and white mug on the right plate*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.50 (100ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.50 (50% dropout)
- **PhysicsShiftStressor**: 93% baseline, max deg 93%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 100% baseline, max deg 100%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 100% baseline, max deg 100%, bp=0.30 (obs std 0.3 @step100)
- **ImageNoiseStressor**: 100% baseline, max deg 53%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 100% baseline, max deg 20%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 13%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.30 (74px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 220 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 253 |
| 0.20 | 40ms (2 steps) | 100% | 0% | 1.00 | 262 |
| 0.30 | 60ms (3 steps) | 87% | 0% | 0.87 | 352 |
| 0.50 | 100ms (5 steps) | 27% | 0% | 0.27 | 495 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 520 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 520 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 223 |
| 0.10 | 10% drop prob | 73% | 0% | 0.73 | 338 |
| 0.20 | 20% drop prob | 80% | 0% | 0.80 | 345 |
| 0.30 | 30% drop prob | 53% | 0% | 0.53 | 437 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 520 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 520 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 520 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 93% | 0% | 0.93 | 239 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 520 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 520 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 520 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 520 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 520 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 520 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 223 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 520 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 520 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 520 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 520 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 520 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 20% | 0% | 0.20 | 416 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 224 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 80% | 0% | 0.80 | 296 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 87% | 0% | 0.87 | 276 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 47% | 0% | 0.47 | 395 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 7% | 0% | 0.07 | 505 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 0% | 0% | 0.00 | 520 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 0% | 0% | 0.00 | 520 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 223 |
| 0.10 | std=5/255 (2%) | 93% | 0% | 0.93 | 310 |
| 0.20 | std=10/255 (4%) | 93% | 0% | 0.93 | 340 |
| 0.30 | std=15/255 (6%) | 93% | 0% | 0.93 | 343 |
| 0.50 | std=25/255 (10%) | 87% | 0% | 0.87 | 350 |
| 0.70 | std=35/255 (14%) | 80% | 0% | 0.80 | 345 |
| 1.00 | std=50/255 (20%) | 47% | 0% | 0.47 | 407 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 219 |
| 0.10 | 1 patches, up to 3% each | 93% | 0% | 0.93 | 300 |
| 0.20 | 1 patches, up to 6% each | 100% | 0% | 1.00 | 285 |
| 0.30 | 1 patches, up to 9% each | 87% | 0% | 0.87 | 309 |
| 0.50 | 2 patches, up to 15% each | 93% | 0% | 0.93 | 291 |
| 0.70 | 3 patches, up to 21% each | 100% | 0% | 1.00 | 288 |
| 1.00 | 5 patches, up to 30% each | 80% | 0% | 0.80 | 358 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 223 |
| 0.10 | +/-8/255 (3%) | 93% | 0% | 0.93 | 283 |
| 0.20 | +/-16/255 (6%) | 93% | 0% | 0.93 | 292 |
| 0.30 | +/-24/255 (9%) | 93% | 0% | 0.93 | 287 |
| 0.50 | +/-40/255 (16%) | 87% | 0% | 0.87 | 315 |
| 0.70 | +/-56/255 (22%) | 93% | 0% | 0.93 | 296 |
| 1.00 | +/-80/255 (31%) | 93% | 0% | 0.93 | 297 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 233 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 222 |
| 0.20 | 112px effective (2x downscale) | 100% | 0% | 1.00 | 313 |
| 0.30 | 74px effective (3x downscale) | 40% | 0% | 0.40 | 442 |
| 0.50 | 56px effective (4x downscale) | 20% | 0% | 0.20 | 485 |
| 0.70 | 44px effective (5x downscale) | 13% | 0% | 0.13 | 495 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 520 |

### Task 7: *put both the alphabet soup and the cream cheese box in the basket*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.70 (140ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.50 (50% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 13%, robust
- **EmbodimentStressor**: 100% baseline, max deg 0%, robust
- **LongHorizonDriftStressor**: 100% baseline, max deg 100%, bp=0.50 (obs std 0.5 @step100)
- **ImageNoiseStressor**: 100% baseline, max deg 73%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 100% baseline, max deg 20%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 20%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.70 (44px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 279 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 309 |
| 0.20 | 40ms (2 steps) | 93% | 0% | 0.93 | 307 |
| 0.30 | 60ms (3 steps) | 93% | 0% | 0.93 | 362 |
| 0.50 | 100ms (5 steps) | 67% | 0% | 0.67 | 434 |
| 0.70 | 140ms (7 steps) | 20% | 0% | 0.20 | 493 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 520 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 269 |
| 0.10 | 10% drop prob | 100% | 0% | 1.00 | 281 |
| 0.20 | 20% drop prob | 87% | 0% | 0.87 | 349 |
| 0.30 | 30% drop prob | 87% | 0% | 0.87 | 383 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 520 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 520 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 520 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 264 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 100% | 0% | 1.00 | 1 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 100% | 0% | 1.00 | 1 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 100% | 0% | 1.00 | 1 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 100% | 0% | 1.00 | 1 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 87% | 0% | 0.87 | 70 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 100% | 0% | 1.00 | 1 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 275 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 100% | 0% | 1.00 | 1 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 100% | 0% | 1.00 | 1 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 100% | 0% | 1.00 | 1 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 100% | 0% | 1.00 | 1 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 100% | 0% | 1.00 | 1 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 100% | 0% | 1.00 | 1 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 262 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 293 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 73% | 0% | 0.73 | 366 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 60% | 0% | 0.60 | 381 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 7% | 0% | 0.07 | 502 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 0% | 0% | 0.00 | 520 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 0% | 0% | 0.00 | 520 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 286 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 333 |
| 0.20 | std=10/255 (4%) | 93% | 0% | 0.93 | 345 |
| 0.30 | std=15/255 (6%) | 93% | 0% | 0.93 | 343 |
| 0.50 | std=25/255 (10%) | 100% | 0% | 1.00 | 329 |
| 0.70 | std=35/255 (14%) | 80% | 0% | 0.80 | 363 |
| 1.00 | std=50/255 (20%) | 27% | 0% | 0.27 | 497 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 267 |
| 0.10 | 1 patches, up to 3% each | 93% | 0% | 0.93 | 328 |
| 0.20 | 1 patches, up to 6% each | 93% | 0% | 0.93 | 337 |
| 0.30 | 1 patches, up to 9% each | 93% | 0% | 0.93 | 350 |
| 0.50 | 2 patches, up to 15% each | 100% | 0% | 1.00 | 329 |
| 0.70 | 3 patches, up to 21% each | 100% | 0% | 1.00 | 326 |
| 1.00 | 5 patches, up to 30% each | 80% | 0% | 0.80 | 360 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 283 |
| 0.10 | +/-8/255 (3%) | 93% | 0% | 0.93 | 328 |
| 0.20 | +/-16/255 (6%) | 93% | 0% | 0.93 | 335 |
| 0.30 | +/-24/255 (9%) | 93% | 0% | 0.93 | 350 |
| 0.50 | +/-40/255 (16%) | 93% | 0% | 0.93 | 360 |
| 0.70 | +/-56/255 (22%) | 100% | 0% | 1.00 | 337 |
| 1.00 | +/-80/255 (31%) | 80% | 0% | 0.80 | 365 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 270 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 270 |
| 0.20 | 112px effective (2x downscale) | 100% | 0% | 1.00 | 336 |
| 0.30 | 74px effective (3x downscale) | 73% | 0% | 0.73 | 404 |
| 0.50 | 56px effective (4x downscale) | 87% | 0% | 0.87 | 464 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 520 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 520 |

### Task 9: *put the yellow and white mug in the microwave and close it*

- **LatencyStressor**: 93% baseline, max deg 87%, bp=0.50 (100ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.10 (10% dropout)
- **PhysicsShiftStressor**: 87% baseline, max deg 87%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 100% baseline, max deg 100%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 93% baseline, max deg 93%, bp=0.50 (obs std 0.5 @step100)
- **ImageNoiseStressor**: 87% baseline, max deg 40%, bp=0.20 (noise std=10)
- **OcclusionStressor**: 100% baseline, max deg 47%, robust
- **BrightnessShiftStressor**: 93% baseline, max deg 33%, robust
- **ResolutionStressor**: 93% baseline, max deg 93%, bp=0.30 (74px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 93% | 0% | 0.93 | 282 |
| 0.10 | 20ms (1 steps) | 87% | 0% | 0.87 | 322 |
| 0.20 | 40ms (2 steps) | 80% | 0% | 0.80 | 375 |
| 0.30 | 60ms (3 steps) | 53% | 0% | 0.53 | 445 |
| 0.50 | 100ms (5 steps) | 20% | 0% | 0.20 | 492 |
| 0.70 | 140ms (7 steps) | 20% | 0% | 0.20 | 506 |
| 1.00 | 200ms (10 steps) | 7% | 0% | 0.07 | 512 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 269 |
| 0.10 | 10% drop prob | 47% | 0% | 0.47 | 408 |
| 0.20 | 20% drop prob | 33% | 0% | 0.33 | 472 |
| 0.30 | 30% drop prob | 13% | 0% | 0.13 | 497 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 520 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 520 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 520 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 87% | 0% | 0.87 | 289 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 520 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 520 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 520 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 520 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 520 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 520 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 271 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 520 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 520 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 520 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 520 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 520 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 520 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 93% | 0% | 0.93 | 277 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 87% | 0% | 0.87 | 290 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 283 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 73% | 0% | 0.73 | 375 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 13% | 0% | 0.13 | 496 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 0% | 0% | 0.00 | 520 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 0% | 0% | 0.00 | 520 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 87% | 0% | 0.87 | 294 |
| 0.10 | std=5/255 (2%) | 60% | 0% | 0.60 | 409 |
| 0.20 | std=10/255 (4%) | 47% | 0% | 0.47 | 452 |
| 0.30 | std=15/255 (6%) | 73% | 0% | 0.73 | 376 |
| 0.50 | std=25/255 (10%) | 53% | 0% | 0.53 | 435 |
| 0.70 | std=35/255 (14%) | 73% | 0% | 0.73 | 403 |
| 1.00 | std=50/255 (20%) | 60% | 0% | 0.60 | 389 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 268 |
| 0.10 | 1 patches, up to 3% each | 80% | 0% | 0.80 | 382 |
| 0.20 | 1 patches, up to 6% each | 53% | 0% | 0.53 | 419 |
| 0.30 | 1 patches, up to 9% each | 53% | 0% | 0.53 | 423 |
| 0.50 | 2 patches, up to 15% each | 53% | 0% | 0.53 | 421 |
| 0.70 | 3 patches, up to 21% each | 53% | 0% | 0.53 | 428 |
| 1.00 | 5 patches, up to 30% each | 53% | 0% | 0.53 | 449 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 93% | 0% | 0.93 | 277 |
| 0.10 | +/-8/255 (3%) | 67% | 0% | 0.67 | 402 |
| 0.20 | +/-16/255 (6%) | 67% | 0% | 0.67 | 396 |
| 0.30 | +/-24/255 (9%) | 73% | 0% | 0.73 | 380 |
| 0.50 | +/-40/255 (16%) | 73% | 0% | 0.73 | 378 |
| 0.70 | +/-56/255 (22%) | 87% | 0% | 0.87 | 360 |
| 1.00 | +/-80/255 (31%) | 60% | 0% | 0.60 | 414 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 93% | 0% | 0.93 | 297 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 277 |
| 0.20 | 112px effective (2x downscale) | 60% | 0% | 0.60 | 425 |
| 0.30 | 74px effective (3x downscale) | 33% | 0% | 0.33 | 460 |
| 0.50 | 56px effective (4x downscale) | 0% | 0% | 0.00 | 520 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 520 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 520 |


## Breakpoint Comparison

Average intensity at which success rate drops below 50%:

- **LatencyStressor**: libero_10: 0.54
- **DropoutStressor**: libero_10: 0.36
- **PhysicsShiftStressor**: libero_10: 0.10
- **EmbodimentStressor**: libero_10: 0.10
- **LongHorizonDriftStressor**: libero_10: 0.42
- **ImageNoiseStressor**: libero_10: 0.80
- **OcclusionStressor**: libero_10: robust
- **BrightnessShiftStressor**: libero_10: robust
- **ResolutionStressor**: libero_10: 0.38

---

*Report generated by Trace Robotics v0.1.0*
*https://tracerobotics.com*