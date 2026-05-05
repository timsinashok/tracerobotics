# Trace Robotics — Multi-Task Robustness Report

**Policy:** Pi0PolicyAdapter  
**Suites:** libero_spatial  
**Task IDs:** [0, 2, 4, 7, 9]  
**Total tasks evaluated:** 5  
**Modalities:** vision, proprioception  
**Generated:** 2026-05-04 19:24  
**Control frequency:** 50Hz (20ms per step)

---

## Cross-Suite Summary

| Stressor | libero_spatial |
|---|---|
| LatencyStressor | 97% (bp=0.54) |
| DropoutStressor | 99% (bp=0.50) |
| PhysicsShiftStressor | 100% (bp=0.10) |
| EmbodimentStressor | 99% (bp=0.10) |
| LongHorizonDriftStressor | 100% (bp=1.00) |
| ImageNoiseStressor | 96% (bp=1.00) |
| OcclusionStressor | 100% (robust) |
| BrightnessShiftStressor | 99% (robust) |
| ResolutionStressor | 99% (bp=0.50) |

## libero_spatial

### Task 0: *pick up the black bowl between the plate and the ramekin and place it on the plate*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.30 (60ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.70 (70% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 100%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 100% baseline, max deg 100%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 100% baseline, max deg 13%, robust
- **ImageNoiseStressor**: 100% baseline, max deg 20%, robust
- **OcclusionStressor**: 100% baseline, max deg 33%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 7%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.50 (56px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 83 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 80 |
| 0.20 | 40ms (2 steps) | 100% | 0% | 1.00 | 102 |
| 0.30 | 60ms (3 steps) | 40% | 0% | 0.40 | 185 |
| 0.50 | 100ms (5 steps) | 7% | 0% | 0.07 | 217 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 220 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 220 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 89 |
| 0.10 | 10% drop prob | 100% | 0% | 1.00 | 89 |
| 0.20 | 20% drop prob | 93% | 0% | 0.93 | 102 |
| 0.30 | 30% drop prob | 93% | 0% | 0.93 | 130 |
| 0.50 | 50% drop prob | 53% | 0% | 0.53 | 179 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 220 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 220 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 82 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 220 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 220 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 220 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 220 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 220 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 220 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 85 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 220 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 220 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 220 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 220 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 220 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 220 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 82 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 87 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 81 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 100% | 0% | 1.00 | 82 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 100% | 0% | 1.00 | 80 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 87% | 0% | 0.87 | 102 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 87% | 0% | 0.87 | 99 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 78 |
| 0.10 | std=5/255 (2%) | 93% | 0% | 0.93 | 125 |
| 0.20 | std=10/255 (4%) | 93% | 0% | 0.93 | 119 |
| 0.30 | std=15/255 (6%) | 87% | 0% | 0.87 | 131 |
| 0.50 | std=25/255 (10%) | 87% | 0% | 0.87 | 111 |
| 0.70 | std=35/255 (14%) | 93% | 0% | 0.93 | 112 |
| 1.00 | std=50/255 (20%) | 80% | 0% | 0.80 | 125 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 82 |
| 0.10 | 1 patches, up to 3% each | 93% | 0% | 0.93 | 114 |
| 0.20 | 1 patches, up to 6% each | 93% | 0% | 0.93 | 111 |
| 0.30 | 1 patches, up to 9% each | 93% | 0% | 0.93 | 125 |
| 0.50 | 2 patches, up to 15% each | 67% | 0% | 0.67 | 145 |
| 0.70 | 3 patches, up to 21% each | 80% | 0% | 0.80 | 140 |
| 1.00 | 5 patches, up to 30% each | 67% | 0% | 0.67 | 145 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 85 |
| 0.10 | +/-8/255 (3%) | 93% | 0% | 0.93 | 128 |
| 0.20 | +/-16/255 (6%) | 93% | 0% | 0.93 | 113 |
| 0.30 | +/-24/255 (9%) | 100% | 0% | 1.00 | 110 |
| 0.50 | +/-40/255 (16%) | 100% | 0% | 1.00 | 114 |
| 0.70 | +/-56/255 (22%) | 93% | 0% | 0.93 | 120 |
| 1.00 | +/-80/255 (31%) | 100% | 0% | 1.00 | 114 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 85 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 78 |
| 0.20 | 112px effective (2x downscale) | 87% | 0% | 0.87 | 128 |
| 0.30 | 74px effective (3x downscale) | 73% | 0% | 0.73 | 157 |
| 0.50 | 56px effective (4x downscale) | 20% | 0% | 0.20 | 201 |
| 0.70 | 44px effective (5x downscale) | 53% | 0% | 0.53 | 181 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 220 |

### Task 2: *pick up the black bowl from table center and place it on the plate*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.70 (140ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.70 (70% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 100%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 100% baseline, max deg 100%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 100% baseline, max deg 33%, robust
- **ImageNoiseStressor**: 100% baseline, max deg 0%, robust
- **OcclusionStressor**: 100% baseline, max deg 20%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 7%, robust
- **ResolutionStressor**: 100% baseline, max deg 93%, bp=0.70 (44px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 95 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 106 |
| 0.20 | 40ms (2 steps) | 100% | 0% | 1.00 | 108 |
| 0.30 | 60ms (3 steps) | 100% | 0% | 1.00 | 124 |
| 0.50 | 100ms (5 steps) | 93% | 0% | 0.93 | 159 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 220 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 220 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 98 |
| 0.10 | 10% drop prob | 100% | 0% | 1.00 | 105 |
| 0.20 | 20% drop prob | 100% | 0% | 1.00 | 118 |
| 0.30 | 30% drop prob | 87% | 0% | 0.87 | 145 |
| 0.50 | 50% drop prob | 53% | 0% | 0.53 | 189 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 220 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 220 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 94 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 220 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 220 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 220 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 220 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 220 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 220 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 97 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 220 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 220 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 220 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 220 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 220 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 220 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 93 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 94 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 96 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 100% | 0% | 1.00 | 96 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 100% | 0% | 1.00 | 99 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 93% | 0% | 0.93 | 110 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 67% | 0% | 0.67 | 160 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 97 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 111 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 111 |
| 0.30 | std=15/255 (6%) | 100% | 0% | 1.00 | 111 |
| 0.50 | std=25/255 (10%) | 100% | 0% | 1.00 | 113 |
| 0.70 | std=35/255 (14%) | 100% | 0% | 1.00 | 116 |
| 1.00 | std=50/255 (20%) | 100% | 0% | 1.00 | 131 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 95 |
| 0.10 | 1 patches, up to 3% each | 93% | 0% | 0.93 | 120 |
| 0.20 | 1 patches, up to 6% each | 100% | 0% | 1.00 | 113 |
| 0.30 | 1 patches, up to 9% each | 100% | 0% | 1.00 | 116 |
| 0.50 | 2 patches, up to 15% each | 100% | 0% | 1.00 | 119 |
| 0.70 | 3 patches, up to 21% each | 100% | 0% | 1.00 | 117 |
| 1.00 | 5 patches, up to 30% each | 80% | 0% | 0.80 | 146 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 94 |
| 0.10 | +/-8/255 (3%) | 100% | 0% | 1.00 | 123 |
| 0.20 | +/-16/255 (6%) | 100% | 0% | 1.00 | 117 |
| 0.30 | +/-24/255 (9%) | 100% | 0% | 1.00 | 114 |
| 0.50 | +/-40/255 (16%) | 93% | 0% | 0.93 | 119 |
| 0.70 | +/-56/255 (22%) | 100% | 0% | 1.00 | 116 |
| 1.00 | +/-80/255 (31%) | 100% | 0% | 1.00 | 115 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 97 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 95 |
| 0.20 | 112px effective (2x downscale) | 100% | 0% | 1.00 | 123 |
| 0.30 | 74px effective (3x downscale) | 100% | 0% | 1.00 | 135 |
| 0.50 | 56px effective (4x downscale) | 67% | 0% | 0.67 | 169 |
| 0.70 | 44px effective (5x downscale) | 47% | 0% | 0.47 | 200 |
| 1.00 | 28px effective (8x downscale) | 7% | 0% | 0.07 | 218 |

### Task 4: *pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate*

- **LatencyStressor**: 93% baseline, max deg 93%, bp=0.50 (100ms)
- **DropoutStressor**: 93% baseline, max deg 93%, bp=0.30 (30% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 100%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 93% baseline, max deg 93%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 100% baseline, max deg 47%, robust
- **ImageNoiseStressor**: 87% baseline, max deg 80%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 100% baseline, max deg 13%, robust
- **BrightnessShiftStressor**: 93% baseline, max deg 7%, robust
- **ResolutionStressor**: 93% baseline, max deg 93%, bp=0.50 (56px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 93% | 0% | 0.93 | 128 |
| 0.10 | 20ms (1 steps) | 93% | 0% | 0.93 | 140 |
| 0.20 | 40ms (2 steps) | 93% | 0% | 0.93 | 150 |
| 0.30 | 60ms (3 steps) | 87% | 0% | 0.87 | 160 |
| 0.50 | 100ms (5 steps) | 13% | 0% | 0.13 | 214 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 220 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 220 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 93% | 0% | 0.93 | 135 |
| 0.10 | 10% drop prob | 87% | 0% | 0.87 | 142 |
| 0.20 | 20% drop prob | 60% | 0% | 0.60 | 181 |
| 0.30 | 30% drop prob | 27% | 0% | 0.27 | 198 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 220 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 220 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 220 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 136 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 220 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 220 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 220 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 220 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 220 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 220 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 93% | 0% | 0.93 | 137 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 220 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 220 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 220 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 220 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 220 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 220 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 123 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 138 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 93% | 0% | 0.93 | 133 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 100% | 0% | 1.00 | 131 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 80% | 0% | 0.80 | 155 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 53% | 0% | 0.53 | 170 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 53% | 0% | 0.53 | 178 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 87% | 0% | 0.87 | 142 |
| 0.10 | std=5/255 (2%) | 73% | 0% | 0.73 | 164 |
| 0.20 | std=10/255 (4%) | 87% | 0% | 0.87 | 160 |
| 0.30 | std=15/255 (6%) | 93% | 0% | 0.93 | 161 |
| 0.50 | std=25/255 (10%) | 80% | 0% | 0.80 | 165 |
| 0.70 | std=35/255 (14%) | 87% | 0% | 0.87 | 169 |
| 1.00 | std=50/255 (20%) | 7% | 0% | 0.07 | 220 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 122 |
| 0.10 | 1 patches, up to 3% each | 87% | 0% | 0.87 | 158 |
| 0.20 | 1 patches, up to 6% each | 100% | 0% | 1.00 | 144 |
| 0.30 | 1 patches, up to 9% each | 93% | 0% | 0.93 | 151 |
| 0.50 | 2 patches, up to 15% each | 100% | 0% | 1.00 | 144 |
| 0.70 | 3 patches, up to 21% each | 100% | 0% | 1.00 | 146 |
| 1.00 | 5 patches, up to 30% each | 93% | 0% | 0.93 | 163 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 93% | 0% | 0.93 | 135 |
| 0.10 | +/-8/255 (3%) | 100% | 0% | 1.00 | 142 |
| 0.20 | +/-16/255 (6%) | 87% | 0% | 0.87 | 157 |
| 0.30 | +/-24/255 (9%) | 93% | 0% | 0.93 | 148 |
| 0.50 | +/-40/255 (16%) | 87% | 0% | 0.87 | 157 |
| 0.70 | +/-56/255 (22%) | 93% | 0% | 0.93 | 150 |
| 1.00 | +/-80/255 (31%) | 93% | 0% | 0.93 | 149 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 93% | 0% | 0.93 | 127 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 126 |
| 0.20 | 112px effective (2x downscale) | 93% | 0% | 0.93 | 155 |
| 0.30 | 74px effective (3x downscale) | 67% | 0% | 0.67 | 187 |
| 0.50 | 56px effective (4x downscale) | 0% | 0% | 0.00 | 220 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 220 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 220 |

### Task 7: *pick up the black bowl on the stove and place it on the plate*

- **LatencyStressor**: 100% baseline, max deg 87%, bp=0.70 (140ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.50 (50% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 100%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 100% baseline, max deg 100%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 100% baseline, max deg 33%, robust
- **ImageNoiseStressor**: 100% baseline, max deg 7%, robust
- **OcclusionStressor**: 100% baseline, max deg 7%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 7%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.50 (56px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 114 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 119 |
| 0.20 | 40ms (2 steps) | 100% | 0% | 1.00 | 134 |
| 0.30 | 60ms (3 steps) | 93% | 0% | 0.93 | 145 |
| 0.50 | 100ms (5 steps) | 100% | 0% | 1.00 | 162 |
| 0.70 | 140ms (7 steps) | 40% | 0% | 0.40 | 209 |
| 1.00 | 200ms (10 steps) | 13% | 0% | 0.13 | 220 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 113 |
| 0.10 | 10% drop prob | 100% | 0% | 1.00 | 122 |
| 0.20 | 20% drop prob | 100% | 0% | 1.00 | 143 |
| 0.30 | 30% drop prob | 80% | 0% | 0.80 | 162 |
| 0.50 | 50% drop prob | 27% | 0% | 0.27 | 213 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 220 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 220 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 112 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 220 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 220 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 220 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 220 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 220 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 220 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 117 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 220 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 220 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 220 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 220 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 220 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 220 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 113 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 116 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 114 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 100% | 0% | 1.00 | 115 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 93% | 0% | 0.93 | 125 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 67% | 0% | 0.67 | 154 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 73% | 0% | 0.73 | 158 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 114 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 130 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 133 |
| 0.30 | std=15/255 (6%) | 100% | 0% | 1.00 | 132 |
| 0.50 | std=25/255 (10%) | 100% | 0% | 1.00 | 134 |
| 0.70 | std=35/255 (14%) | 100% | 0% | 1.00 | 139 |
| 1.00 | std=50/255 (20%) | 93% | 0% | 0.93 | 151 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 113 |
| 0.10 | 1 patches, up to 3% each | 100% | 0% | 1.00 | 142 |
| 0.20 | 1 patches, up to 6% each | 93% | 0% | 0.93 | 140 |
| 0.30 | 1 patches, up to 9% each | 100% | 0% | 1.00 | 138 |
| 0.50 | 2 patches, up to 15% each | 100% | 0% | 1.00 | 145 |
| 0.70 | 3 patches, up to 21% each | 93% | 0% | 0.93 | 146 |
| 1.00 | 5 patches, up to 30% each | 100% | 0% | 1.00 | 154 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 114 |
| 0.10 | +/-8/255 (3%) | 100% | 0% | 1.00 | 137 |
| 0.20 | +/-16/255 (6%) | 100% | 0% | 1.00 | 135 |
| 0.30 | +/-24/255 (9%) | 93% | 0% | 0.93 | 143 |
| 0.50 | +/-40/255 (16%) | 100% | 0% | 1.00 | 143 |
| 0.70 | +/-56/255 (22%) | 100% | 0% | 1.00 | 135 |
| 1.00 | +/-80/255 (31%) | 100% | 0% | 1.00 | 139 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 114 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 113 |
| 0.20 | 112px effective (2x downscale) | 100% | 0% | 1.00 | 141 |
| 0.30 | 74px effective (3x downscale) | 100% | 0% | 1.00 | 148 |
| 0.50 | 56px effective (4x downscale) | 40% | 0% | 0.40 | 202 |
| 0.70 | 44px effective (5x downscale) | 27% | 0% | 0.27 | 214 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 220 |

### Task 9: *pick up the black bowl on the wooden cabinet and place it on the plate*

- **LatencyStressor**: 93% baseline, max deg 93%, bp=0.50 (100ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.30 (30% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 100%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 100% baseline, max deg 100%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 100% baseline, max deg 67%, bp=1.00 (obs std 1.0 @step100)
- **ImageNoiseStressor**: 93% baseline, max deg 93%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 100% baseline, max deg 27%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 27%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.30 (74px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 93% | 0% | 0.93 | 124 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 126 |
| 0.20 | 40ms (2 steps) | 100% | 0% | 1.00 | 136 |
| 0.30 | 60ms (3 steps) | 80% | 0% | 0.80 | 165 |
| 0.50 | 100ms (5 steps) | 7% | 0% | 0.07 | 217 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 220 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 220 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 115 |
| 0.10 | 10% drop prob | 73% | 0% | 0.73 | 163 |
| 0.20 | 20% drop prob | 67% | 0% | 0.67 | 178 |
| 0.30 | 30% drop prob | 33% | 0% | 0.33 | 200 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 220 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 220 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 220 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 121 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 220 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 220 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 220 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 220 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 220 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 220 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 120 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 220 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 220 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 220 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 220 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 220 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 220 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 115 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 120 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 124 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 93% | 0% | 0.93 | 134 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 80% | 0% | 0.80 | 145 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 53% | 0% | 0.53 | 166 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 33% | 0% | 0.33 | 188 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 93% | 0% | 0.93 | 125 |
| 0.10 | std=5/255 (2%) | 93% | 0% | 0.93 | 147 |
| 0.20 | std=10/255 (4%) | 87% | 0% | 0.87 | 152 |
| 0.30 | std=15/255 (6%) | 93% | 0% | 0.93 | 142 |
| 0.50 | std=25/255 (10%) | 80% | 0% | 0.80 | 163 |
| 0.70 | std=35/255 (14%) | 53% | 0% | 0.53 | 189 |
| 1.00 | std=50/255 (20%) | 0% | 0% | 0.00 | 220 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 127 |
| 0.10 | 1 patches, up to 3% each | 80% | 0% | 0.80 | 157 |
| 0.20 | 1 patches, up to 6% each | 73% | 0% | 0.73 | 160 |
| 0.30 | 1 patches, up to 9% each | 73% | 0% | 0.73 | 158 |
| 0.50 | 2 patches, up to 15% each | 80% | 0% | 0.80 | 158 |
| 0.70 | 3 patches, up to 21% each | 100% | 0% | 1.00 | 146 |
| 1.00 | 5 patches, up to 30% each | 73% | 0% | 0.73 | 175 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 116 |
| 0.10 | +/-8/255 (3%) | 100% | 0% | 1.00 | 141 |
| 0.20 | +/-16/255 (6%) | 73% | 0% | 0.73 | 167 |
| 0.30 | +/-24/255 (9%) | 80% | 0% | 0.80 | 154 |
| 0.50 | +/-40/255 (16%) | 80% | 0% | 0.80 | 156 |
| 0.70 | +/-56/255 (22%) | 100% | 0% | 1.00 | 137 |
| 1.00 | +/-80/255 (31%) | 80% | 0% | 0.80 | 157 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 118 |
| 0.10 | 224px (native) | 93% | 0% | 0.93 | 124 |
| 0.20 | 112px effective (2x downscale) | 80% | 0% | 0.80 | 172 |
| 0.30 | 74px effective (3x downscale) | 33% | 0% | 0.33 | 208 |
| 0.50 | 56px effective (4x downscale) | 7% | 0% | 0.07 | 218 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 220 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 220 |


## Breakpoint Comparison

Average intensity at which success rate drops below 50%:

- **LatencyStressor**: libero_spatial: 0.54
- **DropoutStressor**: libero_spatial: 0.50
- **PhysicsShiftStressor**: libero_spatial: 0.10
- **EmbodimentStressor**: libero_spatial: 0.10
- **LongHorizonDriftStressor**: libero_spatial: 1.00
- **ImageNoiseStressor**: libero_spatial: 1.00
- **OcclusionStressor**: libero_spatial: robust
- **BrightnessShiftStressor**: libero_spatial: robust
- **ResolutionStressor**: libero_spatial: 0.50

---

*Report generated by Trace Robotics v0.1.0*
*https://tracerobotics.com*