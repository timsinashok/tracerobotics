# Trace Robotics — Multi-Task Robustness Report

**Policy:** OpenVLAAdapter  
**Suites:** libero_spatial, libero_object, libero_goal, libero_10  
**Task IDs:** [0, 4, 9]  
**Total tasks evaluated:** 10  
**Modalities:** vision, proprioception  
**Generated:** 2026-03-18 11:06  
**Control frequency:** 50Hz (20ms per step)

---

## Cross-Suite Summary

| Stressor | libero_spatial | libero_object | libero_goal | libero_10 |
|---|---|---|---|---|
| LatencyStressor | 98% (bp=0.37) | 0% (bp=0.00) | 0% (bp=0.00) | 0% (bp=0.00) |
| DropoutStressor | 100% (bp=0.23) | 0% (bp=0.00) | 0% (bp=0.00) | 0% (bp=0.00) |
| PhysicsShiftStressor | 100% (robust) | 0% (bp=0.00) | 0% (bp=0.00) | 0% (bp=0.00) |
| EmbodimentStressor | 98% (robust) | 0% (bp=0.00) | 0% (bp=0.00) | 0% (bp=0.00) |
| LongHorizonDriftStressor | 100% (bp=0.13) | 0% (bp=0.00) | 0% (bp=0.00) | 0% (bp=0.00) |
| ImageNoiseStressor | 98% (robust) | 0% (bp=0.00) | 0% (bp=0.00) | 0% (bp=0.00) |
| OcclusionStressor | 98% (robust) | 0% (bp=0.00) | 0% (bp=0.00) | 0% (bp=0.00) |
| BrightnessShiftStressor | 98% (robust) | 0% (bp=0.00) | 0% (bp=0.00) | 0% (bp=0.00) |
| ResolutionStressor | 100% (bp=1.00) | 0% (bp=0.00) | 0% (bp=0.00) | 0% (bp=0.00) |

## libero_spatial

### Task 0: *pick up the black bowl between the plate and the ramekin and place it on the plate*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.30 (60ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.30 (30% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 7%, robust
- **EmbodimentStressor**: 100% baseline, max deg 0%, robust
- **LongHorizonDriftStressor**: 100% baseline, max deg 100%, bp=0.20 (obs std 0.2 @step100)
- **ImageNoiseStressor**: 93% baseline, max deg 0%, robust
- **OcclusionStressor**: 100% baseline, max deg 0%, robust
- **BrightnessShiftStressor**: 93% baseline, max deg 0%, robust
- **ResolutionStressor**: 100% baseline, max deg 47%, robust

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 74 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 81 |
| 0.20 | 40ms (2 steps) | 80% | 0% | 0.80 | 114 |
| 0.30 | 60ms (3 steps) | 7% | 0% | 0.07 | 212 |
| 0.50 | 100ms (5 steps) | 0% | 0% | 0.00 | 220 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 220 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 220 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 75 |
| 0.10 | 10% drop prob | 87% | 0% | 0.87 | 98 |
| 0.20 | 20% drop prob | 60% | 0% | 0.60 | 140 |
| 0.30 | 30% drop prob | 20% | 0% | 0.20 | 196 |
| 0.50 | 50% drop prob | 7% | 0% | 0.07 | 218 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 220 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 220 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 74 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 100% | 0% | 1.00 | 75 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 93% | 0% | 0.93 | 85 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 100% | 0% | 1.00 | 75 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 93% | 0% | 0.93 | 85 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 100% | 0% | 1.00 | 79 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 93% | 0% | 0.93 | 85 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 74 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 100% | 0% | 1.00 | 75 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 100% | 0% | 1.00 | 75 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 100% | 0% | 1.00 | 75 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 100% | 0% | 1.00 | 75 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 100% | 0% | 1.00 | 75 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 100% | 0% | 1.00 | 74 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 75 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 80% | 0% | 0.80 | 110 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 33% | 0% | 0.33 | 181 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 60% | 0% | 0.60 | 146 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 27% | 0% | 0.27 | 185 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 7% | 0% | 0.07 | 214 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 0% | 0% | 0.00 | 220 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 93% | 0% | 0.93 | 84 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 75 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 75 |
| 0.30 | std=15/255 (6%) | 100% | 0% | 1.00 | 75 |
| 0.50 | std=25/255 (10%) | 100% | 0% | 1.00 | 74 |
| 0.70 | std=35/255 (14%) | 100% | 0% | 1.00 | 75 |
| 1.00 | std=50/255 (20%) | 100% | 0% | 1.00 | 75 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 74 |
| 0.10 | 1 patches, up to 3% each | 100% | 0% | 1.00 | 74 |
| 0.20 | 1 patches, up to 6% each | 100% | 0% | 1.00 | 75 |
| 0.30 | 1 patches, up to 9% each | 100% | 0% | 1.00 | 75 |
| 0.50 | 2 patches, up to 15% each | 100% | 0% | 1.00 | 75 |
| 0.70 | 3 patches, up to 21% each | 100% | 0% | 1.00 | 74 |
| 1.00 | 5 patches, up to 30% each | 100% | 0% | 1.00 | 76 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 93% | 0% | 0.93 | 84 |
| 0.10 | +/-8/255 (3%) | 100% | 0% | 1.00 | 74 |
| 0.20 | +/-16/255 (6%) | 100% | 0% | 1.00 | 74 |
| 0.30 | +/-24/255 (9%) | 100% | 0% | 1.00 | 75 |
| 0.50 | +/-40/255 (16%) | 100% | 0% | 1.00 | 74 |
| 0.70 | +/-56/255 (22%) | 100% | 0% | 1.00 | 75 |
| 1.00 | +/-80/255 (31%) | 100% | 0% | 1.00 | 75 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 74 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 74 |
| 0.20 | 112px effective (2x downscale) | 100% | 0% | 1.00 | 75 |
| 0.30 | 74px effective (3x downscale) | 100% | 0% | 1.00 | 76 |
| 0.50 | 56px effective (4x downscale) | 100% | 0% | 1.00 | 76 |
| 0.70 | 44px effective (5x downscale) | 100% | 0% | 1.00 | 78 |
| 1.00 | 28px effective (8x downscale) | 53% | 0% | 0.53 | 142 |

### Task 4: *pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate*

- **LatencyStressor**: 93% baseline, max deg 93%, bp=0.50 (100ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.20 (20% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 20%, robust
- **EmbodimentStressor**: 93% baseline, max deg 27%, robust
- **LongHorizonDriftStressor**: 100% baseline, max deg 100%, bp=0.10 (obs std 0.1 @step100)
- **ImageNoiseStressor**: 100% baseline, max deg 13%, robust
- **OcclusionStressor**: 93% baseline, max deg 0%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 0%, robust
- **ResolutionStressor**: 100% baseline, max deg 47%, robust

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 93% | 0% | 0.93 | 130 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 134 |
| 0.20 | 40ms (2 steps) | 93% | 0% | 0.93 | 152 |
| 0.30 | 60ms (3 steps) | 53% | 0% | 0.53 | 187 |
| 0.50 | 100ms (5 steps) | 0% | 0% | 0.00 | 220 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 220 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 220 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 124 |
| 0.10 | 10% drop prob | 80% | 0% | 0.80 | 149 |
| 0.20 | 20% drop prob | 27% | 0% | 0.27 | 199 |
| 0.30 | 30% drop prob | 13% | 0% | 0.13 | 214 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 220 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 220 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 220 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 124 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 93% | 0% | 0.93 | 131 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 93% | 0% | 0.93 | 132 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 100% | 0% | 1.00 | 125 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 100% | 0% | 1.00 | 122 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 100% | 0% | 1.00 | 123 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 80% | 0% | 0.80 | 144 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 93% | 0% | 0.93 | 130 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 100% | 0% | 1.00 | 123 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 100% | 0% | 1.00 | 124 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 100% | 0% | 1.00 | 122 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 100% | 0% | 1.00 | 122 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 93% | 0% | 0.93 | 129 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 67% | 0% | 0.67 | 160 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 123 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 13% | 0% | 0.13 | 209 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 0% | 0% | 0.00 | 220 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 0% | 0% | 0.00 | 220 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 0% | 0% | 0.00 | 220 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 0% | 0% | 0.00 | 220 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 7% | 0% | 0.07 | 219 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 124 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 124 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 123 |
| 0.30 | std=15/255 (6%) | 100% | 0% | 1.00 | 122 |
| 0.50 | std=25/255 (10%) | 100% | 0% | 1.00 | 123 |
| 0.70 | std=35/255 (14%) | 100% | 0% | 1.00 | 124 |
| 1.00 | std=50/255 (20%) | 87% | 0% | 0.87 | 137 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 93% | 0% | 0.93 | 131 |
| 0.10 | 1 patches, up to 3% each | 100% | 0% | 1.00 | 126 |
| 0.20 | 1 patches, up to 6% each | 100% | 0% | 1.00 | 124 |
| 0.30 | 1 patches, up to 9% each | 100% | 0% | 1.00 | 124 |
| 0.50 | 2 patches, up to 15% each | 100% | 0% | 1.00 | 124 |
| 0.70 | 3 patches, up to 21% each | 93% | 0% | 0.93 | 130 |
| 1.00 | 5 patches, up to 30% each | 100% | 0% | 1.00 | 124 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 125 |
| 0.10 | +/-8/255 (3%) | 100% | 0% | 1.00 | 124 |
| 0.20 | +/-16/255 (6%) | 100% | 0% | 1.00 | 124 |
| 0.30 | +/-24/255 (9%) | 100% | 0% | 1.00 | 126 |
| 0.50 | +/-40/255 (16%) | 100% | 0% | 1.00 | 123 |
| 0.70 | +/-56/255 (22%) | 100% | 0% | 1.00 | 124 |
| 1.00 | +/-80/255 (31%) | 100% | 0% | 1.00 | 124 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 124 |
| 0.10 | 224px (native) | 93% | 0% | 0.93 | 131 |
| 0.20 | 112px effective (2x downscale) | 100% | 0% | 1.00 | 123 |
| 0.30 | 74px effective (3x downscale) | 100% | 0% | 1.00 | 126 |
| 0.50 | 56px effective (4x downscale) | 100% | 0% | 1.00 | 127 |
| 0.70 | 44px effective (5x downscale) | 100% | 0% | 1.00 | 124 |
| 1.00 | 28px effective (8x downscale) | 53% | 0% | 0.53 | 168 |

### Task 9: *pick up the black bowl on the wooden cabinet and place it on the plate*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.30 (60ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.20 (20% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 13%, robust
- **EmbodimentStressor**: 100% baseline, max deg 7%, robust
- **LongHorizonDriftStressor**: 100% baseline, max deg 100%, bp=0.10 (obs std 0.1 @step100)
- **ImageNoiseStressor**: 100% baseline, max deg 7%, robust
- **OcclusionStressor**: 100% baseline, max deg 7%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 0%, robust
- **ResolutionStressor**: 100% baseline, max deg 93%, bp=1.00 (28px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 116 |
| 0.10 | 20ms (1 steps) | 93% | 0% | 0.93 | 128 |
| 0.20 | 40ms (2 steps) | 87% | 0% | 0.87 | 162 |
| 0.30 | 60ms (3 steps) | 47% | 0% | 0.47 | 204 |
| 0.50 | 100ms (5 steps) | 0% | 0% | 0.00 | 220 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 220 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 220 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 116 |
| 0.10 | 10% drop prob | 60% | 0% | 0.60 | 164 |
| 0.20 | 20% drop prob | 33% | 0% | 0.33 | 193 |
| 0.30 | 30% drop prob | 13% | 0% | 0.13 | 216 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 220 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 220 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 220 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 116 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 100% | 0% | 1.00 | 116 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 100% | 0% | 1.00 | 116 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 93% | 0% | 0.93 | 123 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 100% | 0% | 1.00 | 115 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 100% | 0% | 1.00 | 116 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 87% | 0% | 0.87 | 130 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 116 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 93% | 0% | 0.93 | 123 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 100% | 0% | 1.00 | 115 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 93% | 0% | 0.93 | 122 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 100% | 0% | 1.00 | 115 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 100% | 0% | 1.00 | 114 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 93% | 0% | 0.93 | 122 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 116 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 7% | 0% | 0.07 | 213 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 7% | 0% | 0.07 | 215 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 7% | 0% | 0.07 | 216 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 0% | 0% | 0.00 | 220 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 0% | 0% | 0.00 | 220 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 0% | 0% | 0.00 | 220 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 116 |
| 0.10 | std=5/255 (2%) | 93% | 0% | 0.93 | 123 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 117 |
| 0.30 | std=15/255 (6%) | 100% | 0% | 1.00 | 118 |
| 0.50 | std=25/255 (10%) | 93% | 0% | 0.93 | 126 |
| 0.70 | std=35/255 (14%) | 100% | 0% | 1.00 | 121 |
| 1.00 | std=50/255 (20%) | 100% | 0% | 1.00 | 126 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 116 |
| 0.10 | 1 patches, up to 3% each | 100% | 0% | 1.00 | 116 |
| 0.20 | 1 patches, up to 6% each | 93% | 0% | 0.93 | 123 |
| 0.30 | 1 patches, up to 9% each | 100% | 0% | 1.00 | 116 |
| 0.50 | 2 patches, up to 15% each | 93% | 0% | 0.93 | 123 |
| 0.70 | 3 patches, up to 21% each | 100% | 0% | 1.00 | 117 |
| 1.00 | 5 patches, up to 30% each | 100% | 0% | 1.00 | 125 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 117 |
| 0.10 | +/-8/255 (3%) | 100% | 0% | 1.00 | 116 |
| 0.20 | +/-16/255 (6%) | 100% | 0% | 1.00 | 116 |
| 0.30 | +/-24/255 (9%) | 100% | 0% | 1.00 | 116 |
| 0.50 | +/-40/255 (16%) | 100% | 0% | 1.00 | 116 |
| 0.70 | +/-56/255 (22%) | 100% | 0% | 1.00 | 115 |
| 1.00 | +/-80/255 (31%) | 100% | 0% | 1.00 | 116 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 116 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 116 |
| 0.20 | 112px effective (2x downscale) | 100% | 0% | 1.00 | 117 |
| 0.30 | 74px effective (3x downscale) | 100% | 0% | 1.00 | 117 |
| 0.50 | 56px effective (4x downscale) | 100% | 0% | 1.00 | 118 |
| 0.70 | 44px effective (5x downscale) | 100% | 0% | 1.00 | 120 |
| 1.00 | 28px effective (8x downscale) | 7% | 0% | 0.07 | 214 |


## libero_object

### Task 0: *pick up the alphabet soup and place it in the basket*

- **LatencyStressor**: 0% baseline, max deg 0%, bp=0.00 (0ms)
- **DropoutStressor**: 0% baseline, max deg 0%, bp=0.00 (0% dropout)
- **PhysicsShiftStressor**: 0% baseline, max deg 0%, bp=0.00 (nominal)
- **EmbodimentStressor**: 0% baseline, max deg 0%, bp=0.00 (nominal)
- **LongHorizonDriftStressor**: 0% baseline, max deg 0%, bp=0.00 (obs std 0.0 @step100)
- **ImageNoiseStressor**: 0% baseline, max deg 0%, bp=0.00 (noise std=0)
- **OcclusionStressor**: 0% baseline, max deg 0%, bp=0.00 (none)
- **BrightnessShiftStressor**: 0% baseline, max deg 0%, bp=0.00 (+/-0 px)
- **ResolutionStressor**: 0% baseline, max deg 0%, bp=0.00 (224px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 0% | 0% | 0.00 | 280 |
| 0.10 | 20ms (1 steps) | 0% | 0% | 0.00 | 280 |
| 0.20 | 40ms (2 steps) | 0% | 0% | 0.00 | 280 |
| 0.30 | 60ms (3 steps) | 0% | 0% | 0.00 | 280 |
| 0.50 | 100ms (5 steps) | 0% | 0% | 0.00 | 280 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 280 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 280 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 0% | 0% | 0.00 | 280 |
| 0.10 | 10% drop prob | 0% | 0% | 0.00 | 280 |
| 0.20 | 20% drop prob | 0% | 0% | 0.00 | 280 |
| 0.30 | 30% drop prob | 0% | 0% | 0.00 | 280 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 280 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 280 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 280 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 0% | 0% | 0.00 | 280 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 280 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 280 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 280 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 280 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 280 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 280 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 0% | 0% | 0.00 | 280 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 280 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 280 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 280 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 280 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 280 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 280 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 0% | 0% | 0.00 | 280 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 0% | 0% | 0.00 | 280 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 0% | 0% | 0.00 | 280 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 0% | 0% | 0.00 | 280 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 0% | 0% | 0.00 | 280 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 0% | 0% | 0.00 | 280 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 0% | 0% | 0.00 | 280 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 0% | 0% | 0.00 | 280 |
| 0.10 | std=5/255 (2%) | 0% | 0% | 0.00 | 280 |
| 0.20 | std=10/255 (4%) | 0% | 0% | 0.00 | 280 |
| 0.30 | std=15/255 (6%) | 0% | 0% | 0.00 | 280 |
| 0.50 | std=25/255 (10%) | 0% | 0% | 0.00 | 280 |
| 0.70 | std=35/255 (14%) | 0% | 0% | 0.00 | 280 |
| 1.00 | std=50/255 (20%) | 0% | 0% | 0.00 | 280 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 0% | 0% | 0.00 | 280 |
| 0.10 | 1 patches, up to 3% each | 0% | 0% | 0.00 | 280 |
| 0.20 | 1 patches, up to 6% each | 0% | 0% | 0.00 | 280 |
| 0.30 | 1 patches, up to 9% each | 0% | 0% | 0.00 | 280 |
| 0.50 | 2 patches, up to 15% each | 0% | 0% | 0.00 | 280 |
| 0.70 | 3 patches, up to 21% each | 0% | 0% | 0.00 | 280 |
| 1.00 | 5 patches, up to 30% each | 0% | 0% | 0.00 | 280 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 0% | 0% | 0.00 | 280 |
| 0.10 | +/-8/255 (3%) | 0% | 0% | 0.00 | 280 |
| 0.20 | +/-16/255 (6%) | 0% | 0% | 0.00 | 280 |
| 0.30 | +/-24/255 (9%) | 0% | 0% | 0.00 | 280 |
| 0.50 | +/-40/255 (16%) | 0% | 0% | 0.00 | 280 |
| 0.70 | +/-56/255 (22%) | 0% | 0% | 0.00 | 280 |
| 1.00 | +/-80/255 (31%) | 0% | 0% | 0.00 | 280 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 0% | 0% | 0.00 | 280 |
| 0.10 | 224px (native) | 0% | 0% | 0.00 | 280 |
| 0.20 | 112px effective (2x downscale) | 0% | 0% | 0.00 | 280 |
| 0.30 | 74px effective (3x downscale) | 0% | 0% | 0.00 | 280 |
| 0.50 | 56px effective (4x downscale) | 0% | 0% | 0.00 | 280 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 280 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 280 |

### Task 4: *pick up the ketchup and place it in the basket*

- **LatencyStressor**: 0% baseline, max deg 0%, bp=0.00 (0ms)
- **DropoutStressor**: 0% baseline, max deg 0%, bp=0.00 (0% dropout)
- **PhysicsShiftStressor**: 0% baseline, max deg 0%, bp=0.00 (nominal)
- **EmbodimentStressor**: 0% baseline, max deg 0%, bp=0.00 (nominal)
- **LongHorizonDriftStressor**: 0% baseline, max deg 0%, bp=0.00 (obs std 0.0 @step100)
- **ImageNoiseStressor**: 0% baseline, max deg 0%, bp=0.00 (noise std=0)
- **OcclusionStressor**: 0% baseline, max deg 0%, bp=0.00 (none)
- **BrightnessShiftStressor**: 0% baseline, max deg 0%, bp=0.00 (+/-0 px)
- **ResolutionStressor**: 0% baseline, max deg 0%, bp=0.00 (224px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 0% | 0% | 0.00 | 280 |
| 0.10 | 20ms (1 steps) | 0% | 0% | 0.00 | 280 |
| 0.20 | 40ms (2 steps) | 0% | 0% | 0.00 | 280 |
| 0.30 | 60ms (3 steps) | 0% | 0% | 0.00 | 280 |
| 0.50 | 100ms (5 steps) | 0% | 0% | 0.00 | 280 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 280 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 280 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 0% | 0% | 0.00 | 280 |
| 0.10 | 10% drop prob | 0% | 0% | 0.00 | 280 |
| 0.20 | 20% drop prob | 0% | 0% | 0.00 | 280 |
| 0.30 | 30% drop prob | 0% | 0% | 0.00 | 280 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 280 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 280 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 280 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 0% | 0% | 0.00 | 280 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 280 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 280 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 280 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 280 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 280 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 280 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 0% | 0% | 0.00 | 280 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 280 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 280 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 280 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 280 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 280 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 280 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 0% | 0% | 0.00 | 280 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 0% | 0% | 0.00 | 280 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 0% | 0% | 0.00 | 280 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 0% | 0% | 0.00 | 280 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 0% | 0% | 0.00 | 280 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 0% | 0% | 0.00 | 280 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 0% | 0% | 0.00 | 280 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 0% | 0% | 0.00 | 280 |
| 0.10 | std=5/255 (2%) | 0% | 0% | 0.00 | 280 |
| 0.20 | std=10/255 (4%) | 0% | 0% | 0.00 | 280 |
| 0.30 | std=15/255 (6%) | 0% | 0% | 0.00 | 280 |
| 0.50 | std=25/255 (10%) | 0% | 0% | 0.00 | 280 |
| 0.70 | std=35/255 (14%) | 0% | 0% | 0.00 | 280 |
| 1.00 | std=50/255 (20%) | 0% | 0% | 0.00 | 280 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 0% | 0% | 0.00 | 280 |
| 0.10 | 1 patches, up to 3% each | 0% | 0% | 0.00 | 280 |
| 0.20 | 1 patches, up to 6% each | 0% | 0% | 0.00 | 280 |
| 0.30 | 1 patches, up to 9% each | 0% | 0% | 0.00 | 280 |
| 0.50 | 2 patches, up to 15% each | 0% | 0% | 0.00 | 280 |
| 0.70 | 3 patches, up to 21% each | 0% | 0% | 0.00 | 280 |
| 1.00 | 5 patches, up to 30% each | 0% | 0% | 0.00 | 280 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 0% | 0% | 0.00 | 280 |
| 0.10 | +/-8/255 (3%) | 0% | 0% | 0.00 | 280 |
| 0.20 | +/-16/255 (6%) | 0% | 0% | 0.00 | 280 |
| 0.30 | +/-24/255 (9%) | 0% | 0% | 0.00 | 280 |
| 0.50 | +/-40/255 (16%) | 0% | 0% | 0.00 | 280 |
| 0.70 | +/-56/255 (22%) | 0% | 0% | 0.00 | 280 |
| 1.00 | +/-80/255 (31%) | 0% | 0% | 0.00 | 280 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 0% | 0% | 0.00 | 280 |
| 0.10 | 224px (native) | 0% | 0% | 0.00 | 280 |
| 0.20 | 112px effective (2x downscale) | 0% | 0% | 0.00 | 280 |
| 0.30 | 74px effective (3x downscale) | 0% | 0% | 0.00 | 280 |
| 0.50 | 56px effective (4x downscale) | 0% | 0% | 0.00 | 280 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 280 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 280 |

### Task 9: *pick up the orange juice and place it in the basket*

- **LatencyStressor**: 0% baseline, max deg 0%, bp=0.00 (0ms)
- **DropoutStressor**: 0% baseline, max deg 0%, bp=0.00 (0% dropout)
- **PhysicsShiftStressor**: 0% baseline, max deg 0%, bp=0.00 (nominal)
- **EmbodimentStressor**: 0% baseline, max deg 0%, bp=0.00 (nominal)
- **LongHorizonDriftStressor**: 0% baseline, max deg 0%, bp=0.00 (obs std 0.0 @step100)
- **ImageNoiseStressor**: 0% baseline, max deg 0%, bp=0.00 (noise std=0)
- **OcclusionStressor**: 0% baseline, max deg 0%, bp=0.00 (none)
- **BrightnessShiftStressor**: 0% baseline, max deg 0%, bp=0.00 (+/-0 px)
- **ResolutionStressor**: 0% baseline, max deg 0%, bp=0.00 (224px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 0% | 0% | 0.00 | 280 |
| 0.10 | 20ms (1 steps) | 0% | 0% | 0.00 | 280 |
| 0.20 | 40ms (2 steps) | 0% | 0% | 0.00 | 280 |
| 0.30 | 60ms (3 steps) | 0% | 0% | 0.00 | 280 |
| 0.50 | 100ms (5 steps) | 0% | 0% | 0.00 | 280 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 280 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 280 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 0% | 0% | 0.00 | 280 |
| 0.10 | 10% drop prob | 0% | 0% | 0.00 | 280 |
| 0.20 | 20% drop prob | 0% | 0% | 0.00 | 280 |
| 0.30 | 30% drop prob | 0% | 0% | 0.00 | 280 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 280 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 280 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 280 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 0% | 0% | 0.00 | 280 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 280 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 280 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 280 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 280 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 280 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 280 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 0% | 0% | 0.00 | 280 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 280 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 280 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 280 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 280 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 280 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 280 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 0% | 0% | 0.00 | 280 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 0% | 0% | 0.00 | 280 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 0% | 0% | 0.00 | 280 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 0% | 0% | 0.00 | 280 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 0% | 0% | 0.00 | 280 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 0% | 0% | 0.00 | 280 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 0% | 0% | 0.00 | 280 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 0% | 0% | 0.00 | 280 |
| 0.10 | std=5/255 (2%) | 0% | 0% | 0.00 | 280 |
| 0.20 | std=10/255 (4%) | 0% | 0% | 0.00 | 280 |
| 0.30 | std=15/255 (6%) | 0% | 0% | 0.00 | 280 |
| 0.50 | std=25/255 (10%) | 0% | 0% | 0.00 | 280 |
| 0.70 | std=35/255 (14%) | 0% | 0% | 0.00 | 280 |
| 1.00 | std=50/255 (20%) | 0% | 0% | 0.00 | 280 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 0% | 0% | 0.00 | 280 |
| 0.10 | 1 patches, up to 3% each | 0% | 0% | 0.00 | 280 |
| 0.20 | 1 patches, up to 6% each | 0% | 0% | 0.00 | 280 |
| 0.30 | 1 patches, up to 9% each | 0% | 0% | 0.00 | 280 |
| 0.50 | 2 patches, up to 15% each | 0% | 0% | 0.00 | 280 |
| 0.70 | 3 patches, up to 21% each | 0% | 0% | 0.00 | 280 |
| 1.00 | 5 patches, up to 30% each | 0% | 0% | 0.00 | 280 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 0% | 0% | 0.00 | 280 |
| 0.10 | +/-8/255 (3%) | 0% | 0% | 0.00 | 280 |
| 0.20 | +/-16/255 (6%) | 0% | 0% | 0.00 | 280 |
| 0.30 | +/-24/255 (9%) | 0% | 0% | 0.00 | 280 |
| 0.50 | +/-40/255 (16%) | 0% | 0% | 0.00 | 280 |
| 0.70 | +/-56/255 (22%) | 0% | 0% | 0.00 | 280 |
| 1.00 | +/-80/255 (31%) | 0% | 0% | 0.00 | 280 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 0% | 0% | 0.00 | 280 |
| 0.10 | 224px (native) | 0% | 0% | 0.00 | 280 |
| 0.20 | 112px effective (2x downscale) | 0% | 0% | 0.00 | 280 |
| 0.30 | 74px effective (3x downscale) | 0% | 0% | 0.00 | 280 |
| 0.50 | 56px effective (4x downscale) | 0% | 0% | 0.00 | 280 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 280 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 280 |


## libero_goal

### Task 0: *open the middle drawer of the cabinet*

- **LatencyStressor**: 0% baseline, max deg 0%, bp=0.00 (0ms)
- **DropoutStressor**: 0% baseline, max deg 0%, bp=0.00 (0% dropout)
- **PhysicsShiftStressor**: 0% baseline, max deg 0%, bp=0.00 (nominal)
- **EmbodimentStressor**: 0% baseline, max deg 0%, bp=0.00 (nominal)
- **LongHorizonDriftStressor**: 0% baseline, max deg 0%, bp=0.00 (obs std 0.0 @step100)
- **ImageNoiseStressor**: 0% baseline, max deg 0%, bp=0.00 (noise std=0)
- **OcclusionStressor**: 0% baseline, max deg 0%, bp=0.00 (none)
- **BrightnessShiftStressor**: 0% baseline, max deg 0%, bp=0.00 (+/-0 px)
- **ResolutionStressor**: 0% baseline, max deg 0%, bp=0.00 (224px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 0% | 0% | 0.00 | 300 |
| 0.10 | 20ms (1 steps) | 0% | 0% | 0.00 | 300 |
| 0.20 | 40ms (2 steps) | 0% | 0% | 0.00 | 300 |
| 0.30 | 60ms (3 steps) | 0% | 0% | 0.00 | 300 |
| 0.50 | 100ms (5 steps) | 0% | 0% | 0.00 | 300 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 300 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 300 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 0% | 0% | 0.00 | 300 |
| 0.10 | 10% drop prob | 0% | 0% | 0.00 | 300 |
| 0.20 | 20% drop prob | 0% | 0% | 0.00 | 300 |
| 0.30 | 30% drop prob | 0% | 0% | 0.00 | 300 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 300 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 300 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 300 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 0% | 0% | 0.00 | 300 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 300 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 300 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 300 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 300 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 300 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 300 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 0% | 0% | 0.00 | 300 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 300 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 300 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 300 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 300 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 300 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 300 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 0% | 0% | 0.00 | 300 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 0% | 0% | 0.00 | 300 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 0% | 0% | 0.00 | 300 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 0% | 0% | 0.00 | 300 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 0% | 0% | 0.00 | 300 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 0% | 0% | 0.00 | 300 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 0% | 0% | 0.00 | 300 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 0% | 0% | 0.00 | 300 |
| 0.10 | std=5/255 (2%) | 0% | 0% | 0.00 | 300 |
| 0.20 | std=10/255 (4%) | 0% | 0% | 0.00 | 300 |
| 0.30 | std=15/255 (6%) | 0% | 0% | 0.00 | 300 |
| 0.50 | std=25/255 (10%) | 0% | 0% | 0.00 | 300 |
| 0.70 | std=35/255 (14%) | 0% | 0% | 0.00 | 300 |
| 1.00 | std=50/255 (20%) | 0% | 0% | 0.00 | 300 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 0% | 0% | 0.00 | 300 |
| 0.10 | 1 patches, up to 3% each | 0% | 0% | 0.00 | 300 |
| 0.20 | 1 patches, up to 6% each | 0% | 0% | 0.00 | 300 |
| 0.30 | 1 patches, up to 9% each | 0% | 0% | 0.00 | 300 |
| 0.50 | 2 patches, up to 15% each | 0% | 0% | 0.00 | 300 |
| 0.70 | 3 patches, up to 21% each | 0% | 0% | 0.00 | 300 |
| 1.00 | 5 patches, up to 30% each | 0% | 0% | 0.00 | 300 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 0% | 0% | 0.00 | 300 |
| 0.10 | +/-8/255 (3%) | 0% | 0% | 0.00 | 300 |
| 0.20 | +/-16/255 (6%) | 0% | 0% | 0.00 | 300 |
| 0.30 | +/-24/255 (9%) | 0% | 0% | 0.00 | 300 |
| 0.50 | +/-40/255 (16%) | 0% | 0% | 0.00 | 300 |
| 0.70 | +/-56/255 (22%) | 0% | 0% | 0.00 | 300 |
| 1.00 | +/-80/255 (31%) | 0% | 0% | 0.00 | 300 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 0% | 0% | 0.00 | 300 |
| 0.10 | 224px (native) | 0% | 0% | 0.00 | 300 |
| 0.20 | 112px effective (2x downscale) | 0% | 0% | 0.00 | 300 |
| 0.30 | 74px effective (3x downscale) | 0% | 0% | 0.00 | 300 |
| 0.50 | 56px effective (4x downscale) | 0% | 0% | 0.00 | 300 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 300 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 300 |

### Task 4: *put the bowl on top of the cabinet*

- **LatencyStressor**: 0% baseline, max deg 0%, bp=0.00 (0ms)
- **DropoutStressor**: 0% baseline, max deg 0%, bp=0.00 (0% dropout)
- **PhysicsShiftStressor**: 0% baseline, max deg 0%, bp=0.00 (nominal)
- **EmbodimentStressor**: 0% baseline, max deg 0%, bp=0.00 (nominal)
- **LongHorizonDriftStressor**: 0% baseline, max deg 0%, bp=0.00 (obs std 0.0 @step100)
- **ImageNoiseStressor**: 0% baseline, max deg 0%, bp=0.00 (noise std=0)
- **OcclusionStressor**: 0% baseline, max deg 0%, bp=0.00 (none)
- **BrightnessShiftStressor**: 0% baseline, max deg 0%, bp=0.00 (+/-0 px)
- **ResolutionStressor**: 0% baseline, max deg 0%, bp=0.00 (224px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 0% | 0% | 0.00 | 300 |
| 0.10 | 20ms (1 steps) | 0% | 0% | 0.00 | 300 |
| 0.20 | 40ms (2 steps) | 0% | 0% | 0.00 | 300 |
| 0.30 | 60ms (3 steps) | 0% | 0% | 0.00 | 300 |
| 0.50 | 100ms (5 steps) | 0% | 0% | 0.00 | 300 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 300 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 300 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 0% | 0% | 0.00 | 300 |
| 0.10 | 10% drop prob | 0% | 0% | 0.00 | 300 |
| 0.20 | 20% drop prob | 0% | 0% | 0.00 | 300 |
| 0.30 | 30% drop prob | 0% | 0% | 0.00 | 300 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 300 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 300 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 300 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 0% | 0% | 0.00 | 300 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 300 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 300 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 300 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 300 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 300 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 300 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 0% | 0% | 0.00 | 300 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 300 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 300 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 300 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 300 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 300 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 300 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 0% | 0% | 0.00 | 300 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 0% | 0% | 0.00 | 300 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 0% | 0% | 0.00 | 300 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 0% | 0% | 0.00 | 300 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 0% | 0% | 0.00 | 300 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 0% | 0% | 0.00 | 300 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 0% | 0% | 0.00 | 300 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 0% | 0% | 0.00 | 300 |
| 0.10 | std=5/255 (2%) | 0% | 0% | 0.00 | 300 |
| 0.20 | std=10/255 (4%) | 0% | 0% | 0.00 | 300 |
| 0.30 | std=15/255 (6%) | 0% | 0% | 0.00 | 300 |
| 0.50 | std=25/255 (10%) | 0% | 0% | 0.00 | 300 |
| 0.70 | std=35/255 (14%) | 0% | 0% | 0.00 | 300 |
| 1.00 | std=50/255 (20%) | 0% | 0% | 0.00 | 300 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 0% | 0% | 0.00 | 300 |
| 0.10 | 1 patches, up to 3% each | 0% | 0% | 0.00 | 300 |
| 0.20 | 1 patches, up to 6% each | 0% | 0% | 0.00 | 300 |
| 0.30 | 1 patches, up to 9% each | 0% | 0% | 0.00 | 300 |
| 0.50 | 2 patches, up to 15% each | 0% | 0% | 0.00 | 300 |
| 0.70 | 3 patches, up to 21% each | 0% | 0% | 0.00 | 300 |
| 1.00 | 5 patches, up to 30% each | 0% | 0% | 0.00 | 300 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 0% | 0% | 0.00 | 300 |
| 0.10 | +/-8/255 (3%) | 0% | 0% | 0.00 | 300 |
| 0.20 | +/-16/255 (6%) | 0% | 0% | 0.00 | 300 |
| 0.30 | +/-24/255 (9%) | 0% | 0% | 0.00 | 300 |
| 0.50 | +/-40/255 (16%) | 0% | 0% | 0.00 | 300 |
| 0.70 | +/-56/255 (22%) | 0% | 0% | 0.00 | 300 |
| 1.00 | +/-80/255 (31%) | 0% | 0% | 0.00 | 300 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 0% | 0% | 0.00 | 300 |
| 0.10 | 224px (native) | 0% | 0% | 0.00 | 300 |
| 0.20 | 112px effective (2x downscale) | 0% | 0% | 0.00 | 300 |
| 0.30 | 74px effective (3x downscale) | 0% | 0% | 0.00 | 300 |
| 0.50 | 56px effective (4x downscale) | 0% | 0% | 0.00 | 300 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 300 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 300 |

### Task 9: *put the wine bottle on the rack*

- **LatencyStressor**: 0% baseline, max deg 0%, bp=0.00 (0ms)
- **DropoutStressor**: 0% baseline, max deg 0%, bp=0.00 (0% dropout)
- **PhysicsShiftStressor**: 0% baseline, max deg 0%, bp=0.00 (nominal)
- **EmbodimentStressor**: 0% baseline, max deg 0%, bp=0.00 (nominal)
- **LongHorizonDriftStressor**: 0% baseline, max deg 0%, bp=0.00 (obs std 0.0 @step100)
- **ImageNoiseStressor**: 0% baseline, max deg 0%, bp=0.00 (noise std=0)
- **OcclusionStressor**: 0% baseline, max deg 0%, bp=0.00 (none)
- **BrightnessShiftStressor**: 0% baseline, max deg 0%, bp=0.00 (+/-0 px)
- **ResolutionStressor**: 0% baseline, max deg 0%, bp=0.00 (224px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 0% | 0% | 0.00 | 300 |
| 0.10 | 20ms (1 steps) | 0% | 0% | 0.00 | 300 |
| 0.20 | 40ms (2 steps) | 0% | 0% | 0.00 | 300 |
| 0.30 | 60ms (3 steps) | 0% | 0% | 0.00 | 300 |
| 0.50 | 100ms (5 steps) | 0% | 0% | 0.00 | 300 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 300 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 300 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 0% | 0% | 0.00 | 300 |
| 0.10 | 10% drop prob | 0% | 0% | 0.00 | 300 |
| 0.20 | 20% drop prob | 0% | 0% | 0.00 | 300 |
| 0.30 | 30% drop prob | 0% | 0% | 0.00 | 300 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 300 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 300 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 300 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 0% | 0% | 0.00 | 300 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 300 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 300 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 300 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 300 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 300 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 300 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 0% | 0% | 0.00 | 300 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 300 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 300 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 300 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 300 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 300 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 300 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 0% | 0% | 0.00 | 300 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 0% | 0% | 0.00 | 300 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 0% | 0% | 0.00 | 300 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 0% | 0% | 0.00 | 300 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 0% | 0% | 0.00 | 300 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 0% | 0% | 0.00 | 300 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 0% | 0% | 0.00 | 300 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 0% | 0% | 0.00 | 300 |
| 0.10 | std=5/255 (2%) | 0% | 0% | 0.00 | 300 |
| 0.20 | std=10/255 (4%) | 0% | 0% | 0.00 | 300 |
| 0.30 | std=15/255 (6%) | 0% | 0% | 0.00 | 300 |
| 0.50 | std=25/255 (10%) | 0% | 0% | 0.00 | 300 |
| 0.70 | std=35/255 (14%) | 0% | 0% | 0.00 | 300 |
| 1.00 | std=50/255 (20%) | 0% | 0% | 0.00 | 300 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 0% | 0% | 0.00 | 300 |
| 0.10 | 1 patches, up to 3% each | 0% | 0% | 0.00 | 300 |
| 0.20 | 1 patches, up to 6% each | 0% | 0% | 0.00 | 300 |
| 0.30 | 1 patches, up to 9% each | 0% | 0% | 0.00 | 300 |
| 0.50 | 2 patches, up to 15% each | 0% | 0% | 0.00 | 300 |
| 0.70 | 3 patches, up to 21% each | 0% | 0% | 0.00 | 300 |
| 1.00 | 5 patches, up to 30% each | 0% | 0% | 0.00 | 300 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 0% | 0% | 0.00 | 300 |
| 0.10 | +/-8/255 (3%) | 0% | 0% | 0.00 | 300 |
| 0.20 | +/-16/255 (6%) | 0% | 0% | 0.00 | 300 |
| 0.30 | +/-24/255 (9%) | 0% | 0% | 0.00 | 300 |
| 0.50 | +/-40/255 (16%) | 0% | 0% | 0.00 | 300 |
| 0.70 | +/-56/255 (22%) | 0% | 0% | 0.00 | 300 |
| 1.00 | +/-80/255 (31%) | 0% | 0% | 0.00 | 300 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 0% | 0% | 0.00 | 300 |
| 0.10 | 224px (native) | 0% | 0% | 0.00 | 300 |
| 0.20 | 112px effective (2x downscale) | 0% | 0% | 0.00 | 300 |
| 0.30 | 74px effective (3x downscale) | 0% | 0% | 0.00 | 300 |
| 0.50 | 56px effective (4x downscale) | 0% | 0% | 0.00 | 300 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 300 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 300 |


## libero_10

### Task 0: *put both the alphabet soup and the tomato sauce in the basket*

- **LatencyStressor**: 0% baseline, max deg 0%, bp=0.00 (0ms)
- **DropoutStressor**: 0% baseline, max deg 0%, bp=0.00 (0% dropout)
- **PhysicsShiftStressor**: 0% baseline, max deg 0%, bp=0.00 (nominal)
- **EmbodimentStressor**: 0% baseline, max deg 0%, bp=0.00 (nominal)
- **LongHorizonDriftStressor**: 0% baseline, max deg 0%, bp=0.00 (obs std 0.0 @step100)
- **ImageNoiseStressor**: 0% baseline, max deg 0%, bp=0.00 (noise std=0)
- **OcclusionStressor**: 0% baseline, max deg 0%, bp=0.00 (none)
- **BrightnessShiftStressor**: 0% baseline, max deg 0%, bp=0.00 (+/-0 px)
- **ResolutionStressor**: 0% baseline, max deg 0%, bp=0.00 (224px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 0% | 0% | 0.00 | 520 |
| 0.10 | 20ms (1 steps) | 0% | 0% | 0.00 | 520 |
| 0.20 | 40ms (2 steps) | 0% | 0% | 0.00 | 520 |
| 0.30 | 60ms (3 steps) | 0% | 0% | 0.00 | 520 |
| 0.50 | 100ms (5 steps) | 0% | 0% | 0.00 | 520 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 520 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 520 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 0% | 0% | 0.00 | 520 |
| 0.10 | 10% drop prob | 0% | 0% | 0.00 | 520 |
| 0.20 | 20% drop prob | 0% | 0% | 0.00 | 520 |
| 0.30 | 30% drop prob | 0% | 0% | 0.00 | 520 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 520 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 520 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 520 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 0% | 0% | 0.00 | 520 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 520 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 520 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 520 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 520 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 520 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 520 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 0% | 0% | 0.00 | 520 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 520 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 520 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 520 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 520 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 520 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 520 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 0% | 0% | 0.00 | 520 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 0% | 0% | 0.00 | 520 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 0% | 0% | 0.00 | 520 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 0% | 0% | 0.00 | 520 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 0% | 0% | 0.00 | 520 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 0% | 0% | 0.00 | 520 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 0% | 0% | 0.00 | 520 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 0% | 0% | 0.00 | 520 |
| 0.10 | std=5/255 (2%) | 0% | 0% | 0.00 | 520 |
| 0.20 | std=10/255 (4%) | 0% | 0% | 0.00 | 520 |
| 0.30 | std=15/255 (6%) | 0% | 0% | 0.00 | 520 |
| 0.50 | std=25/255 (10%) | 0% | 0% | 0.00 | 520 |
| 0.70 | std=35/255 (14%) | 0% | 0% | 0.00 | 520 |
| 1.00 | std=50/255 (20%) | 0% | 0% | 0.00 | 520 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 0% | 0% | 0.00 | 520 |
| 0.10 | 1 patches, up to 3% each | 0% | 0% | 0.00 | 520 |
| 0.20 | 1 patches, up to 6% each | 0% | 0% | 0.00 | 520 |
| 0.30 | 1 patches, up to 9% each | 0% | 0% | 0.00 | 520 |
| 0.50 | 2 patches, up to 15% each | 0% | 0% | 0.00 | 520 |
| 0.70 | 3 patches, up to 21% each | 0% | 0% | 0.00 | 520 |
| 1.00 | 5 patches, up to 30% each | 0% | 0% | 0.00 | 520 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 0% | 0% | 0.00 | 520 |
| 0.10 | +/-8/255 (3%) | 0% | 0% | 0.00 | 520 |
| 0.20 | +/-16/255 (6%) | 0% | 0% | 0.00 | 520 |
| 0.30 | +/-24/255 (9%) | 0% | 0% | 0.00 | 520 |
| 0.50 | +/-40/255 (16%) | 0% | 0% | 0.00 | 520 |
| 0.70 | +/-56/255 (22%) | 0% | 0% | 0.00 | 520 |
| 1.00 | +/-80/255 (31%) | 0% | 0% | 0.00 | 520 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 0% | 0% | 0.00 | 520 |
| 0.10 | 224px (native) | 0% | 0% | 0.00 | 520 |
| 0.20 | 112px effective (2x downscale) | 0% | 0% | 0.00 | 520 |
| 0.30 | 74px effective (3x downscale) | 0% | 0% | 0.00 | 520 |
| 0.50 | 56px effective (4x downscale) | 0% | 0% | 0.00 | 520 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 520 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 520 |


## Breakpoint Comparison

Average intensity at which success rate drops below 50%:

- **LatencyStressor**: libero_spatial: 0.37, libero_object: 0.00, libero_goal: 0.00, libero_10: 0.00
- **DropoutStressor**: libero_spatial: 0.23, libero_object: 0.00, libero_goal: 0.00, libero_10: 0.00
- **PhysicsShiftStressor**: libero_spatial: robust, libero_object: 0.00, libero_goal: 0.00, libero_10: 0.00
- **EmbodimentStressor**: libero_spatial: robust, libero_object: 0.00, libero_goal: 0.00, libero_10: 0.00
- **LongHorizonDriftStressor**: libero_spatial: 0.13, libero_object: 0.00, libero_goal: 0.00, libero_10: 0.00
- **ImageNoiseStressor**: libero_spatial: robust, libero_object: 0.00, libero_goal: 0.00, libero_10: 0.00
- **OcclusionStressor**: libero_spatial: robust, libero_object: 0.00, libero_goal: 0.00, libero_10: 0.00
- **BrightnessShiftStressor**: libero_spatial: robust, libero_object: 0.00, libero_goal: 0.00, libero_10: 0.00
- **ResolutionStressor**: libero_spatial: 1.00, libero_object: 0.00, libero_goal: 0.00, libero_10: 0.00

---

*Report generated by Trace Robotics v0.1.0*
*https://tracerobotics.com*