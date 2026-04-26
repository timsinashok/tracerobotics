# Trace Robotics — Multi-Task Robustness Report

**Policy:** Pi0PolicyAdapter  
**Suites:** libero_spatial, libero_object, libero_goal, libero_10  
**Task IDs:** [0, 4, 9]  
**Total tasks evaluated:** 10  
**Modalities:** vision, proprioception  
**Generated:** 2026-04-26 18:31  
**Control frequency:** 50Hz (20ms per step)

---

## Cross-Suite Summary

| Stressor | libero_spatial | libero_object | libero_goal | libero_10 |
|---|---|---|---|---|
| LatencyStressor | 100% (bp=0.50) | 100% (bp=0.57) | 93% (bp=0.75) | 100% (bp=0.50) |
| DropoutStressor | 93% (bp=0.43) | 98% (bp=0.50) | 96% (bp=0.40) | 100% (bp=0.30) |
| PhysicsShiftStressor | 98% (bp=0.10) | 98% (bp=0.10) | 98% (bp=0.10) | 100% (robust) |
| EmbodimentStressor | 100% (bp=0.10) | 96% (bp=0.10) | 98% (bp=0.10) | 100% (robust) |
| LongHorizonDriftStressor | 100% (bp=0.85) | 100% (bp=1.00) | 98% (bp=0.60) | 100% (bp=0.30) |
| ImageNoiseStressor | 96% (bp=1.00) | 98% (bp=1.00) | 91% (bp=1.00) | 100% (bp=1.00) |
| OcclusionStressor | 100% (robust) | 98% (robust) | 98% (robust) | 100% (robust) |
| BrightnessShiftStressor | 100% (robust) | 100% (robust) | 98% (robust) | 100% (robust) |
| ResolutionStressor | 98% (bp=0.43) | 100% (bp=0.57) | 98% (bp=0.43) | 100% (bp=0.30) |

## libero_spatial

### Task 0: *pick up the black bowl between the plate and the ramekin and place it on the plate*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.50 (100ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.70 (70% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 100%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 100% baseline, max deg 100%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 100% baseline, max deg 20%, robust
- **ImageNoiseStressor**: 100% baseline, max deg 27%, robust
- **OcclusionStressor**: 100% baseline, max deg 47%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 13%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.50 (56px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 84 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 95 |
| 0.20 | 40ms (2 steps) | 100% | 0% | 1.00 | 104 |
| 0.30 | 60ms (3 steps) | 53% | 0% | 0.53 | 196 |
| 0.50 | 100ms (5 steps) | 0% | 0% | 0.00 | 220 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 220 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 220 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 83 |
| 0.10 | 10% drop prob | 93% | 0% | 0.93 | 95 |
| 0.20 | 20% drop prob | 100% | 0% | 1.00 | 98 |
| 0.30 | 30% drop prob | 93% | 0% | 0.93 | 139 |
| 0.50 | 50% drop prob | 60% | 0% | 0.60 | 175 |
| 0.70 | 70% drop prob | 20% | 0% | 0.20 | 213 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 220 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 87 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 220 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 220 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 220 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 220 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 220 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 220 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 90 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 220 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 220 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 220 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 220 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 220 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 220 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 77 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 83 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 93% | 0% | 0.93 | 89 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 100% | 0% | 1.00 | 77 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 100% | 0% | 1.00 | 83 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 100% | 0% | 1.00 | 84 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 80% | 0% | 0.80 | 121 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 85 |
| 0.10 | std=5/255 (2%) | 87% | 0% | 0.87 | 142 |
| 0.20 | std=10/255 (4%) | 93% | 0% | 0.93 | 133 |
| 0.30 | std=15/255 (6%) | 80% | 0% | 0.80 | 134 |
| 0.50 | std=25/255 (10%) | 93% | 0% | 0.93 | 132 |
| 0.70 | std=35/255 (14%) | 80% | 0% | 0.80 | 127 |
| 1.00 | std=50/255 (20%) | 73% | 0% | 0.73 | 145 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 83 |
| 0.10 | 1 patches, up to 3% each | 93% | 0% | 0.93 | 124 |
| 0.20 | 1 patches, up to 6% each | 100% | 0% | 1.00 | 129 |
| 0.30 | 1 patches, up to 9% each | 87% | 0% | 0.87 | 147 |
| 0.50 | 2 patches, up to 15% each | 80% | 0% | 0.80 | 156 |
| 0.70 | 3 patches, up to 21% each | 93% | 0% | 0.93 | 144 |
| 1.00 | 5 patches, up to 30% each | 53% | 0% | 0.53 | 187 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 83 |
| 0.10 | +/-8/255 (3%) | 93% | 0% | 0.93 | 129 |
| 0.20 | +/-16/255 (6%) | 93% | 0% | 0.93 | 125 |
| 0.30 | +/-24/255 (9%) | 100% | 0% | 1.00 | 124 |
| 0.50 | +/-40/255 (16%) | 87% | 0% | 0.87 | 127 |
| 0.70 | +/-56/255 (22%) | 87% | 0% | 0.87 | 137 |
| 1.00 | +/-80/255 (31%) | 87% | 0% | 0.87 | 144 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 83 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 80 |
| 0.20 | 112px effective (2x downscale) | 80% | 0% | 0.80 | 157 |
| 0.30 | 74px effective (3x downscale) | 67% | 0% | 0.67 | 179 |
| 0.50 | 56px effective (4x downscale) | 33% | 0% | 0.33 | 191 |
| 0.70 | 44px effective (5x downscale) | 53% | 0% | 0.53 | 169 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 220 |

### Task 4: *pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.50 (100ms)
- **DropoutStressor**: 93% baseline, max deg 93%, bp=0.30 (30% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 100%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 100% baseline, max deg 100%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 100% baseline, max deg 53%, bp=0.70 (obs std 0.7 @step100)
- **ImageNoiseStressor**: 87% baseline, max deg 87%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 100% baseline, max deg 27%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 40%, robust
- **ResolutionStressor**: 93% baseline, max deg 93%, bp=0.50 (56px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 127 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 137 |
| 0.20 | 40ms (2 steps) | 93% | 0% | 0.93 | 150 |
| 0.30 | 60ms (3 steps) | 100% | 0% | 1.00 | 157 |
| 0.50 | 100ms (5 steps) | 13% | 0% | 0.13 | 215 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 220 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 220 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 93% | 0% | 0.93 | 130 |
| 0.10 | 10% drop prob | 73% | 0% | 0.73 | 161 |
| 0.20 | 20% drop prob | 53% | 0% | 0.53 | 182 |
| 0.30 | 30% drop prob | 27% | 0% | 0.27 | 202 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 220 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 220 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 220 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 131 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 220 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 220 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 220 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 220 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 220 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 220 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 129 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 220 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 220 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 220 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 220 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 220 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 220 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 128 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 130 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 133 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 100% | 0% | 1.00 | 129 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 73% | 0% | 0.73 | 155 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 47% | 0% | 0.47 | 176 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 53% | 0% | 0.53 | 182 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 87% | 0% | 0.87 | 146 |
| 0.10 | std=5/255 (2%) | 60% | 0% | 0.60 | 180 |
| 0.20 | std=10/255 (4%) | 73% | 0% | 0.73 | 177 |
| 0.30 | std=15/255 (6%) | 93% | 0% | 0.93 | 159 |
| 0.50 | std=25/255 (10%) | 87% | 0% | 0.87 | 166 |
| 0.70 | std=35/255 (14%) | 87% | 0% | 0.87 | 166 |
| 1.00 | std=50/255 (20%) | 0% | 0% | 0.00 | 220 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 126 |
| 0.10 | 1 patches, up to 3% each | 87% | 0% | 0.87 | 162 |
| 0.20 | 1 patches, up to 6% each | 93% | 0% | 0.93 | 154 |
| 0.30 | 1 patches, up to 9% each | 73% | 0% | 0.73 | 167 |
| 0.50 | 2 patches, up to 15% each | 80% | 0% | 0.80 | 163 |
| 0.70 | 3 patches, up to 21% each | 100% | 0% | 1.00 | 154 |
| 1.00 | 5 patches, up to 30% each | 93% | 0% | 0.93 | 171 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 126 |
| 0.10 | +/-8/255 (3%) | 60% | 0% | 0.60 | 176 |
| 0.20 | +/-16/255 (6%) | 60% | 0% | 0.60 | 176 |
| 0.30 | +/-24/255 (9%) | 67% | 0% | 0.67 | 174 |
| 0.50 | +/-40/255 (16%) | 80% | 0% | 0.80 | 160 |
| 0.70 | +/-56/255 (22%) | 100% | 0% | 1.00 | 148 |
| 1.00 | +/-80/255 (31%) | 80% | 0% | 0.80 | 166 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 93% | 0% | 0.93 | 131 |
| 0.10 | 224px (native) | 87% | 0% | 0.87 | 141 |
| 0.20 | 112px effective (2x downscale) | 100% | 0% | 1.00 | 153 |
| 0.30 | 74px effective (3x downscale) | 60% | 0% | 0.60 | 191 |
| 0.50 | 56px effective (4x downscale) | 13% | 0% | 0.13 | 213 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 220 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 220 |

### Task 9: *pick up the black bowl on the wooden cabinet and place it on the plate*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.50 (100ms)
- **DropoutStressor**: 87% baseline, max deg 87%, bp=0.30 (30% dropout)
- **PhysicsShiftStressor**: 93% baseline, max deg 93%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 100% baseline, max deg 100%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 100% baseline, max deg 60%, bp=1.00 (obs std 1.0 @step100)
- **ImageNoiseStressor**: 100% baseline, max deg 100%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 100% baseline, max deg 33%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 20%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.30 (74px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 122 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 134 |
| 0.20 | 40ms (2 steps) | 100% | 0% | 1.00 | 140 |
| 0.30 | 60ms (3 steps) | 93% | 0% | 0.93 | 160 |
| 0.50 | 100ms (5 steps) | 0% | 0% | 0.00 | 220 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 220 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 220 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 87% | 0% | 0.87 | 133 |
| 0.10 | 10% drop prob | 87% | 0% | 0.87 | 150 |
| 0.20 | 20% drop prob | 53% | 0% | 0.53 | 181 |
| 0.30 | 30% drop prob | 13% | 0% | 0.13 | 210 |
| 0.50 | 50% drop prob | 7% | 0% | 0.07 | 220 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 220 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 220 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 93% | 0% | 0.93 | 129 |
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
| 0.00 | no drift | 100% | 0% | 1.00 | 124 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 120 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 124 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 93% | 0% | 0.93 | 128 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 93% | 0% | 0.93 | 132 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 73% | 0% | 0.73 | 157 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 40% | 0% | 0.40 | 192 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 120 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 154 |
| 0.20 | std=10/255 (4%) | 80% | 0% | 0.80 | 165 |
| 0.30 | std=15/255 (6%) | 73% | 0% | 0.73 | 171 |
| 0.50 | std=25/255 (10%) | 87% | 0% | 0.87 | 161 |
| 0.70 | std=35/255 (14%) | 67% | 0% | 0.67 | 190 |
| 1.00 | std=50/255 (20%) | 0% | 0% | 0.00 | 220 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 128 |
| 0.10 | 1 patches, up to 3% each | 93% | 0% | 0.93 | 149 |
| 0.20 | 1 patches, up to 6% each | 80% | 0% | 0.80 | 164 |
| 0.30 | 1 patches, up to 9% each | 67% | 0% | 0.67 | 167 |
| 0.50 | 2 patches, up to 15% each | 73% | 0% | 0.73 | 169 |
| 0.70 | 3 patches, up to 21% each | 87% | 0% | 0.87 | 169 |
| 1.00 | 5 patches, up to 30% each | 67% | 0% | 0.67 | 180 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 119 |
| 0.10 | +/-8/255 (3%) | 87% | 0% | 0.87 | 150 |
| 0.20 | +/-16/255 (6%) | 87% | 0% | 0.87 | 154 |
| 0.30 | +/-24/255 (9%) | 80% | 0% | 0.80 | 168 |
| 0.50 | +/-40/255 (16%) | 100% | 0% | 1.00 | 140 |
| 0.70 | +/-56/255 (22%) | 100% | 0% | 1.00 | 142 |
| 1.00 | +/-80/255 (31%) | 87% | 0% | 0.87 | 155 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 120 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 122 |
| 0.20 | 112px effective (2x downscale) | 80% | 0% | 0.80 | 165 |
| 0.30 | 74px effective (3x downscale) | 47% | 0% | 0.47 | 204 |
| 0.50 | 56px effective (4x downscale) | 0% | 0% | 0.00 | 220 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 220 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 220 |


## libero_object

### Task 0: *pick up the alphabet soup and place it in the basket*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.70 (140ms)
- **DropoutStressor**: 93% baseline, max deg 93%, bp=0.50 (50% dropout)
- **PhysicsShiftStressor**: 93% baseline, max deg 93%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 93% baseline, max deg 93%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 100% baseline, max deg 53%, bp=1.00 (obs std 1.0 @step100)
- **ImageNoiseStressor**: 93% baseline, max deg 53%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 93% baseline, max deg 7%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 13%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.70 (44px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 151 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 149 |
| 0.20 | 40ms (2 steps) | 100% | 0% | 1.00 | 159 |
| 0.30 | 60ms (3 steps) | 100% | 0% | 1.00 | 178 |
| 0.50 | 100ms (5 steps) | 67% | 0% | 0.67 | 256 |
| 0.70 | 140ms (7 steps) | 13% | 0% | 0.13 | 278 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 280 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 93% | 0% | 0.93 | 158 |
| 0.10 | 10% drop prob | 100% | 0% | 1.00 | 172 |
| 0.20 | 20% drop prob | 93% | 0% | 0.93 | 190 |
| 0.30 | 30% drop prob | 60% | 0% | 0.60 | 239 |
| 0.50 | 50% drop prob | 7% | 0% | 0.07 | 278 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 280 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 280 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 93% | 0% | 0.93 | 170 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 280 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 280 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 280 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 280 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 280 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 280 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 93% | 0% | 0.93 | 157 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 280 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 280 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 280 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 280 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 280 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 280 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 155 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 152 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 148 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 93% | 0% | 0.93 | 170 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 87% | 0% | 0.87 | 181 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 87% | 0% | 0.87 | 186 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 47% | 0% | 0.47 | 230 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 93% | 0% | 0.93 | 162 |
| 0.10 | std=5/255 (2%) | 87% | 0% | 0.87 | 211 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 180 |
| 0.30 | std=15/255 (6%) | 100% | 0% | 1.00 | 186 |
| 0.50 | std=25/255 (10%) | 100% | 0% | 1.00 | 183 |
| 0.70 | std=35/255 (14%) | 100% | 0% | 1.00 | 189 |
| 1.00 | std=50/255 (20%) | 40% | 0% | 0.40 | 255 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 93% | 0% | 0.93 | 161 |
| 0.10 | 1 patches, up to 3% each | 100% | 0% | 1.00 | 189 |
| 0.20 | 1 patches, up to 6% each | 87% | 0% | 0.87 | 210 |
| 0.30 | 1 patches, up to 9% each | 93% | 0% | 0.93 | 196 |
| 0.50 | 2 patches, up to 15% each | 87% | 0% | 0.87 | 190 |
| 0.70 | 3 patches, up to 21% each | 100% | 0% | 1.00 | 192 |
| 1.00 | 5 patches, up to 30% each | 87% | 0% | 0.87 | 196 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 147 |
| 0.10 | +/-8/255 (3%) | 87% | 0% | 0.87 | 187 |
| 0.20 | +/-16/255 (6%) | 93% | 0% | 0.93 | 185 |
| 0.30 | +/-24/255 (9%) | 93% | 0% | 0.93 | 190 |
| 0.50 | +/-40/255 (16%) | 100% | 0% | 1.00 | 188 |
| 0.70 | +/-56/255 (22%) | 93% | 0% | 0.93 | 197 |
| 1.00 | +/-80/255 (31%) | 100% | 0% | 1.00 | 178 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 156 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 149 |
| 0.20 | 112px effective (2x downscale) | 93% | 0% | 0.93 | 202 |
| 0.30 | 74px effective (3x downscale) | 87% | 0% | 0.87 | 229 |
| 0.50 | 56px effective (4x downscale) | 67% | 0% | 0.67 | 251 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 280 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 280 |

### Task 4: *pick up the ketchup and place it in the basket*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.50 (100ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.50 (50% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 0%, robust
- **EmbodimentStressor**: 93% baseline, max deg 0%, robust
- **LongHorizonDriftStressor**: 100% baseline, max deg 67%, bp=1.00 (obs std 1.0 @step100)
- **ImageNoiseStressor**: 100% baseline, max deg 73%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 100% baseline, max deg 27%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 13%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.50 (56px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 152 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 154 |
| 0.20 | 40ms (2 steps) | 100% | 0% | 1.00 | 171 |
| 0.30 | 60ms (3 steps) | 93% | 0% | 0.93 | 178 |
| 0.50 | 100ms (5 steps) | 40% | 0% | 0.40 | 260 |
| 0.70 | 140ms (7 steps) | 7% | 0% | 0.07 | 278 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 280 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 155 |
| 0.10 | 10% drop prob | 93% | 0% | 0.93 | 167 |
| 0.20 | 20% drop prob | 87% | 0% | 0.87 | 195 |
| 0.30 | 30% drop prob | 80% | 0% | 0.80 | 216 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 280 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 280 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 280 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 151 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 100% | 0% | 1.00 | 3 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 100% | 0% | 1.00 | 3 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 100% | 0% | 1.00 | 3 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 100% | 0% | 1.00 | 3 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 100% | 0% | 1.00 | 5 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 100% | 0% | 1.00 | 7 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 93% | 0% | 0.93 | 165 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 100% | 0% | 1.00 | 2 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 100% | 0% | 1.00 | 2 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 100% | 0% | 1.00 | 2 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 100% | 0% | 1.00 | 2 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 100% | 0% | 1.00 | 2 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 100% | 0% | 1.00 | 2 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 150 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 143 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 148 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 100% | 0% | 1.00 | 151 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 73% | 0% | 0.73 | 187 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 80% | 0% | 0.80 | 189 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 33% | 0% | 0.33 | 249 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 148 |
| 0.10 | std=5/255 (2%) | 87% | 0% | 0.87 | 186 |
| 0.20 | std=10/255 (4%) | 87% | 0% | 0.87 | 200 |
| 0.30 | std=15/255 (6%) | 100% | 0% | 1.00 | 185 |
| 0.50 | std=25/255 (10%) | 80% | 0% | 0.80 | 198 |
| 0.70 | std=35/255 (14%) | 93% | 0% | 0.93 | 194 |
| 1.00 | std=50/255 (20%) | 27% | 0% | 0.27 | 271 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 150 |
| 0.10 | 1 patches, up to 3% each | 100% | 0% | 1.00 | 166 |
| 0.20 | 1 patches, up to 6% each | 93% | 0% | 0.93 | 177 |
| 0.30 | 1 patches, up to 9% each | 93% | 0% | 0.93 | 177 |
| 0.50 | 2 patches, up to 15% each | 100% | 0% | 1.00 | 179 |
| 0.70 | 3 patches, up to 21% each | 93% | 0% | 0.93 | 198 |
| 1.00 | 5 patches, up to 30% each | 73% | 0% | 0.73 | 205 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 155 |
| 0.10 | +/-8/255 (3%) | 87% | 0% | 0.87 | 189 |
| 0.20 | +/-16/255 (6%) | 93% | 0% | 0.93 | 172 |
| 0.30 | +/-24/255 (9%) | 87% | 0% | 0.87 | 183 |
| 0.50 | +/-40/255 (16%) | 87% | 0% | 0.87 | 191 |
| 0.70 | +/-56/255 (22%) | 100% | 0% | 1.00 | 172 |
| 1.00 | +/-80/255 (31%) | 93% | 0% | 0.93 | 171 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 150 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 153 |
| 0.20 | 112px effective (2x downscale) | 93% | 0% | 0.93 | 178 |
| 0.30 | 74px effective (3x downscale) | 53% | 0% | 0.53 | 247 |
| 0.50 | 56px effective (4x downscale) | 27% | 0% | 0.27 | 276 |
| 0.70 | 44px effective (5x downscale) | 27% | 0% | 0.27 | 274 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 280 |

### Task 9: *pick up the orange juice and place it in the basket*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.50 (100ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.50 (50% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 100%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 100% baseline, max deg 100%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 100% baseline, max deg 53%, bp=1.00 (obs std 1.0 @step100)
- **ImageNoiseStressor**: 100% baseline, max deg 20%, robust
- **OcclusionStressor**: 100% baseline, max deg 40%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 13%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.50 (56px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 122 |
| 0.10 | 20ms (1 steps) | 93% | 0% | 0.93 | 154 |
| 0.20 | 40ms (2 steps) | 87% | 0% | 0.87 | 172 |
| 0.30 | 60ms (3 steps) | 93% | 0% | 0.93 | 168 |
| 0.50 | 100ms (5 steps) | 0% | 0% | 0.00 | 280 |
| 0.70 | 140ms (7 steps) | 7% | 0% | 0.07 | 276 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 280 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 126 |
| 0.10 | 10% drop prob | 93% | 0% | 0.93 | 155 |
| 0.20 | 20% drop prob | 100% | 0% | 1.00 | 163 |
| 0.30 | 30% drop prob | 87% | 0% | 0.87 | 200 |
| 0.50 | 50% drop prob | 33% | 0% | 0.33 | 265 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 280 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 280 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 125 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 280 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 280 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 280 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 280 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 280 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 280 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 132 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 280 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 280 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 280 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 280 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 280 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 13% | 0% | 0.13 | 243 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 133 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 93% | 0% | 0.93 | 137 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 125 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 100% | 0% | 1.00 | 132 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 93% | 0% | 0.93 | 146 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 80% | 0% | 0.80 | 167 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 47% | 0% | 0.47 | 219 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 129 |
| 0.10 | std=5/255 (2%) | 93% | 0% | 0.93 | 167 |
| 0.20 | std=10/255 (4%) | 87% | 0% | 0.87 | 189 |
| 0.30 | std=15/255 (6%) | 93% | 0% | 0.93 | 174 |
| 0.50 | std=25/255 (10%) | 100% | 0% | 1.00 | 170 |
| 0.70 | std=35/255 (14%) | 87% | 0% | 0.87 | 198 |
| 1.00 | std=50/255 (20%) | 80% | 0% | 0.80 | 213 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 128 |
| 0.10 | 1 patches, up to 3% each | 93% | 0% | 0.93 | 182 |
| 0.20 | 1 patches, up to 6% each | 87% | 0% | 0.87 | 187 |
| 0.30 | 1 patches, up to 9% each | 93% | 0% | 0.93 | 165 |
| 0.50 | 2 patches, up to 15% each | 87% | 0% | 0.87 | 167 |
| 0.70 | 3 patches, up to 21% each | 87% | 0% | 0.87 | 180 |
| 1.00 | 5 patches, up to 30% each | 60% | 0% | 0.60 | 227 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 138 |
| 0.10 | +/-8/255 (3%) | 93% | 0% | 0.93 | 176 |
| 0.20 | +/-16/255 (6%) | 87% | 0% | 0.87 | 172 |
| 0.30 | +/-24/255 (9%) | 93% | 0% | 0.93 | 172 |
| 0.50 | +/-40/255 (16%) | 93% | 0% | 0.93 | 168 |
| 0.70 | +/-56/255 (22%) | 100% | 0% | 1.00 | 159 |
| 1.00 | +/-80/255 (31%) | 87% | 0% | 0.87 | 190 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 128 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 129 |
| 0.20 | 112px effective (2x downscale) | 100% | 0% | 1.00 | 167 |
| 0.30 | 74px effective (3x downscale) | 80% | 0% | 0.80 | 188 |
| 0.50 | 56px effective (4x downscale) | 33% | 0% | 0.33 | 254 |
| 0.70 | 44px effective (5x downscale) | 40% | 0% | 0.40 | 253 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 280 |


## libero_goal

### Task 0: *open the middle drawer of the cabinet*

- **LatencyStressor**: 87% baseline, max deg 0%, robust
- **DropoutStressor**: 93% baseline, max deg 93%, bp=0.50 (50% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 100%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 100% baseline, max deg 100%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 93% baseline, max deg 80%, bp=0.70 (obs std 0.7 @step100)
- **ImageNoiseStressor**: 93% baseline, max deg 87%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 93% baseline, max deg 0%, robust
- **BrightnessShiftStressor**: 93% baseline, max deg 0%, robust
- **ResolutionStressor**: 93% baseline, max deg 93%, bp=0.30 (74px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 87% | 0% | 0.87 | 144 |
| 0.10 | 20ms (1 steps) | 93% | 0% | 0.93 | 134 |
| 0.20 | 40ms (2 steps) | 93% | 0% | 0.93 | 146 |
| 0.30 | 60ms (3 steps) | 93% | 0% | 0.93 | 148 |
| 0.50 | 100ms (5 steps) | 100% | 0% | 1.00 | 163 |
| 0.70 | 140ms (7 steps) | 87% | 0% | 0.87 | 177 |
| 1.00 | 200ms (10 steps) | 87% | 0% | 0.87 | 194 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 93% | 0% | 0.93 | 137 |
| 0.10 | 10% drop prob | 80% | 0% | 0.80 | 163 |
| 0.20 | 20% drop prob | 53% | 0% | 0.53 | 212 |
| 0.30 | 30% drop prob | 60% | 0% | 0.60 | 211 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 300 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 300 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 300 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 118 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 300 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 300 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 300 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 300 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 300 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 300 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 129 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 300 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 300 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 300 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 300 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 300 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 300 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 93% | 0% | 0.93 | 142 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 93% | 0% | 0.93 | 132 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 93% | 0% | 0.93 | 132 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 80% | 0% | 0.80 | 162 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 67% | 0% | 0.67 | 189 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 20% | 0% | 0.20 | 264 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 13% | 0% | 0.13 | 284 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 93% | 0% | 0.93 | 132 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 154 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 161 |
| 0.30 | std=15/255 (6%) | 100% | 0% | 1.00 | 162 |
| 0.50 | std=25/255 (10%) | 87% | 0% | 0.87 | 179 |
| 0.70 | std=35/255 (14%) | 87% | 0% | 0.87 | 154 |
| 1.00 | std=50/255 (20%) | 7% | 0% | 0.07 | 290 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 93% | 0% | 0.93 | 135 |
| 0.10 | 1 patches, up to 3% each | 93% | 0% | 0.93 | 154 |
| 0.20 | 1 patches, up to 6% each | 93% | 0% | 0.93 | 152 |
| 0.30 | 1 patches, up to 9% each | 93% | 0% | 0.93 | 154 |
| 0.50 | 2 patches, up to 15% each | 100% | 0% | 1.00 | 153 |
| 0.70 | 3 patches, up to 21% each | 100% | 0% | 1.00 | 136 |
| 1.00 | 5 patches, up to 30% each | 93% | 0% | 0.93 | 185 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 93% | 0% | 0.93 | 129 |
| 0.10 | +/-8/255 (3%) | 93% | 0% | 0.93 | 163 |
| 0.20 | +/-16/255 (6%) | 93% | 0% | 0.93 | 153 |
| 0.30 | +/-24/255 (9%) | 100% | 0% | 1.00 | 139 |
| 0.50 | +/-40/255 (16%) | 93% | 0% | 0.93 | 161 |
| 0.70 | +/-56/255 (22%) | 100% | 0% | 1.00 | 151 |
| 1.00 | +/-80/255 (31%) | 93% | 0% | 0.93 | 151 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 93% | 0% | 0.93 | 147 |
| 0.10 | 224px (native) | 93% | 0% | 0.93 | 128 |
| 0.20 | 112px effective (2x downscale) | 100% | 0% | 1.00 | 144 |
| 0.30 | 74px effective (3x downscale) | 40% | 0% | 0.40 | 234 |
| 0.50 | 56px effective (4x downscale) | 0% | 0% | 0.00 | 300 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 300 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 300 |

### Task 4: *put the bowl on top of the cabinet*

- **LatencyStressor**: 100% baseline, max deg 87%, bp=1.00 (200ms)
- **DropoutStressor**: 93% baseline, max deg 93%, bp=0.50 (50% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 100%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 100% baseline, max deg 100%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 100% baseline, max deg 7%, robust
- **ImageNoiseStressor**: 93% baseline, max deg 67%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 100% baseline, max deg 7%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 0%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.50 (56px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 85 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 97 |
| 0.20 | 40ms (2 steps) | 100% | 0% | 1.00 | 106 |
| 0.30 | 60ms (3 steps) | 87% | 0% | 0.87 | 137 |
| 0.50 | 100ms (5 steps) | 80% | 0% | 0.80 | 213 |
| 0.70 | 140ms (7 steps) | 80% | 0% | 0.80 | 245 |
| 1.00 | 200ms (10 steps) | 13% | 0% | 0.13 | 293 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 93% | 0% | 0.93 | 100 |
| 0.10 | 10% drop prob | 100% | 0% | 1.00 | 95 |
| 0.20 | 20% drop prob | 87% | 0% | 0.87 | 136 |
| 0.30 | 30% drop prob | 80% | 0% | 0.80 | 172 |
| 0.50 | 50% drop prob | 47% | 0% | 0.47 | 246 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 300 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 300 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 90 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 300 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 300 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 300 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 300 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 300 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 300 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 85 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 300 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 300 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 300 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 300 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 300 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 300 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 89 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 86 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 86 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 100% | 0% | 1.00 | 91 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 93% | 0% | 0.93 | 108 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 100% | 0% | 1.00 | 95 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 93% | 0% | 0.93 | 115 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 93% | 0% | 0.93 | 102 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 116 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 111 |
| 0.30 | std=15/255 (6%) | 100% | 0% | 1.00 | 113 |
| 0.50 | std=25/255 (10%) | 100% | 0% | 1.00 | 112 |
| 0.70 | std=35/255 (14%) | 100% | 0% | 1.00 | 111 |
| 1.00 | std=50/255 (20%) | 27% | 0% | 0.27 | 256 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 89 |
| 0.10 | 1 patches, up to 3% each | 100% | 0% | 1.00 | 110 |
| 0.20 | 1 patches, up to 6% each | 100% | 0% | 1.00 | 114 |
| 0.30 | 1 patches, up to 9% each | 100% | 0% | 1.00 | 115 |
| 0.50 | 2 patches, up to 15% each | 100% | 0% | 1.00 | 116 |
| 0.70 | 3 patches, up to 21% each | 100% | 0% | 1.00 | 117 |
| 1.00 | 5 patches, up to 30% each | 93% | 0% | 0.93 | 126 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 90 |
| 0.10 | +/-8/255 (3%) | 100% | 0% | 1.00 | 113 |
| 0.20 | +/-16/255 (6%) | 100% | 0% | 1.00 | 108 |
| 0.30 | +/-24/255 (9%) | 100% | 0% | 1.00 | 107 |
| 0.50 | +/-40/255 (16%) | 100% | 0% | 1.00 | 108 |
| 0.70 | +/-56/255 (22%) | 100% | 0% | 1.00 | 113 |
| 1.00 | +/-80/255 (31%) | 100% | 0% | 1.00 | 109 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 88 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 86 |
| 0.20 | 112px effective (2x downscale) | 100% | 0% | 1.00 | 118 |
| 0.30 | 74px effective (3x downscale) | 80% | 0% | 0.80 | 153 |
| 0.50 | 56px effective (4x downscale) | 33% | 0% | 0.33 | 246 |
| 0.70 | 44px effective (5x downscale) | 20% | 0% | 0.20 | 273 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 300 |

### Task 9: *put the wine bottle on the rack*

- **LatencyStressor**: 93% baseline, max deg 93%, bp=0.50 (100ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.20 (20% dropout)
- **PhysicsShiftStressor**: 93% baseline, max deg 93%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 93% baseline, max deg 93%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 100% baseline, max deg 87%, bp=0.50 (obs std 0.5 @step100)
- **ImageNoiseStressor**: 87% baseline, max deg 67%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 100% baseline, max deg 40%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 27%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.50 (56px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 93% | 0% | 0.93 | 158 |
| 0.10 | 20ms (1 steps) | 93% | 0% | 0.93 | 170 |
| 0.20 | 40ms (2 steps) | 93% | 0% | 0.93 | 184 |
| 0.30 | 60ms (3 steps) | 80% | 0% | 0.80 | 212 |
| 0.50 | 100ms (5 steps) | 33% | 0% | 0.33 | 277 |
| 0.70 | 140ms (7 steps) | 20% | 0% | 0.20 | 294 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 300 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 141 |
| 0.10 | 10% drop prob | 67% | 0% | 0.67 | 208 |
| 0.20 | 20% drop prob | 47% | 0% | 0.47 | 236 |
| 0.30 | 30% drop prob | 20% | 0% | 0.20 | 270 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 300 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 300 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 300 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 93% | 0% | 0.93 | 156 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 300 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 300 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 300 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 300 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 300 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 300 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 93% | 0% | 0.93 | 155 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 300 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 300 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 300 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 300 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 300 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 300 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 142 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 93% | 0% | 0.93 | 151 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 73% | 0% | 0.73 | 184 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 60% | 0% | 0.60 | 205 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 47% | 0% | 0.47 | 241 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 33% | 0% | 0.33 | 248 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 13% | 0% | 0.13 | 282 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 87% | 0% | 0.87 | 155 |
| 0.10 | std=5/255 (2%) | 93% | 0% | 0.93 | 189 |
| 0.20 | std=10/255 (4%) | 93% | 0% | 0.93 | 198 |
| 0.30 | std=15/255 (6%) | 87% | 0% | 0.87 | 217 |
| 0.50 | std=25/255 (10%) | 67% | 0% | 0.67 | 231 |
| 0.70 | std=35/255 (14%) | 53% | 0% | 0.53 | 266 |
| 1.00 | std=50/255 (20%) | 20% | 0% | 0.20 | 269 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 141 |
| 0.10 | 1 patches, up to 3% each | 87% | 0% | 0.87 | 210 |
| 0.20 | 1 patches, up to 6% each | 87% | 0% | 0.87 | 201 |
| 0.30 | 1 patches, up to 9% each | 93% | 0% | 0.93 | 199 |
| 0.50 | 2 patches, up to 15% each | 73% | 0% | 0.73 | 223 |
| 0.70 | 3 patches, up to 21% each | 87% | 0% | 0.87 | 207 |
| 1.00 | 5 patches, up to 30% each | 60% | 0% | 0.60 | 234 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 136 |
| 0.10 | +/-8/255 (3%) | 87% | 0% | 0.87 | 195 |
| 0.20 | +/-16/255 (6%) | 73% | 0% | 0.73 | 218 |
| 0.30 | +/-24/255 (9%) | 93% | 0% | 0.93 | 195 |
| 0.50 | +/-40/255 (16%) | 80% | 0% | 0.80 | 213 |
| 0.70 | +/-56/255 (22%) | 93% | 0% | 0.93 | 191 |
| 1.00 | +/-80/255 (31%) | 100% | 0% | 1.00 | 191 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 140 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 143 |
| 0.20 | 112px effective (2x downscale) | 73% | 0% | 0.73 | 223 |
| 0.30 | 74px effective (3x downscale) | 53% | 0% | 0.53 | 257 |
| 0.50 | 56px effective (4x downscale) | 0% | 0% | 0.00 | 300 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 300 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 300 |


## libero_10

### Task 0: *put both the alphabet soup and the tomato sauce in the basket*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.50 (100ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.30 (30% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 0%, robust
- **EmbodimentStressor**: 100% baseline, max deg 13%, robust
- **LongHorizonDriftStressor**: 100% baseline, max deg 100%, bp=0.30 (obs std 0.3 @step100)
- **ImageNoiseStressor**: 100% baseline, max deg 100%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 100% baseline, max deg 27%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 33%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.30 (74px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 276 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 308 |
| 0.20 | 40ms (2 steps) | 93% | 0% | 0.93 | 364 |
| 0.30 | 60ms (3 steps) | 67% | 0% | 0.67 | 416 |
| 0.50 | 100ms (5 steps) | 13% | 0% | 0.13 | 516 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 520 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 520 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 272 |
| 0.10 | 10% drop prob | 100% | 0% | 1.00 | 316 |
| 0.20 | 20% drop prob | 93% | 0% | 0.93 | 384 |
| 0.30 | 30% drop prob | 40% | 0% | 0.40 | 457 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 520 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 520 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 520 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 274 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 100% | 0% | 1.00 | 1 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 100% | 0% | 1.00 | 1 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 100% | 0% | 1.00 | 1 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 100% | 0% | 1.00 | 2 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 100% | 0% | 1.00 | 2 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 100% | 0% | 1.00 | 2 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 278 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 100% | 0% | 1.00 | 1 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 100% | 0% | 1.00 | 1 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 100% | 0% | 1.00 | 1 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 100% | 0% | 1.00 | 1 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 93% | 0% | 0.93 | 70 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 87% | 0% | 0.87 | 92 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 267 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 93% | 0% | 0.93 | 289 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 67% | 0% | 0.67 | 375 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 33% | 0% | 0.33 | 438 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 7% | 0% | 0.07 | 503 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 0% | 0% | 0.00 | 520 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 0% | 0% | 0.00 | 520 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 278 |
| 0.10 | std=5/255 (2%) | 80% | 0% | 0.80 | 374 |
| 0.20 | std=10/255 (4%) | 73% | 0% | 0.73 | 403 |
| 0.30 | std=15/255 (6%) | 80% | 0% | 0.80 | 385 |
| 0.50 | std=25/255 (10%) | 87% | 0% | 0.87 | 378 |
| 0.70 | std=35/255 (14%) | 80% | 0% | 0.80 | 420 |
| 1.00 | std=50/255 (20%) | 0% | 0% | 0.00 | 520 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 272 |
| 0.10 | 1 patches, up to 3% each | 73% | 0% | 0.73 | 402 |
| 0.20 | 1 patches, up to 6% each | 80% | 0% | 0.80 | 386 |
| 0.30 | 1 patches, up to 9% each | 80% | 0% | 0.80 | 387 |
| 0.50 | 2 patches, up to 15% each | 80% | 0% | 0.80 | 407 |
| 0.70 | 3 patches, up to 21% each | 93% | 0% | 0.93 | 374 |
| 1.00 | 5 patches, up to 30% each | 73% | 0% | 0.73 | 436 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 279 |
| 0.10 | +/-8/255 (3%) | 67% | 0% | 0.67 | 399 |
| 0.20 | +/-16/255 (6%) | 73% | 0% | 0.73 | 390 |
| 0.30 | +/-24/255 (9%) | 93% | 0% | 0.93 | 367 |
| 0.50 | +/-40/255 (16%) | 93% | 0% | 0.93 | 373 |
| 0.70 | +/-56/255 (22%) | 73% | 0% | 0.73 | 385 |
| 1.00 | +/-80/255 (31%) | 80% | 0% | 0.80 | 382 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 277 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 275 |
| 0.20 | 112px effective (2x downscale) | 60% | 0% | 0.60 | 418 |
| 0.30 | 74px effective (3x downscale) | 27% | 0% | 0.27 | 493 |
| 0.50 | 56px effective (4x downscale) | 7% | 0% | 0.07 | 519 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 520 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 520 |


## Breakpoint Comparison

Average intensity at which success rate drops below 50%:

- **LatencyStressor**: libero_spatial: 0.50, libero_object: 0.57, libero_goal: 0.75, libero_10: 0.50
- **DropoutStressor**: libero_spatial: 0.43, libero_object: 0.50, libero_goal: 0.40, libero_10: 0.30
- **PhysicsShiftStressor**: libero_spatial: 0.10, libero_object: 0.10, libero_goal: 0.10, libero_10: robust
- **EmbodimentStressor**: libero_spatial: 0.10, libero_object: 0.10, libero_goal: 0.10, libero_10: robust
- **LongHorizonDriftStressor**: libero_spatial: 0.85, libero_object: 1.00, libero_goal: 0.60, libero_10: 0.30
- **ImageNoiseStressor**: libero_spatial: 1.00, libero_object: 1.00, libero_goal: 1.00, libero_10: 1.00
- **OcclusionStressor**: libero_spatial: robust, libero_object: robust, libero_goal: robust, libero_10: robust
- **BrightnessShiftStressor**: libero_spatial: robust, libero_object: robust, libero_goal: robust, libero_10: robust
- **ResolutionStressor**: libero_spatial: 0.43, libero_object: 0.57, libero_goal: 0.43, libero_10: 0.30

---

*Report generated by Trace Robotics v0.1.0*
*https://tracerobotics.com*