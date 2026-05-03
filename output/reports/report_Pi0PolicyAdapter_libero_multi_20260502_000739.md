# Trace Robotics — Multi-Task Robustness Report

**Policy:** Pi0PolicyAdapter  
**Suites:** libero_spatial, libero_object, libero_goal, libero_10  
**Task IDs:** [0, 4, 9]  
**Total tasks evaluated:** 10  
**Modalities:** vision, proprioception  
**Generated:** 2026-05-03 06:44  
**Control frequency:** 50Hz (20ms per step)

---

## Cross-Suite Summary

| Stressor | libero_spatial | libero_object | libero_goal | libero_10 |
|---|---|---|---|---|
| LatencyStressor | 93% (bp=0.50) | 100% (bp=0.57) | 98% (bp=0.75) | 100% (bp=0.50) |
| DropoutStressor | 96% (bp=0.37) | 100% (bp=0.43) | 100% (bp=0.33) | 100% (bp=0.50) |
| PhysicsShiftStressor | 100% (bp=0.10) | 98% (bp=0.10) | 100% (bp=0.10) | 93% (robust) |
| EmbodimentStressor | 100% (bp=0.10) | 98% (bp=0.10) | 98% (bp=0.10) | 80% (robust) |
| LongHorizonDriftStressor | 100% (bp=0.85) | 96% (bp=1.00) | 96% (bp=0.70) | 93% (bp=0.30) |
| ImageNoiseStressor | 98% (bp=0.85) | 100% (bp=1.00) | 100% (bp=0.90) | 100% (bp=1.00) |
| OcclusionStressor | 100% (robust) | 100% (robust) | 100% (robust) | 100% (robust) |
| BrightnessShiftStressor | 96% (robust) | 100% (robust) | 96% (robust) | 100% (robust) |
| ResolutionStressor | 100% (bp=0.43) | 100% (bp=0.50) | 100% (bp=0.50) | 100% (bp=0.30) |

## libero_spatial

### Task 0: *pick up the black bowl between the plate and the ramekin and place it on the plate*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.50 (100ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.50 (50% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 100%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 100% baseline, max deg 100%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 100% baseline, max deg 33%, robust
- **ImageNoiseStressor**: 100% baseline, max deg 20%, robust
- **OcclusionStressor**: 100% baseline, max deg 33%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 7%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.50 (56px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 76 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 87 |
| 0.20 | 40ms (2 steps) | 100% | 0% | 1.00 | 107 |
| 0.30 | 60ms (3 steps) | 80% | 0% | 0.80 | 152 |
| 0.50 | 100ms (5 steps) | 7% | 0% | 0.07 | 218 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 220 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 220 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 84 |
| 0.10 | 10% drop prob | 100% | 0% | 1.00 | 87 |
| 0.20 | 20% drop prob | 100% | 0% | 1.00 | 103 |
| 0.30 | 30% drop prob | 87% | 0% | 0.87 | 134 |
| 0.50 | 50% drop prob | 40% | 0% | 0.40 | 189 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 220 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 220 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 86 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 220 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 220 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 220 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 220 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 220 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 220 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 83 |
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
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 80 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 87 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 93% | 0% | 0.93 | 86 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 93% | 0% | 0.93 | 93 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 100% | 0% | 1.00 | 84 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 67% | 0% | 0.67 | 131 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 84 |
| 0.10 | std=5/255 (2%) | 93% | 0% | 0.93 | 115 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 110 |
| 0.30 | std=15/255 (6%) | 87% | 0% | 0.87 | 120 |
| 0.50 | std=25/255 (10%) | 100% | 0% | 1.00 | 94 |
| 0.70 | std=35/255 (14%) | 80% | 0% | 0.80 | 127 |
| 1.00 | std=50/255 (20%) | 80% | 0% | 0.80 | 133 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 81 |
| 0.10 | 1 patches, up to 3% each | 93% | 0% | 0.93 | 118 |
| 0.20 | 1 patches, up to 6% each | 87% | 0% | 0.87 | 125 |
| 0.30 | 1 patches, up to 9% each | 100% | 0% | 1.00 | 122 |
| 0.50 | 2 patches, up to 15% each | 80% | 0% | 0.80 | 131 |
| 0.70 | 3 patches, up to 21% each | 67% | 0% | 0.67 | 145 |
| 1.00 | 5 patches, up to 30% each | 73% | 0% | 0.73 | 162 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 77 |
| 0.10 | +/-8/255 (3%) | 93% | 0% | 0.93 | 114 |
| 0.20 | +/-16/255 (6%) | 100% | 0% | 1.00 | 118 |
| 0.30 | +/-24/255 (9%) | 93% | 0% | 0.93 | 126 |
| 0.50 | +/-40/255 (16%) | 93% | 0% | 0.93 | 125 |
| 0.70 | +/-56/255 (22%) | 93% | 0% | 0.93 | 120 |
| 1.00 | +/-80/255 (31%) | 93% | 0% | 0.93 | 121 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 82 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 81 |
| 0.20 | 112px effective (2x downscale) | 87% | 0% | 0.87 | 127 |
| 0.30 | 74px effective (3x downscale) | 80% | 0% | 0.80 | 155 |
| 0.50 | 56px effective (4x downscale) | 27% | 0% | 0.27 | 205 |
| 0.70 | 44px effective (5x downscale) | 33% | 0% | 0.33 | 191 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 220 |

### Task 4: *pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate*

- **LatencyStressor**: 80% baseline, max deg 80%, bp=0.50 (100ms)
- **DropoutStressor**: 87% baseline, max deg 87%, bp=0.30 (30% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 100%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 100% baseline, max deg 100%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 100% baseline, max deg 53%, bp=0.70 (obs std 0.7 @step100)
- **ImageNoiseStressor**: 93% baseline, max deg 93%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 100% baseline, max deg 20%, robust
- **BrightnessShiftStressor**: 87% baseline, max deg 13%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.50 (56px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 80% | 0% | 0.80 | 144 |
| 0.10 | 20ms (1 steps) | 87% | 0% | 0.87 | 146 |
| 0.20 | 40ms (2 steps) | 100% | 0% | 1.00 | 143 |
| 0.30 | 60ms (3 steps) | 100% | 0% | 1.00 | 149 |
| 0.50 | 100ms (5 steps) | 20% | 0% | 0.20 | 211 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 220 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 220 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 87% | 0% | 0.87 | 136 |
| 0.10 | 10% drop prob | 93% | 0% | 0.93 | 147 |
| 0.20 | 20% drop prob | 73% | 0% | 0.73 | 171 |
| 0.30 | 30% drop prob | 33% | 0% | 0.33 | 195 |
| 0.50 | 50% drop prob | 7% | 0% | 0.07 | 216 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 220 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 220 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 133 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 220 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 220 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 220 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 220 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 220 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 220 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 127 |
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
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 93% | 0% | 0.93 | 131 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 124 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 93% | 0% | 0.93 | 132 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 67% | 0% | 0.67 | 160 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 47% | 0% | 0.47 | 179 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 67% | 0% | 0.67 | 171 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 93% | 0% | 0.93 | 132 |
| 0.10 | std=5/255 (2%) | 80% | 0% | 0.80 | 162 |
| 0.20 | std=10/255 (4%) | 80% | 0% | 0.80 | 164 |
| 0.30 | std=15/255 (6%) | 93% | 0% | 0.93 | 160 |
| 0.50 | std=25/255 (10%) | 87% | 0% | 0.87 | 169 |
| 0.70 | std=35/255 (14%) | 100% | 0% | 1.00 | 157 |
| 1.00 | std=50/255 (20%) | 0% | 0% | 0.00 | 220 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 125 |
| 0.10 | 1 patches, up to 3% each | 87% | 0% | 0.87 | 155 |
| 0.20 | 1 patches, up to 6% each | 87% | 0% | 0.87 | 156 |
| 0.30 | 1 patches, up to 9% each | 100% | 0% | 1.00 | 146 |
| 0.50 | 2 patches, up to 15% each | 100% | 0% | 1.00 | 145 |
| 0.70 | 3 patches, up to 21% each | 100% | 0% | 1.00 | 147 |
| 1.00 | 5 patches, up to 30% each | 80% | 0% | 0.80 | 172 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 87% | 0% | 0.87 | 139 |
| 0.10 | +/-8/255 (3%) | 80% | 0% | 0.80 | 168 |
| 0.20 | +/-16/255 (6%) | 73% | 0% | 0.73 | 164 |
| 0.30 | +/-24/255 (9%) | 80% | 0% | 0.80 | 158 |
| 0.50 | +/-40/255 (16%) | 87% | 0% | 0.87 | 160 |
| 0.70 | +/-56/255 (22%) | 100% | 0% | 1.00 | 146 |
| 1.00 | +/-80/255 (31%) | 100% | 0% | 1.00 | 144 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 119 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 130 |
| 0.20 | 112px effective (2x downscale) | 93% | 0% | 0.93 | 160 |
| 0.30 | 74px effective (3x downscale) | 60% | 0% | 0.60 | 190 |
| 0.50 | 56px effective (4x downscale) | 7% | 0% | 0.07 | 217 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 220 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 220 |

### Task 9: *pick up the black bowl on the wooden cabinet and place it on the plate*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.50 (100ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.30 (30% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 100%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 100% baseline, max deg 100%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 100% baseline, max deg 80%, bp=1.00 (obs std 1.0 @step100)
- **ImageNoiseStressor**: 100% baseline, max deg 100%, bp=0.70 (noise std=35)
- **OcclusionStressor**: 100% baseline, max deg 33%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 13%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.30 (74px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 115 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 133 |
| 0.20 | 40ms (2 steps) | 93% | 0% | 0.93 | 146 |
| 0.30 | 60ms (3 steps) | 87% | 0% | 0.87 | 159 |
| 0.50 | 100ms (5 steps) | 7% | 0% | 0.07 | 217 |
| 0.70 | 140ms (7 steps) | 13% | 0% | 0.13 | 216 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 220 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 116 |
| 0.10 | 10% drop prob | 93% | 0% | 0.93 | 149 |
| 0.20 | 20% drop prob | 53% | 0% | 0.53 | 179 |
| 0.30 | 30% drop prob | 33% | 0% | 0.33 | 204 |
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
| 0.00 | nominal | 100% | 0% | 1.00 | 116 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 220 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 220 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 220 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 220 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 220 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 220 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 117 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 119 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 114 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 87% | 0% | 0.87 | 131 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 87% | 0% | 0.87 | 139 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 80% | 0% | 0.80 | 152 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 20% | 0% | 0.20 | 202 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 116 |
| 0.10 | std=5/255 (2%) | 80% | 0% | 0.80 | 156 |
| 0.20 | std=10/255 (4%) | 93% | 0% | 0.93 | 149 |
| 0.30 | std=15/255 (6%) | 93% | 0% | 0.93 | 147 |
| 0.50 | std=25/255 (10%) | 93% | 0% | 0.93 | 151 |
| 0.70 | std=35/255 (14%) | 47% | 0% | 0.47 | 184 |
| 1.00 | std=50/255 (20%) | 0% | 0% | 0.00 | 220 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 117 |
| 0.10 | 1 patches, up to 3% each | 87% | 0% | 0.87 | 157 |
| 0.20 | 1 patches, up to 6% each | 100% | 0% | 1.00 | 138 |
| 0.30 | 1 patches, up to 9% each | 93% | 0% | 0.93 | 151 |
| 0.50 | 2 patches, up to 15% each | 100% | 0% | 1.00 | 145 |
| 0.70 | 3 patches, up to 21% each | 93% | 0% | 0.93 | 152 |
| 1.00 | 5 patches, up to 30% each | 67% | 0% | 0.67 | 173 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 116 |
| 0.10 | +/-8/255 (3%) | 100% | 0% | 1.00 | 144 |
| 0.20 | +/-16/255 (6%) | 87% | 0% | 0.87 | 159 |
| 0.30 | +/-24/255 (9%) | 100% | 0% | 1.00 | 142 |
| 0.50 | +/-40/255 (16%) | 93% | 0% | 0.93 | 147 |
| 0.70 | +/-56/255 (22%) | 93% | 0% | 0.93 | 143 |
| 1.00 | +/-80/255 (31%) | 87% | 0% | 0.87 | 151 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 117 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 122 |
| 0.20 | 112px effective (2x downscale) | 93% | 0% | 0.93 | 155 |
| 0.30 | 74px effective (3x downscale) | 13% | 0% | 0.13 | 216 |
| 0.50 | 56px effective (4x downscale) | 0% | 0% | 0.00 | 220 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 220 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 220 |


## libero_object

### Task 0: *pick up the alphabet soup and place it in the basket*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.70 (140ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.30 (30% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 100%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 93% baseline, max deg 93%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 93% baseline, max deg 33%, robust
- **ImageNoiseStressor**: 100% baseline, max deg 40%, robust
- **OcclusionStressor**: 100% baseline, max deg 7%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 7%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.70 (44px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 149 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 149 |
| 0.20 | 40ms (2 steps) | 93% | 0% | 0.93 | 161 |
| 0.30 | 60ms (3 steps) | 100% | 0% | 1.00 | 172 |
| 0.50 | 100ms (5 steps) | 67% | 0% | 0.67 | 252 |
| 0.70 | 140ms (7 steps) | 7% | 0% | 0.07 | 279 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 280 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 150 |
| 0.10 | 10% drop prob | 87% | 0% | 0.87 | 168 |
| 0.20 | 20% drop prob | 73% | 0% | 0.73 | 210 |
| 0.30 | 30% drop prob | 47% | 0% | 0.47 | 241 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 280 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 280 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 280 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 152 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 280 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 280 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 280 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 280 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 280 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 280 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 93% | 0% | 0.93 | 154 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 280 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 280 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 280 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 280 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 280 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 280 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 93% | 0% | 0.93 | 151 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 146 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 153 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 100% | 0% | 1.00 | 162 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 87% | 0% | 0.87 | 177 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 73% | 0% | 0.73 | 198 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 60% | 0% | 0.60 | 211 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 145 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 175 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 158 |
| 0.30 | std=15/255 (6%) | 100% | 0% | 1.00 | 169 |
| 0.50 | std=25/255 (10%) | 93% | 0% | 0.93 | 203 |
| 0.70 | std=35/255 (14%) | 93% | 0% | 0.93 | 218 |
| 1.00 | std=50/255 (20%) | 60% | 0% | 0.60 | 234 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 155 |
| 0.10 | 1 patches, up to 3% each | 100% | 0% | 1.00 | 168 |
| 0.20 | 1 patches, up to 6% each | 100% | 0% | 1.00 | 157 |
| 0.30 | 1 patches, up to 9% each | 100% | 0% | 1.00 | 166 |
| 0.50 | 2 patches, up to 15% each | 100% | 0% | 1.00 | 168 |
| 0.70 | 3 patches, up to 21% each | 100% | 0% | 1.00 | 161 |
| 1.00 | 5 patches, up to 30% each | 93% | 0% | 0.93 | 188 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 148 |
| 0.10 | +/-8/255 (3%) | 100% | 0% | 1.00 | 167 |
| 0.20 | +/-16/255 (6%) | 100% | 0% | 1.00 | 176 |
| 0.30 | +/-24/255 (9%) | 93% | 0% | 0.93 | 173 |
| 0.50 | +/-40/255 (16%) | 100% | 0% | 1.00 | 170 |
| 0.70 | +/-56/255 (22%) | 100% | 0% | 1.00 | 162 |
| 1.00 | +/-80/255 (31%) | 100% | 0% | 1.00 | 176 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 148 |
| 0.10 | 224px (native) | 87% | 0% | 0.87 | 174 |
| 0.20 | 112px effective (2x downscale) | 93% | 0% | 0.93 | 197 |
| 0.30 | 74px effective (3x downscale) | 87% | 0% | 0.87 | 233 |
| 0.50 | 56px effective (4x downscale) | 80% | 0% | 0.80 | 255 |
| 0.70 | 44px effective (5x downscale) | 7% | 0% | 0.07 | 279 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 280 |

### Task 4: *pick up the ketchup and place it in the basket*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.50 (100ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.50 (50% dropout)
- **PhysicsShiftStressor**: 93% baseline, max deg 20%, robust
- **EmbodimentStressor**: 100% baseline, max deg 13%, robust
- **LongHorizonDriftStressor**: 93% baseline, max deg 47%, bp=1.00 (obs std 1.0 @step100)
- **ImageNoiseStressor**: 100% baseline, max deg 80%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 100% baseline, max deg 0%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 13%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.30 (74px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 148 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 164 |
| 0.20 | 40ms (2 steps) | 100% | 0% | 1.00 | 154 |
| 0.30 | 60ms (3 steps) | 93% | 0% | 0.93 | 176 |
| 0.50 | 100ms (5 steps) | 33% | 0% | 0.33 | 253 |
| 0.70 | 140ms (7 steps) | 13% | 0% | 0.13 | 272 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 280 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 147 |
| 0.10 | 10% drop prob | 100% | 0% | 1.00 | 156 |
| 0.20 | 20% drop prob | 93% | 0% | 0.93 | 190 |
| 0.30 | 30% drop prob | 60% | 0% | 0.60 | 243 |
| 0.50 | 50% drop prob | 27% | 0% | 0.27 | 269 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 280 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 280 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 93% | 0% | 0.93 | 154 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 100% | 0% | 1.00 | 1 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 100% | 0% | 1.00 | 1 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 93% | 0% | 0.93 | 20 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 73% | 0% | 0.73 | 76 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 93% | 0% | 0.93 | 21 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 93% | 0% | 0.93 | 22 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 149 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 100% | 0% | 1.00 | 1 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 100% | 0% | 1.00 | 1 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 100% | 0% | 1.00 | 2 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 100% | 0% | 1.00 | 1 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 100% | 0% | 1.00 | 3 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 87% | 0% | 0.87 | 40 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 93% | 0% | 0.93 | 152 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 148 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 151 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 100% | 0% | 1.00 | 161 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 93% | 0% | 0.93 | 168 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 73% | 0% | 0.73 | 199 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 47% | 0% | 0.47 | 223 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 150 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 162 |
| 0.20 | std=10/255 (4%) | 87% | 0% | 0.87 | 190 |
| 0.30 | std=15/255 (6%) | 93% | 0% | 0.93 | 172 |
| 0.50 | std=25/255 (10%) | 87% | 0% | 0.87 | 186 |
| 0.70 | std=35/255 (14%) | 87% | 0% | 0.87 | 187 |
| 1.00 | std=50/255 (20%) | 20% | 0% | 0.20 | 265 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 151 |
| 0.10 | 1 patches, up to 3% each | 100% | 0% | 1.00 | 174 |
| 0.20 | 1 patches, up to 6% each | 100% | 0% | 1.00 | 158 |
| 0.30 | 1 patches, up to 9% each | 100% | 0% | 1.00 | 175 |
| 0.50 | 2 patches, up to 15% each | 100% | 0% | 1.00 | 166 |
| 0.70 | 3 patches, up to 21% each | 100% | 0% | 1.00 | 162 |
| 1.00 | 5 patches, up to 30% each | 100% | 0% | 1.00 | 169 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 146 |
| 0.10 | +/-8/255 (3%) | 87% | 0% | 0.87 | 188 |
| 0.20 | +/-16/255 (6%) | 93% | 0% | 0.93 | 171 |
| 0.30 | +/-24/255 (9%) | 93% | 0% | 0.93 | 170 |
| 0.50 | +/-40/255 (16%) | 93% | 0% | 0.93 | 178 |
| 0.70 | +/-56/255 (22%) | 100% | 0% | 1.00 | 167 |
| 1.00 | +/-80/255 (31%) | 100% | 0% | 1.00 | 163 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 156 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 157 |
| 0.20 | 112px effective (2x downscale) | 100% | 0% | 1.00 | 172 |
| 0.30 | 74px effective (3x downscale) | 33% | 0% | 0.33 | 253 |
| 0.50 | 56px effective (4x downscale) | 7% | 0% | 0.07 | 279 |
| 0.70 | 44px effective (5x downscale) | 13% | 0% | 0.13 | 278 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 280 |

### Task 9: *pick up the orange juice and place it in the basket*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.50 (100ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.50 (50% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 100%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 100% baseline, max deg 100%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 100% baseline, max deg 20%, robust
- **ImageNoiseStressor**: 100% baseline, max deg 20%, robust
- **OcclusionStressor**: 100% baseline, max deg 27%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 27%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.50 (56px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 124 |
| 0.10 | 20ms (1 steps) | 93% | 0% | 0.93 | 151 |
| 0.20 | 40ms (2 steps) | 100% | 0% | 1.00 | 154 |
| 0.30 | 60ms (3 steps) | 100% | 0% | 1.00 | 154 |
| 0.50 | 100ms (5 steps) | 47% | 0% | 0.47 | 248 |
| 0.70 | 140ms (7 steps) | 13% | 0% | 0.13 | 280 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 280 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 127 |
| 0.10 | 10% drop prob | 100% | 0% | 1.00 | 141 |
| 0.20 | 20% drop prob | 87% | 0% | 0.87 | 173 |
| 0.30 | 30% drop prob | 80% | 0% | 0.80 | 191 |
| 0.50 | 50% drop prob | 20% | 0% | 0.20 | 263 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 280 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 280 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 128 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 280 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 280 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 280 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 280 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 280 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 280 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 121 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 280 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 280 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 280 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 280 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 280 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 280 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 122 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 122 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 122 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 100% | 0% | 1.00 | 128 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 93% | 0% | 0.93 | 141 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 87% | 0% | 0.87 | 155 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 80% | 0% | 0.80 | 175 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 122 |
| 0.10 | std=5/255 (2%) | 80% | 0% | 0.80 | 176 |
| 0.20 | std=10/255 (4%) | 93% | 0% | 0.93 | 151 |
| 0.30 | std=15/255 (6%) | 100% | 0% | 1.00 | 151 |
| 0.50 | std=25/255 (10%) | 100% | 0% | 1.00 | 145 |
| 0.70 | std=35/255 (14%) | 100% | 0% | 1.00 | 166 |
| 1.00 | std=50/255 (20%) | 80% | 0% | 0.80 | 198 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 131 |
| 0.10 | 1 patches, up to 3% each | 100% | 0% | 1.00 | 157 |
| 0.20 | 1 patches, up to 6% each | 80% | 0% | 0.80 | 175 |
| 0.30 | 1 patches, up to 9% each | 73% | 0% | 0.73 | 190 |
| 0.50 | 2 patches, up to 15% each | 93% | 0% | 0.93 | 158 |
| 0.70 | 3 patches, up to 21% each | 87% | 0% | 0.87 | 171 |
| 1.00 | 5 patches, up to 30% each | 87% | 0% | 0.87 | 176 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 130 |
| 0.10 | +/-8/255 (3%) | 73% | 0% | 0.73 | 184 |
| 0.20 | +/-16/255 (6%) | 87% | 0% | 0.87 | 168 |
| 0.30 | +/-24/255 (9%) | 100% | 0% | 1.00 | 149 |
| 0.50 | +/-40/255 (16%) | 87% | 0% | 0.87 | 171 |
| 0.70 | +/-56/255 (22%) | 93% | 0% | 0.93 | 164 |
| 1.00 | +/-80/255 (31%) | 87% | 0% | 0.87 | 175 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 130 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 120 |
| 0.20 | 112px effective (2x downscale) | 100% | 0% | 1.00 | 147 |
| 0.30 | 74px effective (3x downscale) | 93% | 0% | 0.93 | 177 |
| 0.50 | 56px effective (4x downscale) | 40% | 0% | 0.40 | 241 |
| 0.70 | 44px effective (5x downscale) | 33% | 0% | 0.33 | 250 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 280 |


## libero_goal

### Task 0: *open the middle drawer of the cabinet*

- **LatencyStressor**: 93% baseline, max deg 0%, robust
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.30 (30% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 100%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 93% baseline, max deg 93%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 93% baseline, max deg 93%, bp=0.70 (obs std 0.7 @step100)
- **ImageNoiseStressor**: 100% baseline, max deg 93%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 100% baseline, max deg 7%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 7%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.50 (56px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 93% | 0% | 0.93 | 130 |
| 0.10 | 20ms (1 steps) | 93% | 0% | 0.93 | 135 |
| 0.20 | 40ms (2 steps) | 93% | 0% | 0.93 | 138 |
| 0.30 | 60ms (3 steps) | 93% | 0% | 0.93 | 140 |
| 0.50 | 100ms (5 steps) | 100% | 0% | 1.00 | 142 |
| 0.70 | 140ms (7 steps) | 100% | 0% | 1.00 | 149 |
| 1.00 | 200ms (10 steps) | 100% | 0% | 1.00 | 164 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 119 |
| 0.10 | 10% drop prob | 87% | 0% | 0.87 | 163 |
| 0.20 | 20% drop prob | 67% | 0% | 0.67 | 213 |
| 0.30 | 30% drop prob | 33% | 0% | 0.33 | 264 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 300 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 300 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 300 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 116 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 300 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 300 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 300 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 300 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 300 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 300 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 93% | 0% | 0.93 | 130 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 300 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 300 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 300 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 300 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 300 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 300 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 93% | 0% | 0.93 | 129 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 115 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 93% | 0% | 0.93 | 129 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 93% | 0% | 0.93 | 138 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 73% | 0% | 0.73 | 172 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 27% | 0% | 0.27 | 249 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 0% | 0% | 0.00 | 300 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 121 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 132 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 134 |
| 0.30 | std=15/255 (6%) | 100% | 0% | 1.00 | 146 |
| 0.50 | std=25/255 (10%) | 93% | 0% | 0.93 | 168 |
| 0.70 | std=35/255 (14%) | 60% | 0% | 0.60 | 196 |
| 1.00 | std=50/255 (20%) | 7% | 0% | 0.07 | 288 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 117 |
| 0.10 | 1 patches, up to 3% each | 93% | 0% | 0.93 | 141 |
| 0.20 | 1 patches, up to 6% each | 100% | 0% | 1.00 | 135 |
| 0.30 | 1 patches, up to 9% each | 93% | 0% | 0.93 | 140 |
| 0.50 | 2 patches, up to 15% each | 100% | 0% | 1.00 | 132 |
| 0.70 | 3 patches, up to 21% each | 100% | 0% | 1.00 | 136 |
| 1.00 | 5 patches, up to 30% each | 100% | 0% | 1.00 | 152 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 116 |
| 0.10 | +/-8/255 (3%) | 93% | 0% | 0.93 | 137 |
| 0.20 | +/-16/255 (6%) | 100% | 0% | 1.00 | 132 |
| 0.30 | +/-24/255 (9%) | 93% | 0% | 0.93 | 145 |
| 0.50 | +/-40/255 (16%) | 100% | 0% | 1.00 | 132 |
| 0.70 | +/-56/255 (22%) | 93% | 0% | 0.93 | 140 |
| 1.00 | +/-80/255 (31%) | 93% | 0% | 0.93 | 152 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 120 |
| 0.10 | 224px (native) | 93% | 0% | 0.93 | 130 |
| 0.20 | 112px effective (2x downscale) | 100% | 0% | 1.00 | 136 |
| 0.30 | 74px effective (3x downscale) | 67% | 0% | 0.67 | 194 |
| 0.50 | 56px effective (4x downscale) | 0% | 0% | 0.00 | 300 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 300 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 300 |

### Task 4: *put the bowl on top of the cabinet*

- **LatencyStressor**: 100% baseline, max deg 93%, bp=1.00 (200ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.50 (50% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 100%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 100% baseline, max deg 100%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 100% baseline, max deg 0%, robust
- **ImageNoiseStressor**: 100% baseline, max deg 67%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 100% baseline, max deg 13%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 0%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.50 (56px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 88 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 90 |
| 0.20 | 40ms (2 steps) | 100% | 0% | 1.00 | 95 |
| 0.30 | 60ms (3 steps) | 100% | 0% | 1.00 | 108 |
| 0.50 | 100ms (5 steps) | 87% | 0% | 0.87 | 201 |
| 0.70 | 140ms (7 steps) | 73% | 0% | 0.73 | 230 |
| 1.00 | 200ms (10 steps) | 7% | 0% | 0.07 | 298 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 90 |
| 0.10 | 10% drop prob | 100% | 0% | 1.00 | 91 |
| 0.20 | 20% drop prob | 87% | 0% | 0.87 | 130 |
| 0.30 | 30% drop prob | 80% | 0% | 0.80 | 145 |
| 0.50 | 50% drop prob | 40% | 0% | 0.40 | 249 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 300 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 300 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 87 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 300 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 300 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 300 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 300 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 300 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 300 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 86 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 300 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 300 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 300 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 300 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 300 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 300 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 86 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 88 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 85 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 100% | 0% | 1.00 | 89 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 100% | 0% | 1.00 | 88 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 100% | 0% | 1.00 | 89 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 100% | 0% | 1.00 | 93 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 91 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 104 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 105 |
| 0.30 | std=15/255 (6%) | 100% | 0% | 1.00 | 106 |
| 0.50 | std=25/255 (10%) | 100% | 0% | 1.00 | 108 |
| 0.70 | std=35/255 (14%) | 100% | 0% | 1.00 | 111 |
| 1.00 | std=50/255 (20%) | 33% | 0% | 0.33 | 255 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 86 |
| 0.10 | 1 patches, up to 3% each | 100% | 0% | 1.00 | 102 |
| 0.20 | 1 patches, up to 6% each | 100% | 0% | 1.00 | 103 |
| 0.30 | 1 patches, up to 9% each | 100% | 0% | 1.00 | 102 |
| 0.50 | 2 patches, up to 15% each | 100% | 0% | 1.00 | 106 |
| 0.70 | 3 patches, up to 21% each | 93% | 0% | 0.93 | 127 |
| 1.00 | 5 patches, up to 30% each | 87% | 0% | 0.87 | 140 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 89 |
| 0.10 | +/-8/255 (3%) | 100% | 0% | 1.00 | 101 |
| 0.20 | +/-16/255 (6%) | 100% | 0% | 1.00 | 102 |
| 0.30 | +/-24/255 (9%) | 100% | 0% | 1.00 | 101 |
| 0.50 | +/-40/255 (16%) | 100% | 0% | 1.00 | 102 |
| 0.70 | +/-56/255 (22%) | 100% | 0% | 1.00 | 102 |
| 1.00 | +/-80/255 (31%) | 100% | 0% | 1.00 | 101 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 86 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 85 |
| 0.20 | 112px effective (2x downscale) | 100% | 0% | 1.00 | 116 |
| 0.30 | 74px effective (3x downscale) | 73% | 0% | 0.73 | 170 |
| 0.50 | 56px effective (4x downscale) | 40% | 0% | 0.40 | 229 |
| 0.70 | 44px effective (5x downscale) | 27% | 0% | 0.27 | 265 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 300 |

### Task 9: *put the wine bottle on the rack*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.50 (100ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.20 (20% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 100%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 100% baseline, max deg 100%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 93% baseline, max deg 73%, bp=0.70 (obs std 0.7 @step100)
- **ImageNoiseStressor**: 100% baseline, max deg 87%, bp=0.70 (noise std=35)
- **OcclusionStressor**: 100% baseline, max deg 40%, robust
- **BrightnessShiftStressor**: 87% baseline, max deg 20%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.50 (56px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 138 |
| 0.10 | 20ms (1 steps) | 93% | 0% | 0.93 | 176 |
| 0.20 | 40ms (2 steps) | 93% | 0% | 0.93 | 176 |
| 0.30 | 60ms (3 steps) | 67% | 0% | 0.67 | 222 |
| 0.50 | 100ms (5 steps) | 27% | 0% | 0.27 | 287 |
| 0.70 | 140ms (7 steps) | 13% | 0% | 0.13 | 298 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 300 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 146 |
| 0.10 | 10% drop prob | 60% | 0% | 0.60 | 211 |
| 0.20 | 20% drop prob | 27% | 0% | 0.27 | 259 |
| 0.30 | 30% drop prob | 13% | 0% | 0.13 | 282 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 300 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 300 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 300 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 150 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 300 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 300 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 300 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 300 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 300 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 300 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 140 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 300 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 300 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 300 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 300 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 300 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 300 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 93% | 0% | 0.93 | 161 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 142 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 73% | 0% | 0.73 | 191 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 87% | 0% | 0.87 | 178 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 60% | 0% | 0.60 | 218 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 47% | 0% | 0.47 | 225 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 20% | 0% | 0.20 | 264 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 150 |
| 0.10 | std=5/255 (2%) | 87% | 0% | 0.87 | 216 |
| 0.20 | std=10/255 (4%) | 73% | 0% | 0.73 | 228 |
| 0.30 | std=15/255 (6%) | 67% | 0% | 0.67 | 253 |
| 0.50 | std=25/255 (10%) | 53% | 0% | 0.53 | 240 |
| 0.70 | std=35/255 (14%) | 47% | 0% | 0.47 | 256 |
| 1.00 | std=50/255 (20%) | 13% | 0% | 0.13 | 280 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 139 |
| 0.10 | 1 patches, up to 3% each | 93% | 0% | 0.93 | 178 |
| 0.20 | 1 patches, up to 6% each | 93% | 0% | 0.93 | 202 |
| 0.30 | 1 patches, up to 9% each | 93% | 0% | 0.93 | 212 |
| 0.50 | 2 patches, up to 15% each | 87% | 0% | 0.87 | 200 |
| 0.70 | 3 patches, up to 21% each | 60% | 0% | 0.60 | 235 |
| 1.00 | 5 patches, up to 30% each | 80% | 0% | 0.80 | 215 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 87% | 0% | 0.87 | 160 |
| 0.10 | +/-8/255 (3%) | 87% | 0% | 0.87 | 197 |
| 0.20 | +/-16/255 (6%) | 93% | 0% | 0.93 | 196 |
| 0.30 | +/-24/255 (9%) | 87% | 0% | 0.87 | 203 |
| 0.50 | +/-40/255 (16%) | 67% | 0% | 0.67 | 229 |
| 0.70 | +/-56/255 (22%) | 87% | 0% | 0.87 | 205 |
| 1.00 | +/-80/255 (31%) | 93% | 0% | 0.93 | 197 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 143 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 138 |
| 0.20 | 112px effective (2x downscale) | 93% | 0% | 0.93 | 189 |
| 0.30 | 74px effective (3x downscale) | 87% | 0% | 0.87 | 219 |
| 0.50 | 56px effective (4x downscale) | 0% | 0% | 0.00 | 300 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 300 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 300 |


## libero_10

### Task 0: *put both the alphabet soup and the tomato sauce in the basket*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.50 (100ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.50 (50% dropout)
- **PhysicsShiftStressor**: 93% baseline, max deg 0%, robust
- **EmbodimentStressor**: 80% baseline, max deg 0%, robust
- **LongHorizonDriftStressor**: 93% baseline, max deg 93%, bp=0.30 (obs std 0.3 @step100)
- **ImageNoiseStressor**: 100% baseline, max deg 93%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 100% baseline, max deg 27%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 33%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.30 (74px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 270 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 300 |
| 0.20 | 40ms (2 steps) | 87% | 0% | 0.87 | 374 |
| 0.30 | 60ms (3 steps) | 87% | 0% | 0.87 | 415 |
| 0.50 | 100ms (5 steps) | 13% | 0% | 0.13 | 511 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 520 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 520 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 270 |
| 0.10 | 10% drop prob | 93% | 0% | 0.93 | 300 |
| 0.20 | 20% drop prob | 80% | 0% | 0.80 | 401 |
| 0.30 | 30% drop prob | 67% | 0% | 0.67 | 449 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 520 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 520 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 520 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 93% | 0% | 0.93 | 289 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 100% | 0% | 1.00 | 1 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 100% | 0% | 1.00 | 1 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 100% | 0% | 1.00 | 2 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 100% | 0% | 1.00 | 2 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 100% | 0% | 1.00 | 1 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 100% | 0% | 1.00 | 1 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 80% | 0% | 0.80 | 329 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 100% | 0% | 1.00 | 4 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 100% | 0% | 1.00 | 4 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 100% | 0% | 1.00 | 39 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 100% | 0% | 1.00 | 2 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 93% | 0% | 0.93 | 71 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 87% | 0% | 0.87 | 120 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 93% | 0% | 0.93 | 281 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 93% | 0% | 0.93 | 287 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 93% | 0% | 0.93 | 307 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 40% | 0% | 0.40 | 433 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 7% | 0% | 0.07 | 504 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 0% | 0% | 0.00 | 520 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 0% | 0% | 0.00 | 520 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 270 |
| 0.10 | std=5/255 (2%) | 87% | 0% | 0.87 | 388 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 349 |
| 0.30 | std=15/255 (6%) | 80% | 0% | 0.80 | 403 |
| 0.50 | std=25/255 (10%) | 93% | 0% | 0.93 | 398 |
| 0.70 | std=35/255 (14%) | 80% | 0% | 0.80 | 395 |
| 1.00 | std=50/255 (20%) | 7% | 0% | 0.07 | 516 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 266 |
| 0.10 | 1 patches, up to 3% each | 80% | 0% | 0.80 | 399 |
| 0.20 | 1 patches, up to 6% each | 73% | 0% | 0.73 | 397 |
| 0.30 | 1 patches, up to 9% each | 80% | 0% | 0.80 | 398 |
| 0.50 | 2 patches, up to 15% each | 73% | 0% | 0.73 | 389 |
| 0.70 | 3 patches, up to 21% each | 80% | 0% | 0.80 | 395 |
| 1.00 | 5 patches, up to 30% each | 73% | 0% | 0.73 | 415 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 273 |
| 0.10 | +/-8/255 (3%) | 87% | 0% | 0.87 | 371 |
| 0.20 | +/-16/255 (6%) | 67% | 0% | 0.67 | 398 |
| 0.30 | +/-24/255 (9%) | 67% | 0% | 0.67 | 390 |
| 0.50 | +/-40/255 (16%) | 80% | 0% | 0.80 | 376 |
| 0.70 | +/-56/255 (22%) | 80% | 0% | 0.80 | 371 |
| 1.00 | +/-80/255 (31%) | 87% | 0% | 0.87 | 384 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 270 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 270 |
| 0.20 | 112px effective (2x downscale) | 80% | 0% | 0.80 | 421 |
| 0.30 | 74px effective (3x downscale) | 33% | 0% | 0.33 | 499 |
| 0.50 | 56px effective (4x downscale) | 7% | 0% | 0.07 | 509 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 520 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 520 |


## Breakpoint Comparison

Average intensity at which success rate drops below 50%:

- **LatencyStressor**: libero_spatial: 0.50, libero_object: 0.57, libero_goal: 0.75, libero_10: 0.50
- **DropoutStressor**: libero_spatial: 0.37, libero_object: 0.43, libero_goal: 0.33, libero_10: 0.50
- **PhysicsShiftStressor**: libero_spatial: 0.10, libero_object: 0.10, libero_goal: 0.10, libero_10: robust
- **EmbodimentStressor**: libero_spatial: 0.10, libero_object: 0.10, libero_goal: 0.10, libero_10: robust
- **LongHorizonDriftStressor**: libero_spatial: 0.85, libero_object: 1.00, libero_goal: 0.70, libero_10: 0.30
- **ImageNoiseStressor**: libero_spatial: 0.85, libero_object: 1.00, libero_goal: 0.90, libero_10: 1.00
- **OcclusionStressor**: libero_spatial: robust, libero_object: robust, libero_goal: robust, libero_10: robust
- **BrightnessShiftStressor**: libero_spatial: robust, libero_object: robust, libero_goal: robust, libero_10: robust
- **ResolutionStressor**: libero_spatial: 0.43, libero_object: 0.50, libero_goal: 0.50, libero_10: 0.30

---

*Report generated by Trace Robotics v0.1.0*
*https://tracerobotics.com*