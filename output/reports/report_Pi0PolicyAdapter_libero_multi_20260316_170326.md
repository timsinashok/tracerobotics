# Trace Robotics — Multi-Task Robustness Report

**Policy:** Pi0PolicyAdapter  
**Suites:** libero_spatial, libero_object, libero_goal, libero_10  
**Task IDs:** [0, 4, 9]  
**Total tasks evaluated:** 12  
**Modalities:** vision, proprioception  
**Generated:** 2026-03-18 03:25  
**Control frequency:** 50Hz (20ms per step)

---

## Cross-Suite Summary

| Stressor | libero_spatial | libero_object | libero_goal | libero_10 |
|---|---|---|---|---|
| LatencyStressor | 100% (bp=0.50) | 100% (bp=0.57) | 96% (bp=0.85) | 98% (bp=0.50) |
| DropoutStressor | 98% (bp=0.40) | 100% (bp=0.50) | 96% (bp=0.33) | 96% (bp=0.40) |
| PhysicsShiftStressor | 96% (robust) | 100% (robust) | 96% (robust) | 98% (robust) |
| EmbodimentStressor | 100% (robust) | 100% (robust) | 98% (bp=0.50) | 98% (bp=0.70) |
| LongHorizonDriftStressor | 96% (bp=1.00) | 98% (bp=1.00) | 100% (bp=0.85) | 98% (bp=0.30) |
| ImageNoiseStressor | 96% (bp=1.00) | 98% (robust) | 96% (bp=1.00) | 93% (bp=1.00) |
| OcclusionStressor | 96% (robust) | 100% (robust) | 98% (robust) | 100% (robust) |
| BrightnessShiftStressor | 98% (robust) | 100% (robust) | 96% (robust) | 100% (robust) |
| ResolutionStressor | 98% (bp=0.73) | 98% (bp=0.80) | 98% (bp=0.60) | 98% (bp=0.50) |

## libero_spatial

### Task 0: *pick up the black bowl between the plate and the ramekin and place it on the plate*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.50 (100ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.70 (70% dropout)
- **PhysicsShiftStressor**: 93% baseline, max deg 0%, robust
- **EmbodimentStressor**: 100% baseline, max deg 0%, robust
- **LongHorizonDriftStressor**: 100% baseline, max deg 20%, robust
- **ImageNoiseStressor**: 100% baseline, max deg 0%, robust
- **OcclusionStressor**: 100% baseline, max deg 13%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 0%, robust
- **ResolutionStressor**: 100% baseline, max deg 87%, bp=1.00 (28px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 86 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 96 |
| 0.20 | 40ms (2 steps) | 87% | 0% | 0.87 | 132 |
| 0.30 | 60ms (3 steps) | 67% | 0% | 0.67 | 179 |
| 0.50 | 100ms (5 steps) | 7% | 0% | 0.07 | 219 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 220 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 220 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 82 |
| 0.10 | 10% drop prob | 100% | 0% | 1.00 | 93 |
| 0.20 | 20% drop prob | 80% | 0% | 0.80 | 118 |
| 0.30 | 30% drop prob | 87% | 0% | 0.87 | 128 |
| 0.50 | 50% drop prob | 60% | 0% | 0.60 | 176 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 220 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 220 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 93% | 0% | 0.93 | 92 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 100% | 0% | 1.00 | 84 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 100% | 0% | 1.00 | 83 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 100% | 0% | 1.00 | 82 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 100% | 0% | 1.00 | 86 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 100% | 0% | 1.00 | 78 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 93% | 0% | 0.93 | 96 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 78 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 100% | 0% | 1.00 | 85 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 100% | 0% | 1.00 | 84 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 100% | 0% | 1.00 | 84 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 100% | 0% | 1.00 | 84 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 100% | 0% | 1.00 | 83 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 100% | 0% | 1.00 | 94 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 82 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 83 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 83 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 100% | 0% | 1.00 | 85 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 100% | 0% | 1.00 | 90 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 87% | 0% | 0.87 | 117 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 80% | 0% | 0.80 | 128 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 78 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 83 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 79 |
| 0.30 | std=15/255 (6%) | 100% | 0% | 1.00 | 79 |
| 0.50 | std=25/255 (10%) | 100% | 0% | 1.00 | 78 |
| 0.70 | std=35/255 (14%) | 100% | 0% | 1.00 | 78 |
| 1.00 | std=50/255 (20%) | 100% | 0% | 1.00 | 82 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 82 |
| 0.10 | 1 patches, up to 3% each | 100% | 0% | 1.00 | 82 |
| 0.20 | 1 patches, up to 6% each | 100% | 0% | 1.00 | 82 |
| 0.30 | 1 patches, up to 9% each | 100% | 0% | 1.00 | 81 |
| 0.50 | 2 patches, up to 15% each | 100% | 0% | 1.00 | 81 |
| 0.70 | 3 patches, up to 21% each | 100% | 0% | 1.00 | 80 |
| 1.00 | 5 patches, up to 30% each | 87% | 0% | 0.87 | 98 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 80 |
| 0.10 | +/-8/255 (3%) | 100% | 0% | 1.00 | 83 |
| 0.20 | +/-16/255 (6%) | 100% | 0% | 1.00 | 83 |
| 0.30 | +/-24/255 (9%) | 100% | 0% | 1.00 | 79 |
| 0.50 | +/-40/255 (16%) | 100% | 0% | 1.00 | 83 |
| 0.70 | +/-56/255 (22%) | 100% | 0% | 1.00 | 80 |
| 1.00 | +/-80/255 (31%) | 100% | 0% | 1.00 | 86 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 81 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 85 |
| 0.20 | 112px effective (2x downscale) | 100% | 0% | 1.00 | 84 |
| 0.30 | 74px effective (3x downscale) | 100% | 0% | 1.00 | 85 |
| 0.50 | 56px effective (4x downscale) | 80% | 0% | 0.80 | 140 |
| 0.70 | 44px effective (5x downscale) | 73% | 0% | 0.73 | 151 |
| 1.00 | 28px effective (8x downscale) | 13% | 0% | 0.13 | 206 |

### Task 4: *pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.50 (100ms)
- **DropoutStressor**: 93% baseline, max deg 93%, bp=0.20 (20% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 27%, robust
- **EmbodimentStressor**: 100% baseline, max deg 27%, robust
- **LongHorizonDriftStressor**: 100% baseline, max deg 80%, bp=1.00 (obs std 1.0 @step100)
- **ImageNoiseStressor**: 93% baseline, max deg 67%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 100% baseline, max deg 13%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 13%, robust
- **ResolutionStressor**: 93% baseline, max deg 93%, bp=0.70 (44px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 126 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 138 |
| 0.20 | 40ms (2 steps) | 93% | 0% | 0.93 | 147 |
| 0.30 | 60ms (3 steps) | 73% | 0% | 0.73 | 172 |
| 0.50 | 100ms (5 steps) | 13% | 0% | 0.13 | 213 |
| 0.70 | 140ms (7 steps) | 7% | 0% | 0.07 | 219 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 220 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 93% | 0% | 0.93 | 131 |
| 0.10 | 10% drop prob | 87% | 0% | 0.87 | 149 |
| 0.20 | 20% drop prob | 40% | 0% | 0.40 | 195 |
| 0.30 | 30% drop prob | 20% | 0% | 0.20 | 204 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 220 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 220 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 220 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 131 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 93% | 0% | 0.93 | 133 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 93% | 0% | 0.93 | 132 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 100% | 0% | 1.00 | 127 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 93% | 0% | 0.93 | 136 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 100% | 0% | 1.00 | 131 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 73% | 0% | 0.73 | 150 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 130 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 93% | 0% | 0.93 | 135 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 93% | 0% | 0.93 | 135 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 80% | 0% | 0.80 | 150 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 73% | 0% | 0.73 | 153 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 87% | 0% | 0.87 | 135 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 80% | 0% | 0.80 | 154 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 130 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 93% | 0% | 0.93 | 136 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 93% | 0% | 0.93 | 132 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 87% | 0% | 0.87 | 140 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 80% | 0% | 0.80 | 152 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 67% | 0% | 0.67 | 163 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 20% | 0% | 0.20 | 204 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 93% | 0% | 0.93 | 133 |
| 0.10 | std=5/255 (2%) | 87% | 0% | 0.87 | 141 |
| 0.20 | std=10/255 (4%) | 93% | 0% | 0.93 | 135 |
| 0.30 | std=15/255 (6%) | 87% | 0% | 0.87 | 140 |
| 0.50 | std=25/255 (10%) | 87% | 0% | 0.87 | 146 |
| 0.70 | std=35/255 (14%) | 87% | 0% | 0.87 | 148 |
| 1.00 | std=50/255 (20%) | 27% | 0% | 0.27 | 203 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 128 |
| 0.10 | 1 patches, up to 3% each | 93% | 0% | 0.93 | 132 |
| 0.20 | 1 patches, up to 6% each | 100% | 0% | 1.00 | 127 |
| 0.30 | 1 patches, up to 9% each | 100% | 0% | 1.00 | 128 |
| 0.50 | 2 patches, up to 15% each | 87% | 0% | 0.87 | 139 |
| 0.70 | 3 patches, up to 21% each | 93% | 0% | 0.93 | 136 |
| 1.00 | 5 patches, up to 30% each | 93% | 0% | 0.93 | 141 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 125 |
| 0.10 | +/-8/255 (3%) | 87% | 0% | 0.87 | 137 |
| 0.20 | +/-16/255 (6%) | 100% | 0% | 1.00 | 130 |
| 0.30 | +/-24/255 (9%) | 93% | 0% | 0.93 | 133 |
| 0.50 | +/-40/255 (16%) | 100% | 0% | 1.00 | 131 |
| 0.70 | +/-56/255 (22%) | 93% | 0% | 0.93 | 136 |
| 1.00 | +/-80/255 (31%) | 93% | 0% | 0.93 | 134 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 93% | 0% | 0.93 | 135 |
| 0.10 | 224px (native) | 93% | 0% | 0.93 | 135 |
| 0.20 | 112px effective (2x downscale) | 100% | 0% | 1.00 | 135 |
| 0.30 | 74px effective (3x downscale) | 87% | 0% | 0.87 | 167 |
| 0.50 | 56px effective (4x downscale) | 53% | 0% | 0.53 | 187 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 220 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 220 |

### Task 9: *pick up the black bowl on the wooden cabinet and place it on the plate*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.50 (100ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.30 (30% dropout)
- **PhysicsShiftStressor**: 93% baseline, max deg 7%, robust
- **EmbodimentStressor**: 100% baseline, max deg 27%, robust
- **LongHorizonDriftStressor**: 87% baseline, max deg 40%, bp=1.00 (obs std 1.0 @step100)
- **ImageNoiseStressor**: 93% baseline, max deg 87%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 87% baseline, max deg 0%, robust
- **BrightnessShiftStressor**: 93% baseline, max deg 13%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.50 (56px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 121 |
| 0.10 | 20ms (1 steps) | 93% | 0% | 0.93 | 141 |
| 0.20 | 40ms (2 steps) | 100% | 0% | 1.00 | 141 |
| 0.30 | 60ms (3 steps) | 100% | 0% | 1.00 | 152 |
| 0.50 | 100ms (5 steps) | 0% | 0% | 0.00 | 220 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 220 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 220 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 123 |
| 0.10 | 10% drop prob | 100% | 0% | 1.00 | 143 |
| 0.20 | 20% drop prob | 67% | 0% | 0.67 | 170 |
| 0.30 | 30% drop prob | 27% | 0% | 0.27 | 207 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 220 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 220 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 220 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 93% | 0% | 0.93 | 126 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 93% | 0% | 0.93 | 126 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 93% | 0% | 0.93 | 129 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 93% | 0% | 0.93 | 127 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 93% | 0% | 0.93 | 126 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 100% | 0% | 1.00 | 128 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 87% | 0% | 0.87 | 133 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 121 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 100% | 0% | 1.00 | 122 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 100% | 0% | 1.00 | 125 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 100% | 0% | 1.00 | 124 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 100% | 0% | 1.00 | 129 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 93% | 0% | 0.93 | 136 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 73% | 0% | 0.73 | 150 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 87% | 0% | 0.87 | 134 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 121 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 121 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 93% | 0% | 0.93 | 128 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 87% | 0% | 0.87 | 143 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 60% | 0% | 0.60 | 170 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 47% | 0% | 0.47 | 183 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 93% | 0% | 0.93 | 126 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 122 |
| 0.20 | std=10/255 (4%) | 93% | 0% | 0.93 | 126 |
| 0.30 | std=15/255 (6%) | 87% | 0% | 0.87 | 135 |
| 0.50 | std=25/255 (10%) | 100% | 0% | 1.00 | 121 |
| 0.70 | std=35/255 (14%) | 93% | 0% | 0.93 | 128 |
| 1.00 | std=50/255 (20%) | 7% | 0% | 0.07 | 216 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 87% | 0% | 0.87 | 134 |
| 0.10 | 1 patches, up to 3% each | 87% | 0% | 0.87 | 133 |
| 0.20 | 1 patches, up to 6% each | 87% | 0% | 0.87 | 134 |
| 0.30 | 1 patches, up to 9% each | 87% | 0% | 0.87 | 135 |
| 0.50 | 2 patches, up to 15% each | 93% | 0% | 0.93 | 130 |
| 0.70 | 3 patches, up to 21% each | 100% | 0% | 1.00 | 121 |
| 1.00 | 5 patches, up to 30% each | 100% | 0% | 1.00 | 122 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 93% | 0% | 0.93 | 130 |
| 0.10 | +/-8/255 (3%) | 93% | 0% | 0.93 | 128 |
| 0.20 | +/-16/255 (6%) | 93% | 0% | 0.93 | 126 |
| 0.30 | +/-24/255 (9%) | 100% | 0% | 1.00 | 120 |
| 0.50 | +/-40/255 (16%) | 93% | 0% | 0.93 | 125 |
| 0.70 | +/-56/255 (22%) | 80% | 0% | 0.80 | 140 |
| 1.00 | +/-80/255 (31%) | 87% | 0% | 0.87 | 131 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 124 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 123 |
| 0.20 | 112px effective (2x downscale) | 100% | 0% | 1.00 | 122 |
| 0.30 | 74px effective (3x downscale) | 93% | 0% | 0.93 | 145 |
| 0.50 | 56px effective (4x downscale) | 40% | 0% | 0.40 | 189 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 220 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 220 |


## libero_object

### Task 0: *pick up the alphabet soup and place it in the basket*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.70 (140ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.50 (50% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 13%, robust
- **EmbodimentStressor**: 100% baseline, max deg 13%, robust
- **LongHorizonDriftStressor**: 100% baseline, max deg 40%, robust
- **ImageNoiseStressor**: 93% baseline, max deg 0%, robust
- **OcclusionStressor**: 100% baseline, max deg 7%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 7%, robust
- **ResolutionStressor**: 93% baseline, max deg 93%, bp=0.70 (44px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 150 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 155 |
| 0.20 | 40ms (2 steps) | 100% | 0% | 1.00 | 163 |
| 0.30 | 60ms (3 steps) | 93% | 0% | 0.93 | 204 |
| 0.50 | 100ms (5 steps) | 73% | 0% | 0.73 | 258 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 280 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 280 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 149 |
| 0.10 | 10% drop prob | 100% | 0% | 1.00 | 168 |
| 0.20 | 20% drop prob | 100% | 0% | 1.00 | 181 |
| 0.30 | 30% drop prob | 67% | 0% | 0.67 | 247 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 280 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 280 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 280 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 142 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 93% | 0% | 0.93 | 156 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 100% | 0% | 1.00 | 145 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 100% | 0% | 1.00 | 154 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 100% | 0% | 1.00 | 154 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 87% | 0% | 0.87 | 169 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 100% | 0% | 1.00 | 151 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 147 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 100% | 0% | 1.00 | 147 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 100% | 0% | 1.00 | 149 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 93% | 0% | 0.93 | 156 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 100% | 0% | 1.00 | 154 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 87% | 0% | 0.87 | 172 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 87% | 0% | 0.87 | 182 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 147 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 155 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 149 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 93% | 0% | 0.93 | 172 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 93% | 0% | 0.93 | 181 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 80% | 0% | 0.80 | 199 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 60% | 0% | 0.60 | 211 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 93% | 0% | 0.93 | 161 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 146 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 146 |
| 0.30 | std=15/255 (6%) | 100% | 0% | 1.00 | 158 |
| 0.50 | std=25/255 (10%) | 100% | 0% | 1.00 | 145 |
| 0.70 | std=35/255 (14%) | 93% | 0% | 0.93 | 154 |
| 1.00 | std=50/255 (20%) | 93% | 0% | 0.93 | 186 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 156 |
| 0.10 | 1 patches, up to 3% each | 100% | 0% | 1.00 | 145 |
| 0.20 | 1 patches, up to 6% each | 100% | 0% | 1.00 | 148 |
| 0.30 | 1 patches, up to 9% each | 100% | 0% | 1.00 | 145 |
| 0.50 | 2 patches, up to 15% each | 93% | 0% | 0.93 | 171 |
| 0.70 | 3 patches, up to 21% each | 100% | 0% | 1.00 | 152 |
| 1.00 | 5 patches, up to 30% each | 93% | 0% | 0.93 | 186 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 149 |
| 0.10 | +/-8/255 (3%) | 100% | 0% | 1.00 | 146 |
| 0.20 | +/-16/255 (6%) | 100% | 0% | 1.00 | 151 |
| 0.30 | +/-24/255 (9%) | 100% | 0% | 1.00 | 147 |
| 0.50 | +/-40/255 (16%) | 100% | 0% | 1.00 | 148 |
| 0.70 | +/-56/255 (22%) | 93% | 0% | 0.93 | 153 |
| 1.00 | +/-80/255 (31%) | 93% | 0% | 0.93 | 163 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 93% | 0% | 0.93 | 157 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 149 |
| 0.20 | 112px effective (2x downscale) | 93% | 0% | 0.93 | 162 |
| 0.30 | 74px effective (3x downscale) | 80% | 0% | 0.80 | 219 |
| 0.50 | 56px effective (4x downscale) | 87% | 0% | 0.87 | 206 |
| 0.70 | 44px effective (5x downscale) | 40% | 0% | 0.40 | 251 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 280 |

### Task 4: *pick up the ketchup and place it in the basket*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.50 (100ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.50 (50% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 0%, robust
- **EmbodimentStressor**: 100% baseline, max deg 13%, robust
- **LongHorizonDriftStressor**: 100% baseline, max deg 33%, robust
- **ImageNoiseStressor**: 100% baseline, max deg 7%, robust
- **OcclusionStressor**: 100% baseline, max deg 0%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 0%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.70 (44px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 143 |
| 0.10 | 20ms (1 steps) | 93% | 0% | 0.93 | 162 |
| 0.20 | 40ms (2 steps) | 100% | 0% | 1.00 | 163 |
| 0.30 | 60ms (3 steps) | 80% | 0% | 0.80 | 210 |
| 0.50 | 100ms (5 steps) | 40% | 0% | 0.40 | 263 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 280 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 280 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 149 |
| 0.10 | 10% drop prob | 100% | 0% | 1.00 | 160 |
| 0.20 | 20% drop prob | 80% | 0% | 0.80 | 191 |
| 0.30 | 30% drop prob | 60% | 0% | 0.60 | 236 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 280 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 280 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 280 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 148 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 100% | 0% | 1.00 | 149 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 100% | 0% | 1.00 | 155 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 100% | 0% | 1.00 | 153 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 100% | 0% | 1.00 | 158 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 100% | 0% | 1.00 | 149 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 100% | 0% | 1.00 | 151 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 149 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 100% | 0% | 1.00 | 147 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 100% | 0% | 1.00 | 152 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 100% | 0% | 1.00 | 154 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 100% | 0% | 1.00 | 145 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 100% | 0% | 1.00 | 155 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 87% | 0% | 0.87 | 162 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 156 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 93% | 0% | 0.93 | 164 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 151 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 100% | 0% | 1.00 | 149 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 87% | 0% | 0.87 | 174 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 73% | 0% | 0.73 | 200 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 67% | 0% | 0.67 | 207 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 151 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 152 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 152 |
| 0.30 | std=15/255 (6%) | 100% | 0% | 1.00 | 147 |
| 0.50 | std=25/255 (10%) | 100% | 0% | 1.00 | 147 |
| 0.70 | std=35/255 (14%) | 100% | 0% | 1.00 | 147 |
| 1.00 | std=50/255 (20%) | 93% | 0% | 0.93 | 166 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 150 |
| 0.10 | 1 patches, up to 3% each | 100% | 0% | 1.00 | 159 |
| 0.20 | 1 patches, up to 6% each | 100% | 0% | 1.00 | 149 |
| 0.30 | 1 patches, up to 9% each | 100% | 0% | 1.00 | 149 |
| 0.50 | 2 patches, up to 15% each | 100% | 0% | 1.00 | 157 |
| 0.70 | 3 patches, up to 21% each | 100% | 0% | 1.00 | 151 |
| 1.00 | 5 patches, up to 30% each | 100% | 0% | 1.00 | 150 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 146 |
| 0.10 | +/-8/255 (3%) | 100% | 0% | 1.00 | 151 |
| 0.20 | +/-16/255 (6%) | 100% | 0% | 1.00 | 156 |
| 0.30 | +/-24/255 (9%) | 100% | 0% | 1.00 | 146 |
| 0.50 | +/-40/255 (16%) | 100% | 0% | 1.00 | 154 |
| 0.70 | +/-56/255 (22%) | 100% | 0% | 1.00 | 149 |
| 1.00 | +/-80/255 (31%) | 100% | 0% | 1.00 | 152 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 153 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 150 |
| 0.20 | 112px effective (2x downscale) | 100% | 0% | 1.00 | 149 |
| 0.30 | 74px effective (3x downscale) | 93% | 0% | 0.93 | 168 |
| 0.50 | 56px effective (4x downscale) | 73% | 0% | 0.73 | 209 |
| 0.70 | 44px effective (5x downscale) | 33% | 0% | 0.33 | 261 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 280 |

### Task 9: *pick up the orange juice and place it in the basket*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.50 (100ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.50 (50% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 20%, robust
- **EmbodimentStressor**: 100% baseline, max deg 27%, robust
- **LongHorizonDriftStressor**: 93% baseline, max deg 47%, bp=1.00 (obs std 1.0 @step100)
- **ImageNoiseStressor**: 100% baseline, max deg 0%, robust
- **OcclusionStressor**: 100% baseline, max deg 7%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 0%, robust
- **ResolutionStressor**: 100% baseline, max deg 67%, bp=1.00 (28px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 119 |
| 0.10 | 20ms (1 steps) | 93% | 0% | 0.93 | 149 |
| 0.20 | 40ms (2 steps) | 87% | 0% | 0.87 | 167 |
| 0.30 | 60ms (3 steps) | 87% | 0% | 0.87 | 187 |
| 0.50 | 100ms (5 steps) | 7% | 0% | 0.07 | 273 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 280 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 280 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 128 |
| 0.10 | 10% drop prob | 100% | 0% | 1.00 | 139 |
| 0.20 | 20% drop prob | 87% | 0% | 0.87 | 184 |
| 0.30 | 30% drop prob | 60% | 0% | 0.60 | 221 |
| 0.50 | 50% drop prob | 33% | 0% | 0.33 | 263 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 280 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 280 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 130 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 93% | 0% | 0.93 | 142 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 100% | 0% | 1.00 | 128 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 93% | 0% | 0.93 | 140 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 100% | 0% | 1.00 | 131 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 100% | 0% | 1.00 | 125 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 80% | 0% | 0.80 | 157 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 124 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 100% | 0% | 1.00 | 132 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 100% | 0% | 1.00 | 138 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 93% | 0% | 0.93 | 137 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 87% | 0% | 0.87 | 152 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 93% | 0% | 0.93 | 144 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 73% | 0% | 0.73 | 168 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 93% | 0% | 0.93 | 148 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 130 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 123 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 100% | 0% | 1.00 | 130 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 93% | 0% | 0.93 | 140 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 67% | 0% | 0.67 | 181 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 47% | 0% | 0.47 | 220 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 130 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 133 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 129 |
| 0.30 | std=15/255 (6%) | 100% | 0% | 1.00 | 123 |
| 0.50 | std=25/255 (10%) | 100% | 0% | 1.00 | 120 |
| 0.70 | std=35/255 (14%) | 100% | 0% | 1.00 | 130 |
| 1.00 | std=50/255 (20%) | 100% | 0% | 1.00 | 130 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 135 |
| 0.10 | 1 patches, up to 3% each | 100% | 0% | 1.00 | 121 |
| 0.20 | 1 patches, up to 6% each | 100% | 0% | 1.00 | 131 |
| 0.30 | 1 patches, up to 9% each | 100% | 0% | 1.00 | 125 |
| 0.50 | 2 patches, up to 15% each | 93% | 0% | 0.93 | 142 |
| 0.70 | 3 patches, up to 21% each | 100% | 0% | 1.00 | 124 |
| 1.00 | 5 patches, up to 30% each | 100% | 0% | 1.00 | 127 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 125 |
| 0.10 | +/-8/255 (3%) | 100% | 0% | 1.00 | 130 |
| 0.20 | +/-16/255 (6%) | 100% | 0% | 1.00 | 127 |
| 0.30 | +/-24/255 (9%) | 100% | 0% | 1.00 | 123 |
| 0.50 | +/-40/255 (16%) | 100% | 0% | 1.00 | 125 |
| 0.70 | +/-56/255 (22%) | 100% | 0% | 1.00 | 126 |
| 1.00 | +/-80/255 (31%) | 100% | 0% | 1.00 | 133 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 132 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 126 |
| 0.20 | 112px effective (2x downscale) | 100% | 0% | 1.00 | 138 |
| 0.30 | 74px effective (3x downscale) | 100% | 0% | 1.00 | 128 |
| 0.50 | 56px effective (4x downscale) | 100% | 0% | 1.00 | 154 |
| 0.70 | 44px effective (5x downscale) | 100% | 0% | 1.00 | 145 |
| 1.00 | 28px effective (8x downscale) | 33% | 0% | 0.33 | 243 |


## libero_goal

### Task 0: *open the middle drawer of the cabinet*

- **LatencyStressor**: 93% baseline, max deg 7%, robust
- **DropoutStressor**: 93% baseline, max deg 93%, bp=0.30 (30% dropout)
- **PhysicsShiftStressor**: 93% baseline, max deg 13%, robust
- **EmbodimentStressor**: 93% baseline, max deg 87%, bp=0.30 (links 0.97-1.03x, gains 0.91-1.09x)
- **LongHorizonDriftStressor**: 100% baseline, max deg 87%, bp=0.70 (obs std 0.7 @step100)
- **ImageNoiseStressor**: 93% baseline, max deg 93%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 100% baseline, max deg 7%, robust
- **BrightnessShiftStressor**: 93% baseline, max deg 7%, robust
- **ResolutionStressor**: 93% baseline, max deg 93%, bp=0.30 (74px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 93% | 0% | 0.93 | 144 |
| 0.10 | 20ms (1 steps) | 93% | 0% | 0.93 | 137 |
| 0.20 | 40ms (2 steps) | 93% | 0% | 0.93 | 142 |
| 0.30 | 60ms (3 steps) | 93% | 0% | 0.93 | 152 |
| 0.50 | 100ms (5 steps) | 87% | 0% | 0.87 | 173 |
| 0.70 | 140ms (7 steps) | 93% | 0% | 0.93 | 186 |
| 1.00 | 200ms (10 steps) | 87% | 0% | 0.87 | 193 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 93% | 0% | 0.93 | 130 |
| 0.10 | 10% drop prob | 87% | 0% | 0.87 | 162 |
| 0.20 | 20% drop prob | 80% | 0% | 0.80 | 179 |
| 0.30 | 30% drop prob | 20% | 0% | 0.20 | 276 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 300 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 300 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 300 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 93% | 0% | 0.93 | 129 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 93% | 0% | 0.93 | 132 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 93% | 0% | 0.93 | 131 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 93% | 0% | 0.93 | 133 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 100% | 0% | 1.00 | 128 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 100% | 0% | 1.00 | 130 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 80% | 0% | 0.80 | 157 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 93% | 0% | 0.93 | 131 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 100% | 0% | 1.00 | 122 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 93% | 0% | 0.93 | 140 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 40% | 0% | 0.40 | 229 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 33% | 0% | 0.33 | 245 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 20% | 0% | 0.20 | 267 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 7% | 0% | 0.07 | 290 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 123 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 93% | 0% | 0.93 | 135 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 93% | 0% | 0.93 | 132 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 80% | 0% | 0.80 | 156 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 53% | 0% | 0.53 | 216 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 20% | 0% | 0.20 | 268 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 13% | 0% | 0.13 | 276 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 93% | 0% | 0.93 | 133 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 124 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 118 |
| 0.30 | std=15/255 (6%) | 100% | 0% | 1.00 | 121 |
| 0.50 | std=25/255 (10%) | 100% | 0% | 1.00 | 127 |
| 0.70 | std=35/255 (14%) | 87% | 0% | 0.87 | 143 |
| 1.00 | std=50/255 (20%) | 0% | 0% | 0.00 | 300 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 121 |
| 0.10 | 1 patches, up to 3% each | 100% | 0% | 1.00 | 118 |
| 0.20 | 1 patches, up to 6% each | 93% | 0% | 0.93 | 134 |
| 0.30 | 1 patches, up to 9% each | 93% | 0% | 0.93 | 133 |
| 0.50 | 2 patches, up to 15% each | 93% | 0% | 0.93 | 133 |
| 0.70 | 3 patches, up to 21% each | 100% | 0% | 1.00 | 122 |
| 1.00 | 5 patches, up to 30% each | 100% | 0% | 1.00 | 140 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 93% | 0% | 0.93 | 133 |
| 0.10 | +/-8/255 (3%) | 93% | 0% | 0.93 | 131 |
| 0.20 | +/-16/255 (6%) | 93% | 0% | 0.93 | 142 |
| 0.30 | +/-24/255 (9%) | 93% | 0% | 0.93 | 131 |
| 0.50 | +/-40/255 (16%) | 87% | 0% | 0.87 | 142 |
| 0.70 | +/-56/255 (22%) | 93% | 0% | 0.93 | 136 |
| 1.00 | +/-80/255 (31%) | 93% | 0% | 0.93 | 129 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 93% | 0% | 0.93 | 132 |
| 0.10 | 224px (native) | 93% | 0% | 0.93 | 132 |
| 0.20 | 112px effective (2x downscale) | 100% | 0% | 1.00 | 122 |
| 0.30 | 74px effective (3x downscale) | 47% | 0% | 0.47 | 221 |
| 0.50 | 56px effective (4x downscale) | 0% | 0% | 0.00 | 300 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 300 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 300 |

### Task 4: *put the bowl on top of the cabinet*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=1.00 (200ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.50 (50% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 7%, robust
- **EmbodimentStressor**: 100% baseline, max deg 7%, robust
- **LongHorizonDriftStressor**: 100% baseline, max deg 7%, robust
- **ImageNoiseStressor**: 93% baseline, max deg 0%, robust
- **OcclusionStressor**: 100% baseline, max deg 0%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 0%, robust
- **ResolutionStressor**: 100% baseline, max deg 80%, bp=1.00 (28px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 88 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 94 |
| 0.20 | 40ms (2 steps) | 100% | 0% | 1.00 | 104 |
| 0.30 | 60ms (3 steps) | 100% | 0% | 1.00 | 126 |
| 0.50 | 100ms (5 steps) | 73% | 0% | 0.73 | 222 |
| 0.70 | 140ms (7 steps) | 73% | 0% | 0.73 | 240 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 300 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 95 |
| 0.10 | 10% drop prob | 93% | 0% | 0.93 | 107 |
| 0.20 | 20% drop prob | 100% | 0% | 1.00 | 117 |
| 0.30 | 30% drop prob | 67% | 0% | 0.67 | 193 |
| 0.50 | 50% drop prob | 7% | 0% | 0.07 | 294 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 300 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 300 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 86 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 100% | 0% | 1.00 | 87 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 100% | 0% | 1.00 | 87 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 100% | 0% | 1.00 | 88 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 100% | 0% | 1.00 | 91 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 93% | 0% | 0.93 | 101 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 100% | 0% | 1.00 | 85 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 87 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 100% | 0% | 1.00 | 90 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 100% | 0% | 1.00 | 90 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 100% | 0% | 1.00 | 90 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 93% | 0% | 0.93 | 103 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 93% | 0% | 0.93 | 102 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 93% | 0% | 0.93 | 107 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 86 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 87 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 86 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 100% | 0% | 1.00 | 89 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 100% | 0% | 1.00 | 89 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 100% | 0% | 1.00 | 91 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 93% | 0% | 0.93 | 112 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 93% | 0% | 0.93 | 102 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 89 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 87 |
| 0.30 | std=15/255 (6%) | 100% | 0% | 1.00 | 92 |
| 0.50 | std=25/255 (10%) | 100% | 0% | 1.00 | 87 |
| 0.70 | std=35/255 (14%) | 100% | 0% | 1.00 | 89 |
| 1.00 | std=50/255 (20%) | 93% | 0% | 0.93 | 113 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 86 |
| 0.10 | 1 patches, up to 3% each | 100% | 0% | 1.00 | 86 |
| 0.20 | 1 patches, up to 6% each | 100% | 0% | 1.00 | 90 |
| 0.30 | 1 patches, up to 9% each | 100% | 0% | 1.00 | 89 |
| 0.50 | 2 patches, up to 15% each | 100% | 0% | 1.00 | 86 |
| 0.70 | 3 patches, up to 21% each | 100% | 0% | 1.00 | 88 |
| 1.00 | 5 patches, up to 30% each | 100% | 0% | 1.00 | 93 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 88 |
| 0.10 | +/-8/255 (3%) | 100% | 0% | 1.00 | 92 |
| 0.20 | +/-16/255 (6%) | 100% | 0% | 1.00 | 90 |
| 0.30 | +/-24/255 (9%) | 100% | 0% | 1.00 | 91 |
| 0.50 | +/-40/255 (16%) | 100% | 0% | 1.00 | 87 |
| 0.70 | +/-56/255 (22%) | 100% | 0% | 1.00 | 89 |
| 1.00 | +/-80/255 (31%) | 100% | 0% | 1.00 | 87 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 86 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 88 |
| 0.20 | 112px effective (2x downscale) | 100% | 0% | 1.00 | 90 |
| 0.30 | 74px effective (3x downscale) | 93% | 0% | 0.93 | 106 |
| 0.50 | 56px effective (4x downscale) | 100% | 0% | 1.00 | 99 |
| 0.70 | 44px effective (5x downscale) | 93% | 0% | 0.93 | 127 |
| 1.00 | 28px effective (8x downscale) | 20% | 0% | 0.20 | 269 |

### Task 9: *put the wine bottle on the rack*

- **LatencyStressor**: 93% baseline, max deg 93%, bp=0.70 (140ms)
- **DropoutStressor**: 93% baseline, max deg 93%, bp=0.20 (20% dropout)
- **PhysicsShiftStressor**: 93% baseline, max deg 0%, robust
- **EmbodimentStressor**: 100% baseline, max deg 73%, bp=0.70 (links 0.93-1.07x, gains 0.79-1.21x)
- **LongHorizonDriftStressor**: 100% baseline, max deg 67%, bp=1.00 (obs std 1.0 @step100)
- **ImageNoiseStressor**: 100% baseline, max deg 67%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 93% baseline, max deg 0%, robust
- **BrightnessShiftStressor**: 93% baseline, max deg 7%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.50 (56px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 93% | 0% | 0.93 | 159 |
| 0.10 | 20ms (1 steps) | 87% | 0% | 0.87 | 177 |
| 0.20 | 40ms (2 steps) | 80% | 0% | 0.80 | 207 |
| 0.30 | 60ms (3 steps) | 100% | 0% | 1.00 | 196 |
| 0.50 | 100ms (5 steps) | 53% | 0% | 0.53 | 273 |
| 0.70 | 140ms (7 steps) | 13% | 0% | 0.13 | 294 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 300 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 93% | 0% | 0.93 | 148 |
| 0.10 | 10% drop prob | 60% | 0% | 0.60 | 210 |
| 0.20 | 20% drop prob | 7% | 0% | 0.07 | 290 |
| 0.30 | 30% drop prob | 7% | 0% | 0.07 | 289 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 300 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 300 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 300 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 93% | 0% | 0.93 | 154 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 93% | 0% | 0.93 | 164 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 93% | 0% | 0.93 | 152 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 93% | 0% | 0.93 | 151 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 100% | 0% | 1.00 | 154 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 93% | 0% | 0.93 | 157 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 100% | 0% | 1.00 | 160 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 151 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 93% | 0% | 0.93 | 151 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 87% | 0% | 0.87 | 174 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 100% | 0% | 1.00 | 152 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 67% | 0% | 0.67 | 202 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 40% | 0% | 0.40 | 239 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 27% | 0% | 0.27 | 258 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 151 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 93% | 0% | 0.93 | 157 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 93% | 0% | 0.93 | 152 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 60% | 0% | 0.60 | 203 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 53% | 0% | 0.53 | 223 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 67% | 0% | 0.67 | 198 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 33% | 0% | 0.33 | 254 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 140 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 139 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 162 |
| 0.30 | std=15/255 (6%) | 100% | 0% | 1.00 | 162 |
| 0.50 | std=25/255 (10%) | 93% | 0% | 0.93 | 162 |
| 0.70 | std=35/255 (14%) | 73% | 0% | 0.73 | 186 |
| 1.00 | std=50/255 (20%) | 33% | 0% | 0.33 | 250 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 93% | 0% | 0.93 | 157 |
| 0.10 | 1 patches, up to 3% each | 100% | 0% | 1.00 | 149 |
| 0.20 | 1 patches, up to 6% each | 100% | 0% | 1.00 | 146 |
| 0.30 | 1 patches, up to 9% each | 100% | 0% | 1.00 | 141 |
| 0.50 | 2 patches, up to 15% each | 100% | 0% | 1.00 | 144 |
| 0.70 | 3 patches, up to 21% each | 100% | 0% | 1.00 | 147 |
| 1.00 | 5 patches, up to 30% each | 100% | 0% | 1.00 | 158 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 93% | 0% | 0.93 | 153 |
| 0.10 | +/-8/255 (3%) | 100% | 0% | 1.00 | 139 |
| 0.20 | +/-16/255 (6%) | 100% | 0% | 1.00 | 140 |
| 0.30 | +/-24/255 (9%) | 100% | 0% | 1.00 | 148 |
| 0.50 | +/-40/255 (16%) | 100% | 0% | 1.00 | 138 |
| 0.70 | +/-56/255 (22%) | 93% | 0% | 0.93 | 146 |
| 1.00 | +/-80/255 (31%) | 87% | 0% | 0.87 | 167 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 139 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 139 |
| 0.20 | 112px effective (2x downscale) | 100% | 0% | 1.00 | 154 |
| 0.30 | 74px effective (3x downscale) | 93% | 0% | 0.93 | 177 |
| 0.50 | 56px effective (4x downscale) | 0% | 0% | 0.00 | 300 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 300 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 300 |


## libero_10

### Task 0: *put both the alphabet soup and the tomato sauce in the basket*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.50 (100ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.50 (50% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 0%, robust
- **EmbodimentStressor**: 100% baseline, max deg 40%, robust
- **LongHorizonDriftStressor**: 100% baseline, max deg 100%, bp=0.30 (obs std 0.3 @step100)
- **ImageNoiseStressor**: 100% baseline, max deg 100%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 100% baseline, max deg 7%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 7%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.50 (56px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 266 |
| 0.10 | 20ms (1 steps) | 93% | 0% | 0.93 | 318 |
| 0.20 | 40ms (2 steps) | 80% | 0% | 0.80 | 381 |
| 0.30 | 60ms (3 steps) | 67% | 0% | 0.67 | 414 |
| 0.50 | 100ms (5 steps) | 13% | 0% | 0.13 | 517 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 520 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 520 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 268 |
| 0.10 | 10% drop prob | 93% | 0% | 0.93 | 332 |
| 0.20 | 20% drop prob | 93% | 0% | 0.93 | 365 |
| 0.30 | 30% drop prob | 73% | 0% | 0.73 | 425 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 520 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 520 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 520 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 279 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 100% | 0% | 1.00 | 279 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 100% | 0% | 1.00 | 271 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 100% | 0% | 1.00 | 276 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 100% | 0% | 1.00 | 274 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 100% | 0% | 1.00 | 278 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 100% | 0% | 1.00 | 272 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 272 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 100% | 0% | 1.00 | 293 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 93% | 0% | 0.93 | 306 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 100% | 0% | 1.00 | 288 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 93% | 0% | 0.93 | 325 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 60% | 0% | 0.60 | 391 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 60% | 0% | 0.60 | 415 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 266 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 290 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 73% | 0% | 0.73 | 374 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 40% | 0% | 0.40 | 433 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 20% | 0% | 0.20 | 484 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 0% | 0% | 0.00 | 520 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 0% | 0% | 0.00 | 520 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 278 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 277 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 280 |
| 0.30 | std=15/255 (6%) | 100% | 0% | 1.00 | 267 |
| 0.50 | std=25/255 (10%) | 93% | 0% | 0.93 | 282 |
| 0.70 | std=35/255 (14%) | 100% | 0% | 1.00 | 261 |
| 1.00 | std=50/255 (20%) | 0% | 0% | 0.00 | 520 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 271 |
| 0.10 | 1 patches, up to 3% each | 100% | 0% | 1.00 | 274 |
| 0.20 | 1 patches, up to 6% each | 100% | 0% | 1.00 | 273 |
| 0.30 | 1 patches, up to 9% each | 100% | 0% | 1.00 | 276 |
| 0.50 | 2 patches, up to 15% each | 100% | 0% | 1.00 | 273 |
| 0.70 | 3 patches, up to 21% each | 93% | 0% | 0.93 | 292 |
| 1.00 | 5 patches, up to 30% each | 100% | 0% | 1.00 | 293 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 273 |
| 0.10 | +/-8/255 (3%) | 100% | 0% | 1.00 | 276 |
| 0.20 | +/-16/255 (6%) | 93% | 0% | 0.93 | 293 |
| 0.30 | +/-24/255 (9%) | 100% | 0% | 1.00 | 274 |
| 0.50 | +/-40/255 (16%) | 100% | 0% | 1.00 | 275 |
| 0.70 | +/-56/255 (22%) | 100% | 0% | 1.00 | 271 |
| 1.00 | +/-80/255 (31%) | 93% | 0% | 0.93 | 290 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 274 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 269 |
| 0.20 | 112px effective (2x downscale) | 93% | 0% | 0.93 | 304 |
| 0.30 | 74px effective (3x downscale) | 80% | 0% | 0.80 | 338 |
| 0.50 | 56px effective (4x downscale) | 47% | 0% | 0.47 | 426 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 520 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 520 |

### Task 4: *put the white mug on the left plate and put the yellow and white mug on the right plate*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.50 (100ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.50 (50% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 7%, robust
- **EmbodimentStressor**: 100% baseline, max deg 33%, robust
- **LongHorizonDriftStressor**: 100% baseline, max deg 100%, bp=0.30 (obs std 0.3 @step100)
- **ImageNoiseStressor**: 100% baseline, max deg 33%, robust
- **OcclusionStressor**: 100% baseline, max deg 13%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 0%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.50 (56px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 227 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 248 |
| 0.20 | 40ms (2 steps) | 100% | 0% | 1.00 | 270 |
| 0.30 | 60ms (3 steps) | 80% | 0% | 0.80 | 345 |
| 0.50 | 100ms (5 steps) | 13% | 0% | 0.13 | 499 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 520 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 520 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 230 |
| 0.10 | 10% drop prob | 100% | 0% | 1.00 | 254 |
| 0.20 | 20% drop prob | 87% | 0% | 0.87 | 327 |
| 0.30 | 30% drop prob | 73% | 0% | 0.73 | 375 |
| 0.50 | 50% drop prob | 7% | 0% | 0.07 | 507 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 520 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 520 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 234 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 100% | 0% | 1.00 | 239 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 100% | 0% | 1.00 | 231 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 100% | 0% | 1.00 | 232 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 100% | 0% | 1.00 | 230 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 100% | 0% | 1.00 | 232 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 93% | 0% | 0.93 | 250 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 235 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 100% | 0% | 1.00 | 246 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 93% | 0% | 0.93 | 264 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 93% | 0% | 0.93 | 267 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 93% | 0% | 0.93 | 259 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 73% | 0% | 0.73 | 335 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 67% | 0% | 0.67 | 360 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 229 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 93% | 0% | 0.93 | 247 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 80% | 0% | 0.80 | 319 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 33% | 0% | 0.33 | 429 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 7% | 0% | 0.07 | 502 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 0% | 0% | 0.00 | 520 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 0% | 0% | 0.00 | 520 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 232 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 236 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 235 |
| 0.30 | std=15/255 (6%) | 100% | 0% | 1.00 | 235 |
| 0.50 | std=25/255 (10%) | 100% | 0% | 1.00 | 233 |
| 0.70 | std=35/255 (14%) | 100% | 0% | 1.00 | 239 |
| 1.00 | std=50/255 (20%) | 67% | 0% | 0.67 | 330 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 228 |
| 0.10 | 1 patches, up to 3% each | 100% | 0% | 1.00 | 233 |
| 0.20 | 1 patches, up to 6% each | 100% | 0% | 1.00 | 231 |
| 0.30 | 1 patches, up to 9% each | 100% | 0% | 1.00 | 229 |
| 0.50 | 2 patches, up to 15% each | 100% | 0% | 1.00 | 233 |
| 0.70 | 3 patches, up to 21% each | 100% | 0% | 1.00 | 241 |
| 1.00 | 5 patches, up to 30% each | 87% | 0% | 0.87 | 284 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 229 |
| 0.10 | +/-8/255 (3%) | 100% | 0% | 1.00 | 227 |
| 0.20 | +/-16/255 (6%) | 100% | 0% | 1.00 | 236 |
| 0.30 | +/-24/255 (9%) | 100% | 0% | 1.00 | 227 |
| 0.50 | +/-40/255 (16%) | 100% | 0% | 1.00 | 233 |
| 0.70 | +/-56/255 (22%) | 100% | 0% | 1.00 | 227 |
| 1.00 | +/-80/255 (31%) | 100% | 0% | 1.00 | 231 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 226 |
| 0.10 | 224px (native) | 93% | 0% | 0.93 | 249 |
| 0.20 | 112px effective (2x downscale) | 93% | 0% | 0.93 | 270 |
| 0.30 | 74px effective (3x downscale) | 87% | 0% | 0.87 | 284 |
| 0.50 | 56px effective (4x downscale) | 47% | 0% | 0.47 | 406 |
| 0.70 | 44px effective (5x downscale) | 7% | 0% | 0.07 | 513 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 520 |

### Task 9: *put the yellow and white mug in the microwave and close it*

- **LatencyStressor**: 93% baseline, max deg 80%, bp=0.50 (100ms)
- **DropoutStressor**: 87% baseline, max deg 87%, bp=0.20 (20% dropout)
- **PhysicsShiftStressor**: 93% baseline, max deg 13%, robust
- **EmbodimentStressor**: 93% baseline, max deg 53%, bp=0.70 (links 0.93-1.07x, gains 0.79-1.21x)
- **LongHorizonDriftStressor**: 93% baseline, max deg 93%, bp=0.30 (obs std 0.3 @step100)
- **ImageNoiseStressor**: 80% baseline, max deg 13%, robust
- **OcclusionStressor**: 100% baseline, max deg 33%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 20%, robust
- **ResolutionStressor**: 93% baseline, max deg 93%, bp=0.50 (56px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 93% | 0% | 0.93 | 277 |
| 0.10 | 20ms (1 steps) | 87% | 0% | 0.87 | 337 |
| 0.20 | 40ms (2 steps) | 67% | 0% | 0.67 | 387 |
| 0.30 | 60ms (3 steps) | 73% | 0% | 0.73 | 411 |
| 0.50 | 100ms (5 steps) | 47% | 0% | 0.47 | 474 |
| 0.70 | 140ms (7 steps) | 60% | 0% | 0.60 | 466 |
| 1.00 | 200ms (10 steps) | 13% | 0% | 0.13 | 512 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 87% | 0% | 0.87 | 302 |
| 0.10 | 10% drop prob | 60% | 0% | 0.60 | 398 |
| 0.20 | 20% drop prob | 40% | 0% | 0.40 | 442 |
| 0.30 | 30% drop prob | 0% | 0% | 0.00 | 520 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 520 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 520 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 520 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 93% | 0% | 0.93 | 293 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 80% | 0% | 0.80 | 307 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 87% | 0% | 0.87 | 298 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 100% | 0% | 1.00 | 278 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 93% | 0% | 0.93 | 277 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 93% | 0% | 0.93 | 280 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 80% | 0% | 0.80 | 329 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 93% | 0% | 0.93 | 290 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 93% | 0% | 0.93 | 286 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 93% | 0% | 0.93 | 300 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 80% | 0% | 0.80 | 343 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 60% | 0% | 0.60 | 378 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 40% | 0% | 0.40 | 451 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 47% | 0% | 0.47 | 423 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 93% | 0% | 0.93 | 294 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 93% | 0% | 0.93 | 297 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 80% | 0% | 0.80 | 333 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 33% | 0% | 0.33 | 442 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 7% | 0% | 0.07 | 512 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 0% | 0% | 0.00 | 520 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 0% | 0% | 0.00 | 520 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 80% | 0% | 0.80 | 332 |
| 0.10 | std=5/255 (2%) | 93% | 0% | 0.93 | 291 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 285 |
| 0.30 | std=15/255 (6%) | 93% | 0% | 0.93 | 281 |
| 0.50 | std=25/255 (10%) | 93% | 0% | 0.93 | 287 |
| 0.70 | std=35/255 (14%) | 87% | 0% | 0.87 | 294 |
| 1.00 | std=50/255 (20%) | 67% | 0% | 0.67 | 350 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 278 |
| 0.10 | 1 patches, up to 3% each | 93% | 0% | 0.93 | 282 |
| 0.20 | 1 patches, up to 6% each | 80% | 0% | 0.80 | 325 |
| 0.30 | 1 patches, up to 9% each | 93% | 0% | 0.93 | 286 |
| 0.50 | 2 patches, up to 15% each | 80% | 0% | 0.80 | 346 |
| 0.70 | 3 patches, up to 21% each | 67% | 0% | 0.67 | 376 |
| 1.00 | 5 patches, up to 30% each | 67% | 0% | 0.67 | 387 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 277 |
| 0.10 | +/-8/255 (3%) | 80% | 0% | 0.80 | 316 |
| 0.20 | +/-16/255 (6%) | 93% | 0% | 0.93 | 292 |
| 0.30 | +/-24/255 (9%) | 93% | 0% | 0.93 | 294 |
| 0.50 | +/-40/255 (16%) | 93% | 0% | 0.93 | 278 |
| 0.70 | +/-56/255 (22%) | 100% | 0% | 1.00 | 270 |
| 1.00 | +/-80/255 (31%) | 87% | 0% | 0.87 | 305 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 93% | 0% | 0.93 | 278 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 284 |
| 0.20 | 112px effective (2x downscale) | 93% | 0% | 0.93 | 318 |
| 0.30 | 74px effective (3x downscale) | 53% | 0% | 0.53 | 420 |
| 0.50 | 56px effective (4x downscale) | 0% | 0% | 0.00 | 520 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 520 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 520 |


## Breakpoint Comparison

Average intensity at which success rate drops below 50%:

- **LatencyStressor**: libero_spatial: 0.50, libero_object: 0.57, libero_goal: 0.85, libero_10: 0.50
- **DropoutStressor**: libero_spatial: 0.40, libero_object: 0.50, libero_goal: 0.33, libero_10: 0.40
- **PhysicsShiftStressor**: libero_spatial: robust, libero_object: robust, libero_goal: robust, libero_10: robust
- **EmbodimentStressor**: libero_spatial: robust, libero_object: robust, libero_goal: 0.50, libero_10: 0.70
- **LongHorizonDriftStressor**: libero_spatial: 1.00, libero_object: 1.00, libero_goal: 0.85, libero_10: 0.30
- **ImageNoiseStressor**: libero_spatial: 1.00, libero_object: robust, libero_goal: 1.00, libero_10: 1.00
- **OcclusionStressor**: libero_spatial: robust, libero_object: robust, libero_goal: robust, libero_10: robust
- **BrightnessShiftStressor**: libero_spatial: robust, libero_object: robust, libero_goal: robust, libero_10: robust
- **ResolutionStressor**: libero_spatial: 0.73, libero_object: 0.80, libero_goal: 0.60, libero_10: 0.50

---

*Report generated by Trace Robotics v0.1.0*
*https://tracerobotics.com*