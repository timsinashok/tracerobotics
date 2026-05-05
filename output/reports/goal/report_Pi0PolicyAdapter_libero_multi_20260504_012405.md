# Trace Robotics — Multi-Task Robustness Report

**Policy:** Pi0PolicyAdapter  
**Suites:** libero_goal  
**Task IDs:** [0, 2, 4, 7, 9]  
**Total tasks evaluated:** 5  
**Modalities:** vision, proprioception  
**Generated:** 2026-05-04 18:06  
**Control frequency:** 50Hz (20ms per step)

---

## Cross-Suite Summary

| Stressor | libero_goal |
|---|---|
| LatencyStressor | 99% (bp=0.83) |
| DropoutStressor | 100% (bp=0.50) |
| PhysicsShiftStressor | 95% (bp=0.10) |
| EmbodimentStressor | 97% (bp=0.10) |
| LongHorizonDriftStressor | 97% (bp=0.70) |
| ImageNoiseStressor | 100% (bp=0.86) |
| OcclusionStressor | 100% (robust) |
| BrightnessShiftStressor | 97% (robust) |
| ResolutionStressor | 97% (bp=0.45) |

## libero_goal

### Task 0: *open the middle drawer of the cabinet*

- **LatencyStressor**: 100% baseline, max deg 7%, robust
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.30 (30% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 100%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 100% baseline, max deg 100%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 100% baseline, max deg 100%, bp=0.70 (obs std 0.7 @step100)
- **ImageNoiseStressor**: 100% baseline, max deg 87%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 100% baseline, max deg 7%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 13%, robust
- **ResolutionStressor**: 93% baseline, max deg 93%, bp=0.50 (56px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 119 |
| 0.10 | 20ms (1 steps) | 93% | 0% | 0.93 | 135 |
| 0.20 | 40ms (2 steps) | 100% | 0% | 1.00 | 129 |
| 0.30 | 60ms (3 steps) | 93% | 0% | 0.93 | 143 |
| 0.50 | 100ms (5 steps) | 93% | 0% | 0.93 | 158 |
| 0.70 | 140ms (7 steps) | 100% | 0% | 1.00 | 154 |
| 1.00 | 200ms (10 steps) | 100% | 0% | 1.00 | 162 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 118 |
| 0.10 | 10% drop prob | 87% | 0% | 0.87 | 154 |
| 0.20 | 20% drop prob | 67% | 0% | 0.67 | 184 |
| 0.30 | 30% drop prob | 47% | 0% | 0.47 | 226 |
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
| 0.00 | nominal | 100% | 0% | 1.00 | 117 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 300 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 300 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 300 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 300 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 300 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 300 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 117 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 120 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 115 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 93% | 0% | 0.93 | 141 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 80% | 0% | 0.80 | 160 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 27% | 0% | 0.27 | 263 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 0% | 0% | 0.00 | 300 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 120 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 127 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 126 |
| 0.30 | std=15/255 (6%) | 100% | 0% | 1.00 | 143 |
| 0.50 | std=25/255 (10%) | 100% | 0% | 1.00 | 143 |
| 0.70 | std=35/255 (14%) | 67% | 0% | 0.67 | 187 |
| 1.00 | std=50/255 (20%) | 13% | 0% | 0.13 | 284 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 119 |
| 0.10 | 1 patches, up to 3% each | 93% | 0% | 0.93 | 140 |
| 0.20 | 1 patches, up to 6% each | 93% | 0% | 0.93 | 139 |
| 0.30 | 1 patches, up to 9% each | 93% | 0% | 0.93 | 145 |
| 0.50 | 2 patches, up to 15% each | 100% | 0% | 1.00 | 137 |
| 0.70 | 3 patches, up to 21% each | 100% | 0% | 1.00 | 140 |
| 1.00 | 5 patches, up to 30% each | 93% | 0% | 0.93 | 147 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 115 |
| 0.10 | +/-8/255 (3%) | 100% | 0% | 1.00 | 128 |
| 0.20 | +/-16/255 (6%) | 93% | 0% | 0.93 | 137 |
| 0.30 | +/-24/255 (9%) | 87% | 0% | 0.87 | 155 |
| 0.50 | +/-40/255 (16%) | 93% | 0% | 0.93 | 134 |
| 0.70 | +/-56/255 (22%) | 93% | 0% | 0.93 | 149 |
| 1.00 | +/-80/255 (31%) | 100% | 0% | 1.00 | 136 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 93% | 0% | 0.93 | 125 |
| 0.10 | 224px (native) | 93% | 0% | 0.93 | 130 |
| 0.20 | 112px effective (2x downscale) | 100% | 0% | 1.00 | 128 |
| 0.30 | 74px effective (3x downscale) | 60% | 0% | 0.60 | 202 |
| 0.50 | 56px effective (4x downscale) | 0% | 0% | 0.00 | 300 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 300 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 300 |

### Task 2: *put the wine bottle on top of the cabinet*

- **LatencyStressor**: 100% baseline, max deg 67%, bp=1.00 (200ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.50 (50% dropout)
- **PhysicsShiftStressor**: 93% baseline, max deg 93%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 100% baseline, max deg 100%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 87% baseline, max deg 0%, robust
- **ImageNoiseStressor**: 100% baseline, max deg 73%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 100% baseline, max deg 20%, robust
- **BrightnessShiftStressor**: 93% baseline, max deg 13%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.50 (56px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 84 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 93 |
| 0.20 | 40ms (2 steps) | 80% | 0% | 0.80 | 140 |
| 0.30 | 60ms (3 steps) | 87% | 0% | 0.87 | 137 |
| 0.50 | 100ms (5 steps) | 73% | 0% | 0.73 | 200 |
| 0.70 | 140ms (7 steps) | 60% | 0% | 0.60 | 233 |
| 1.00 | 200ms (10 steps) | 33% | 0% | 0.33 | 263 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 84 |
| 0.10 | 10% drop prob | 87% | 0% | 0.87 | 121 |
| 0.20 | 20% drop prob | 67% | 0% | 0.67 | 178 |
| 0.30 | 30% drop prob | 60% | 0% | 0.60 | 181 |
| 0.50 | 50% drop prob | 20% | 0% | 0.20 | 266 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 300 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 300 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 93% | 0% | 0.93 | 99 |
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
| 0.00 | no drift | 87% | 0% | 0.87 | 112 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 85 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 85 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 100% | 0% | 1.00 | 95 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 100% | 0% | 1.00 | 85 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 93% | 0% | 0.93 | 102 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 93% | 0% | 0.93 | 107 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 85 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 108 |
| 0.20 | std=10/255 (4%) | 93% | 0% | 0.93 | 123 |
| 0.30 | std=15/255 (6%) | 87% | 0% | 0.87 | 143 |
| 0.50 | std=25/255 (10%) | 93% | 0% | 0.93 | 124 |
| 0.70 | std=35/255 (14%) | 87% | 0% | 0.87 | 140 |
| 1.00 | std=50/255 (20%) | 27% | 0% | 0.27 | 254 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 85 |
| 0.10 | 1 patches, up to 3% each | 100% | 0% | 1.00 | 107 |
| 0.20 | 1 patches, up to 6% each | 87% | 0% | 0.87 | 144 |
| 0.30 | 1 patches, up to 9% each | 93% | 0% | 0.93 | 126 |
| 0.50 | 2 patches, up to 15% each | 80% | 0% | 0.80 | 150 |
| 0.70 | 3 patches, up to 21% each | 93% | 0% | 0.93 | 122 |
| 1.00 | 5 patches, up to 30% each | 93% | 0% | 0.93 | 128 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 93% | 0% | 0.93 | 99 |
| 0.10 | +/-8/255 (3%) | 87% | 0% | 0.87 | 139 |
| 0.20 | +/-16/255 (6%) | 80% | 0% | 0.80 | 142 |
| 0.30 | +/-24/255 (9%) | 87% | 0% | 0.87 | 131 |
| 0.50 | +/-40/255 (16%) | 87% | 0% | 0.87 | 134 |
| 0.70 | +/-56/255 (22%) | 93% | 0% | 0.93 | 121 |
| 1.00 | +/-80/255 (31%) | 87% | 0% | 0.87 | 135 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 91 |
| 0.10 | 224px (native) | 93% | 0% | 0.93 | 99 |
| 0.20 | 112px effective (2x downscale) | 87% | 0% | 0.87 | 139 |
| 0.30 | 74px effective (3x downscale) | 80% | 0% | 0.80 | 169 |
| 0.50 | 56px effective (4x downscale) | 0% | 0% | 0.00 | 300 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 300 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 300 |

### Task 4: *put the bowl on top of the cabinet*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=1.00 (200ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.50 (50% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 100%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 100% baseline, max deg 100%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 100% baseline, max deg 7%, robust
- **ImageNoiseStressor**: 100% baseline, max deg 67%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 100% baseline, max deg 7%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 0%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.50 (56px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 86 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 92 |
| 0.20 | 40ms (2 steps) | 100% | 0% | 1.00 | 97 |
| 0.30 | 60ms (3 steps) | 93% | 0% | 0.93 | 120 |
| 0.50 | 100ms (5 steps) | 73% | 0% | 0.73 | 211 |
| 0.70 | 140ms (7 steps) | 60% | 0% | 0.60 | 250 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 300 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 85 |
| 0.10 | 10% drop prob | 93% | 0% | 0.93 | 106 |
| 0.20 | 20% drop prob | 93% | 0% | 0.93 | 128 |
| 0.30 | 30% drop prob | 93% | 0% | 0.93 | 137 |
| 0.50 | 50% drop prob | 33% | 0% | 0.33 | 260 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 300 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 300 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 86 |
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
| 0.00 | no drift | 100% | 0% | 1.00 | 87 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 88 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 85 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 100% | 0% | 1.00 | 86 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 100% | 0% | 1.00 | 89 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 100% | 0% | 1.00 | 87 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 93% | 0% | 0.93 | 103 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 86 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 103 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 106 |
| 0.30 | std=15/255 (6%) | 93% | 0% | 0.93 | 118 |
| 0.50 | std=25/255 (10%) | 100% | 0% | 1.00 | 109 |
| 0.70 | std=35/255 (14%) | 100% | 0% | 1.00 | 110 |
| 1.00 | std=50/255 (20%) | 33% | 0% | 0.33 | 245 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 86 |
| 0.10 | 1 patches, up to 3% each | 100% | 0% | 1.00 | 102 |
| 0.20 | 1 patches, up to 6% each | 100% | 0% | 1.00 | 102 |
| 0.30 | 1 patches, up to 9% each | 100% | 0% | 1.00 | 103 |
| 0.50 | 2 patches, up to 15% each | 100% | 0% | 1.00 | 102 |
| 0.70 | 3 patches, up to 21% each | 100% | 0% | 1.00 | 107 |
| 1.00 | 5 patches, up to 30% each | 93% | 0% | 0.93 | 135 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 93 |
| 0.10 | +/-8/255 (3%) | 100% | 0% | 1.00 | 104 |
| 0.20 | +/-16/255 (6%) | 100% | 0% | 1.00 | 101 |
| 0.30 | +/-24/255 (9%) | 100% | 0% | 1.00 | 101 |
| 0.50 | +/-40/255 (16%) | 100% | 0% | 1.00 | 101 |
| 0.70 | +/-56/255 (22%) | 100% | 0% | 1.00 | 103 |
| 1.00 | +/-80/255 (31%) | 100% | 0% | 1.00 | 102 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 87 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 87 |
| 0.20 | 112px effective (2x downscale) | 100% | 0% | 1.00 | 108 |
| 0.30 | 74px effective (3x downscale) | 80% | 0% | 0.80 | 174 |
| 0.50 | 56px effective (4x downscale) | 47% | 0% | 0.47 | 234 |
| 0.70 | 44px effective (5x downscale) | 7% | 0% | 0.07 | 289 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 300 |

### Task 7: *turn on the stove*

- **LatencyStressor**: 100% baseline, max deg 0%, robust
- **DropoutStressor**: 100% baseline, max deg 93%, bp=1.00 (100% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 100%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 100% baseline, max deg 100%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 100% baseline, max deg 27%, robust
- **ImageNoiseStressor**: 100% baseline, max deg 67%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 100% baseline, max deg 0%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 0%, robust
- **ResolutionStressor**: 100% baseline, max deg 27%, robust

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 76 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 78 |
| 0.20 | 40ms (2 steps) | 100% | 0% | 1.00 | 79 |
| 0.30 | 60ms (3 steps) | 100% | 0% | 1.00 | 81 |
| 0.50 | 100ms (5 steps) | 100% | 0% | 1.00 | 93 |
| 0.70 | 140ms (7 steps) | 100% | 0% | 1.00 | 119 |
| 1.00 | 200ms (10 steps) | 100% | 0% | 1.00 | 111 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 76 |
| 0.10 | 10% drop prob | 100% | 0% | 1.00 | 72 |
| 0.20 | 20% drop prob | 100% | 0% | 1.00 | 70 |
| 0.30 | 30% drop prob | 87% | 0% | 0.87 | 110 |
| 0.50 | 50% drop prob | 73% | 0% | 0.73 | 144 |
| 0.70 | 70% drop prob | 73% | 0% | 0.73 | 144 |
| 1.00 | 100% drop prob | 7% | 0% | 0.07 | 294 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 77 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 300 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 300 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 300 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 300 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 300 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 300 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 75 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 300 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 300 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 300 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 300 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 300 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 300 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 74 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 74 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 73 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 100% | 0% | 1.00 | 76 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 93% | 0% | 0.93 | 91 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 100% | 0% | 1.00 | 87 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 73% | 0% | 0.73 | 145 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 77 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 84 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 83 |
| 0.30 | std=15/255 (6%) | 100% | 0% | 1.00 | 83 |
| 0.50 | std=25/255 (10%) | 100% | 0% | 1.00 | 79 |
| 0.70 | std=35/255 (14%) | 100% | 0% | 1.00 | 80 |
| 1.00 | std=50/255 (20%) | 33% | 0% | 0.33 | 231 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 77 |
| 0.10 | 1 patches, up to 3% each | 100% | 0% | 1.00 | 80 |
| 0.20 | 1 patches, up to 6% each | 100% | 0% | 1.00 | 82 |
| 0.30 | 1 patches, up to 9% each | 100% | 0% | 1.00 | 82 |
| 0.50 | 2 patches, up to 15% each | 100% | 0% | 1.00 | 81 |
| 0.70 | 3 patches, up to 21% each | 100% | 0% | 1.00 | 83 |
| 1.00 | 5 patches, up to 30% each | 100% | 0% | 1.00 | 86 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 75 |
| 0.10 | +/-8/255 (3%) | 100% | 0% | 1.00 | 81 |
| 0.20 | +/-16/255 (6%) | 100% | 0% | 1.00 | 79 |
| 0.30 | +/-24/255 (9%) | 100% | 0% | 1.00 | 81 |
| 0.50 | +/-40/255 (16%) | 100% | 0% | 1.00 | 81 |
| 0.70 | +/-56/255 (22%) | 100% | 0% | 1.00 | 81 |
| 1.00 | +/-80/255 (31%) | 100% | 0% | 1.00 | 80 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 76 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 74 |
| 0.20 | 112px effective (2x downscale) | 100% | 0% | 1.00 | 85 |
| 0.30 | 74px effective (3x downscale) | 100% | 0% | 1.00 | 84 |
| 0.50 | 56px effective (4x downscale) | 100% | 0% | 1.00 | 109 |
| 0.70 | 44px effective (5x downscale) | 80% | 0% | 0.80 | 159 |
| 1.00 | 28px effective (8x downscale) | 73% | 0% | 0.73 | 179 |

### Task 9: *put the wine bottle on the rack*

- **LatencyStressor**: 93% baseline, max deg 93%, bp=0.50 (100ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.20 (20% dropout)
- **PhysicsShiftStressor**: 80% baseline, max deg 80%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 87% baseline, max deg 87%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 100% baseline, max deg 60%, bp=0.70 (obs std 0.7 @step100)
- **ImageNoiseStressor**: 100% baseline, max deg 87%, bp=0.30 (noise std=15)
- **OcclusionStressor**: 100% baseline, max deg 27%, robust
- **BrightnessShiftStressor**: 93% baseline, max deg 13%, robust
- **ResolutionStressor**: 93% baseline, max deg 93%, bp=0.30 (74px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 93% | 0% | 0.93 | 150 |
| 0.10 | 20ms (1 steps) | 93% | 0% | 0.93 | 170 |
| 0.20 | 40ms (2 steps) | 100% | 0% | 1.00 | 188 |
| 0.30 | 60ms (3 steps) | 93% | 0% | 0.93 | 206 |
| 0.50 | 100ms (5 steps) | 40% | 0% | 0.40 | 279 |
| 0.70 | 140ms (7 steps) | 13% | 0% | 0.13 | 294 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 300 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 145 |
| 0.10 | 10% drop prob | 53% | 0% | 0.53 | 218 |
| 0.20 | 20% drop prob | 47% | 0% | 0.47 | 230 |
| 0.30 | 30% drop prob | 13% | 0% | 0.13 | 283 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 300 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 300 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 300 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 80% | 0% | 0.80 | 173 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 300 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 300 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 300 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 300 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 300 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 300 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 87% | 0% | 0.87 | 169 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 300 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 300 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 300 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 300 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 300 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 300 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 141 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 144 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 87% | 0% | 0.87 | 168 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 80% | 0% | 0.80 | 187 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 73% | 0% | 0.73 | 189 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 40% | 0% | 0.40 | 235 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 40% | 0% | 0.40 | 250 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 139 |
| 0.10 | std=5/255 (2%) | 80% | 0% | 0.80 | 217 |
| 0.20 | std=10/255 (4%) | 87% | 0% | 0.87 | 221 |
| 0.30 | std=15/255 (6%) | 40% | 0% | 0.40 | 263 |
| 0.50 | std=25/255 (10%) | 33% | 0% | 0.33 | 279 |
| 0.70 | std=35/255 (14%) | 47% | 0% | 0.47 | 238 |
| 1.00 | std=50/255 (20%) | 13% | 0% | 0.13 | 281 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 139 |
| 0.10 | 1 patches, up to 3% each | 80% | 0% | 0.80 | 206 |
| 0.20 | 1 patches, up to 6% each | 73% | 0% | 0.73 | 217 |
| 0.30 | 1 patches, up to 9% each | 80% | 0% | 0.80 | 205 |
| 0.50 | 2 patches, up to 15% each | 87% | 0% | 0.87 | 199 |
| 0.70 | 3 patches, up to 21% each | 73% | 0% | 0.73 | 226 |
| 1.00 | 5 patches, up to 30% each | 87% | 0% | 0.87 | 205 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 93% | 0% | 0.93 | 154 |
| 0.10 | +/-8/255 (3%) | 80% | 0% | 0.80 | 206 |
| 0.20 | +/-16/255 (6%) | 80% | 0% | 0.80 | 211 |
| 0.30 | +/-24/255 (9%) | 100% | 0% | 1.00 | 172 |
| 0.50 | +/-40/255 (16%) | 100% | 0% | 1.00 | 185 |
| 0.70 | +/-56/255 (22%) | 87% | 0% | 0.87 | 199 |
| 1.00 | +/-80/255 (31%) | 80% | 0% | 0.80 | 208 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 93% | 0% | 0.93 | 151 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 138 |
| 0.20 | 112px effective (2x downscale) | 87% | 0% | 0.87 | 208 |
| 0.30 | 74px effective (3x downscale) | 47% | 0% | 0.47 | 255 |
| 0.50 | 56px effective (4x downscale) | 0% | 0% | 0.00 | 300 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 300 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 300 |


## Breakpoint Comparison

Average intensity at which success rate drops below 50%:

- **LatencyStressor**: libero_goal: 0.83
- **DropoutStressor**: libero_goal: 0.50
- **PhysicsShiftStressor**: libero_goal: 0.10
- **EmbodimentStressor**: libero_goal: 0.10
- **LongHorizonDriftStressor**: libero_goal: 0.70
- **ImageNoiseStressor**: libero_goal: 0.86
- **OcclusionStressor**: libero_goal: robust
- **BrightnessShiftStressor**: libero_goal: robust
- **ResolutionStressor**: libero_goal: 0.45

---

*Report generated by Trace Robotics v0.1.0*
*https://tracerobotics.com*