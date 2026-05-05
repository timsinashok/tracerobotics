# Trace Robotics — Multi-Task Robustness Report

**Policy:** Pi0PolicyAdapter  
**Suites:** libero_object  
**Task IDs:** [0, 2, 4, 7, 9]  
**Total tasks evaluated:** 5  
**Modalities:** vision, proprioception  
**Generated:** 2026-05-04 22:39  
**Control frequency:** 50Hz (20ms per step)

---

## Cross-Suite Summary

| Stressor | libero_object |
|---|---|
| LatencyStressor | 97% (bp=0.62) |
| DropoutStressor | 100% (bp=0.46) |
| PhysicsShiftStressor | 100% (bp=0.10) |
| EmbodimentStressor | 99% (bp=0.10) |
| LongHorizonDriftStressor | 100% (robust) |
| ImageNoiseStressor | 100% (bp=1.00) |
| OcclusionStressor | 99% (robust) |
| BrightnessShiftStressor | 97% (robust) |
| ResolutionStressor | 99% (bp=0.64) |

## libero_object

### Task 0: *pick up the alphabet soup and place it in the basket*

- **LatencyStressor**: 87% baseline, max deg 87%, bp=0.70 (140ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.50 (50% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 100%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 100% baseline, max deg 100%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 100% baseline, max deg 40%, robust
- **ImageNoiseStressor**: 100% baseline, max deg 53%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 93% baseline, max deg 7%, robust
- **BrightnessShiftStressor**: 93% baseline, max deg 0%, robust
- **ResolutionStressor**: 93% baseline, max deg 93%, bp=0.70 (44px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 87% | 0% | 0.87 | 163 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 144 |
| 0.20 | 40ms (2 steps) | 100% | 0% | 1.00 | 153 |
| 0.30 | 60ms (3 steps) | 87% | 0% | 0.87 | 189 |
| 0.50 | 100ms (5 steps) | 80% | 0% | 0.80 | 234 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 280 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 280 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 142 |
| 0.10 | 10% drop prob | 93% | 0% | 0.93 | 161 |
| 0.20 | 20% drop prob | 93% | 0% | 0.93 | 199 |
| 0.30 | 30% drop prob | 53% | 0% | 0.53 | 239 |
| 0.50 | 50% drop prob | 7% | 0% | 0.07 | 274 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 280 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 280 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 144 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 280 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 280 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 280 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 280 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 280 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 280 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 155 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 280 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 280 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 280 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 280 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 280 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 280 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 154 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 152 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 93% | 0% | 0.93 | 154 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 100% | 0% | 1.00 | 164 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 87% | 0% | 0.87 | 169 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 80% | 0% | 0.80 | 186 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 60% | 0% | 0.60 | 226 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 149 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 166 |
| 0.20 | std=10/255 (4%) | 93% | 0% | 0.93 | 170 |
| 0.30 | std=15/255 (6%) | 93% | 0% | 0.93 | 177 |
| 0.50 | std=25/255 (10%) | 93% | 0% | 0.93 | 189 |
| 0.70 | std=35/255 (14%) | 80% | 0% | 0.80 | 196 |
| 1.00 | std=50/255 (20%) | 47% | 0% | 0.47 | 262 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 93% | 0% | 0.93 | 156 |
| 0.10 | 1 patches, up to 3% each | 93% | 0% | 0.93 | 169 |
| 0.20 | 1 patches, up to 6% each | 100% | 0% | 1.00 | 169 |
| 0.30 | 1 patches, up to 9% each | 87% | 0% | 0.87 | 191 |
| 0.50 | 2 patches, up to 15% each | 87% | 0% | 0.87 | 184 |
| 0.70 | 3 patches, up to 21% each | 100% | 0% | 1.00 | 158 |
| 1.00 | 5 patches, up to 30% each | 100% | 0% | 1.00 | 184 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 93% | 0% | 0.93 | 154 |
| 0.10 | +/-8/255 (3%) | 100% | 0% | 1.00 | 168 |
| 0.20 | +/-16/255 (6%) | 93% | 0% | 0.93 | 177 |
| 0.30 | +/-24/255 (9%) | 100% | 0% | 1.00 | 170 |
| 0.50 | +/-40/255 (16%) | 100% | 0% | 1.00 | 171 |
| 0.70 | +/-56/255 (22%) | 93% | 0% | 0.93 | 173 |
| 1.00 | +/-80/255 (31%) | 100% | 0% | 1.00 | 174 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 93% | 0% | 0.93 | 150 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 146 |
| 0.20 | 112px effective (2x downscale) | 100% | 0% | 1.00 | 197 |
| 0.30 | 74px effective (3x downscale) | 53% | 0% | 0.53 | 246 |
| 0.50 | 56px effective (4x downscale) | 73% | 0% | 0.73 | 257 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 280 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 280 |

### Task 2: *pick up the salad dressing and place it in the basket*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.70 (140ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.30 (30% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 100%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 100% baseline, max deg 100%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 100% baseline, max deg 20%, robust
- **ImageNoiseStressor**: 100% baseline, max deg 53%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 100% baseline, max deg 40%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 13%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=1.00 (28px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 109 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 121 |
| 0.20 | 40ms (2 steps) | 93% | 0% | 0.93 | 136 |
| 0.30 | 60ms (3 steps) | 93% | 0% | 0.93 | 167 |
| 0.50 | 100ms (5 steps) | 67% | 0% | 0.67 | 237 |
| 0.70 | 140ms (7 steps) | 13% | 0% | 0.13 | 279 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 280 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 112 |
| 0.10 | 10% drop prob | 73% | 0% | 0.73 | 167 |
| 0.20 | 20% drop prob | 67% | 0% | 0.67 | 202 |
| 0.30 | 30% drop prob | 20% | 0% | 0.20 | 248 |
| 0.50 | 50% drop prob | 7% | 0% | 0.07 | 272 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 280 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 280 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 117 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 280 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 280 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 280 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 280 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 280 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 280 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 111 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 280 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 280 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 280 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 280 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 280 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 280 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 112 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 114 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 120 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 100% | 0% | 1.00 | 112 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 100% | 0% | 1.00 | 118 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 100% | 0% | 1.00 | 124 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 80% | 0% | 0.80 | 158 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 113 |
| 0.10 | std=5/255 (2%) | 87% | 0% | 0.87 | 185 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 152 |
| 0.30 | std=15/255 (6%) | 87% | 0% | 0.87 | 180 |
| 0.50 | std=25/255 (10%) | 87% | 0% | 0.87 | 159 |
| 0.70 | std=35/255 (14%) | 100% | 0% | 1.00 | 149 |
| 1.00 | std=50/255 (20%) | 47% | 0% | 0.47 | 239 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 114 |
| 0.10 | 1 patches, up to 3% each | 80% | 0% | 0.80 | 188 |
| 0.20 | 1 patches, up to 6% each | 93% | 0% | 0.93 | 156 |
| 0.30 | 1 patches, up to 9% each | 93% | 0% | 0.93 | 170 |
| 0.50 | 2 patches, up to 15% each | 80% | 0% | 0.80 | 186 |
| 0.70 | 3 patches, up to 21% each | 87% | 0% | 0.87 | 186 |
| 1.00 | 5 patches, up to 30% each | 60% | 0% | 0.60 | 198 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 109 |
| 0.10 | +/-8/255 (3%) | 87% | 0% | 0.87 | 170 |
| 0.20 | +/-16/255 (6%) | 93% | 0% | 0.93 | 170 |
| 0.30 | +/-24/255 (9%) | 87% | 0% | 0.87 | 187 |
| 0.50 | +/-40/255 (16%) | 100% | 0% | 1.00 | 161 |
| 0.70 | +/-56/255 (22%) | 93% | 0% | 0.93 | 172 |
| 1.00 | +/-80/255 (31%) | 100% | 0% | 1.00 | 159 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 112 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 111 |
| 0.20 | 112px effective (2x downscale) | 93% | 0% | 0.93 | 143 |
| 0.30 | 74px effective (3x downscale) | 80% | 0% | 0.80 | 184 |
| 0.50 | 56px effective (4x downscale) | 80% | 0% | 0.80 | 191 |
| 0.70 | 44px effective (5x downscale) | 53% | 0% | 0.53 | 237 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 280 |

### Task 4: *pick up the ketchup and place it in the basket*

- **LatencyStressor**: 100% baseline, max deg 93%, bp=0.70 (140ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.50 (50% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 27%, robust
- **EmbodimentStressor**: 100% baseline, max deg 13%, robust
- **LongHorizonDriftStressor**: 100% baseline, max deg 40%, robust
- **ImageNoiseStressor**: 100% baseline, max deg 80%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 100% baseline, max deg 13%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 7%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.30 (74px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 144 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 149 |
| 0.20 | 40ms (2 steps) | 100% | 0% | 1.00 | 155 |
| 0.30 | 60ms (3 steps) | 93% | 0% | 0.93 | 170 |
| 0.50 | 100ms (5 steps) | 53% | 0% | 0.53 | 253 |
| 0.70 | 140ms (7 steps) | 13% | 0% | 0.13 | 276 |
| 1.00 | 200ms (10 steps) | 7% | 0% | 0.07 | 278 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 153 |
| 0.10 | 10% drop prob | 93% | 0% | 0.93 | 167 |
| 0.20 | 20% drop prob | 87% | 0% | 0.87 | 199 |
| 0.30 | 30% drop prob | 53% | 0% | 0.53 | 238 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 280 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 280 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 280 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 150 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 100% | 0% | 1.00 | 1 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 100% | 0% | 1.00 | 1 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 93% | 0% | 0.93 | 20 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 73% | 0% | 0.73 | 76 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 93% | 0% | 0.93 | 21 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 93% | 0% | 0.93 | 22 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 147 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 100% | 0% | 1.00 | 1 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 100% | 0% | 1.00 | 1 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 100% | 0% | 1.00 | 2 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 100% | 0% | 1.00 | 1 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 100% | 0% | 1.00 | 3 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 87% | 0% | 0.87 | 40 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 154 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 154 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 93% | 0% | 0.93 | 155 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 100% | 0% | 1.00 | 154 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 80% | 0% | 0.80 | 181 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 60% | 0% | 0.60 | 205 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 60% | 0% | 0.60 | 209 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 147 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 167 |
| 0.20 | std=10/255 (4%) | 93% | 0% | 0.93 | 172 |
| 0.30 | std=15/255 (6%) | 100% | 0% | 1.00 | 170 |
| 0.50 | std=25/255 (10%) | 73% | 0% | 0.73 | 193 |
| 0.70 | std=35/255 (14%) | 80% | 0% | 0.80 | 200 |
| 1.00 | std=50/255 (20%) | 20% | 0% | 0.20 | 260 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 147 |
| 0.10 | 1 patches, up to 3% each | 87% | 0% | 0.87 | 193 |
| 0.20 | 1 patches, up to 6% each | 100% | 0% | 1.00 | 166 |
| 0.30 | 1 patches, up to 9% each | 100% | 0% | 1.00 | 169 |
| 0.50 | 2 patches, up to 15% each | 100% | 0% | 1.00 | 162 |
| 0.70 | 3 patches, up to 21% each | 100% | 0% | 1.00 | 160 |
| 1.00 | 5 patches, up to 30% each | 100% | 0% | 1.00 | 162 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 144 |
| 0.10 | +/-8/255 (3%) | 93% | 0% | 0.93 | 179 |
| 0.20 | +/-16/255 (6%) | 93% | 0% | 0.93 | 170 |
| 0.30 | +/-24/255 (9%) | 100% | 0% | 1.00 | 159 |
| 0.50 | +/-40/255 (16%) | 93% | 0% | 0.93 | 173 |
| 0.70 | +/-56/255 (22%) | 100% | 0% | 1.00 | 168 |
| 1.00 | +/-80/255 (31%) | 100% | 0% | 1.00 | 167 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 146 |
| 0.10 | 224px (native) | 93% | 0% | 0.93 | 155 |
| 0.20 | 112px effective (2x downscale) | 100% | 0% | 1.00 | 171 |
| 0.30 | 74px effective (3x downscale) | 33% | 0% | 0.33 | 247 |
| 0.50 | 56px effective (4x downscale) | 0% | 0% | 0.00 | 280 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 280 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 280 |

### Task 7: *pick up the milk and place it in the basket*

- **LatencyStressor**: 100% baseline, max deg 93%, bp=0.50 (100ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.50 (50% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 100%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 93% baseline, max deg 93%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 100% baseline, max deg 33%, robust
- **ImageNoiseStressor**: 100% baseline, max deg 93%, bp=1.00 (noise std=50)
- **OcclusionStressor**: 100% baseline, max deg 40%, robust
- **BrightnessShiftStressor**: 93% baseline, max deg 13%, robust
- **ResolutionStressor**: 100% baseline, max deg 100%, bp=0.70 (44px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 131 |
| 0.10 | 20ms (1 steps) | 93% | 0% | 0.93 | 143 |
| 0.20 | 40ms (2 steps) | 100% | 0% | 1.00 | 150 |
| 0.30 | 60ms (3 steps) | 67% | 0% | 0.67 | 218 |
| 0.50 | 100ms (5 steps) | 47% | 0% | 0.47 | 260 |
| 0.70 | 140ms (7 steps) | 27% | 0% | 0.27 | 273 |
| 1.00 | 200ms (10 steps) | 7% | 0% | 0.07 | 276 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 127 |
| 0.10 | 10% drop prob | 93% | 0% | 0.93 | 149 |
| 0.20 | 20% drop prob | 93% | 0% | 0.93 | 179 |
| 0.30 | 30% drop prob | 60% | 0% | 0.60 | 215 |
| 0.50 | 50% drop prob | 0% | 0% | 0.00 | 280 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 280 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 280 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 130 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 280 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 280 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 280 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 280 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 280 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 280 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 93% | 0% | 0.93 | 143 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 0% | 0% | 0.00 | 280 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 0% | 0% | 0.00 | 280 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 0% | 0% | 0.00 | 280 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 0% | 0% | 0.00 | 280 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 0% | 0% | 0.00 | 280 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 0% | 0% | 0.00 | 280 |

#### LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 129 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 135 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 93% | 0% | 0.93 | 141 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 93% | 0% | 0.93 | 141 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 100% | 0% | 1.00 | 139 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 87% | 0% | 0.87 | 170 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 67% | 0% | 0.67 | 197 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 134 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 174 |
| 0.20 | std=10/255 (4%) | 80% | 0% | 0.80 | 175 |
| 0.30 | std=15/255 (6%) | 87% | 0% | 0.87 | 179 |
| 0.50 | std=25/255 (10%) | 100% | 0% | 1.00 | 173 |
| 0.70 | std=35/255 (14%) | 73% | 0% | 0.73 | 201 |
| 1.00 | std=50/255 (20%) | 7% | 0% | 0.07 | 278 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 132 |
| 0.10 | 1 patches, up to 3% each | 73% | 0% | 0.73 | 202 |
| 0.20 | 1 patches, up to 6% each | 87% | 0% | 0.87 | 182 |
| 0.30 | 1 patches, up to 9% each | 93% | 0% | 0.93 | 192 |
| 0.50 | 2 patches, up to 15% each | 93% | 0% | 0.93 | 181 |
| 0.70 | 3 patches, up to 21% each | 80% | 0% | 0.80 | 197 |
| 1.00 | 5 patches, up to 30% each | 60% | 0% | 0.60 | 220 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 93% | 0% | 0.93 | 139 |
| 0.10 | +/-8/255 (3%) | 93% | 0% | 0.93 | 181 |
| 0.20 | +/-16/255 (6%) | 100% | 0% | 1.00 | 165 |
| 0.30 | +/-24/255 (9%) | 93% | 0% | 0.93 | 167 |
| 0.50 | +/-40/255 (16%) | 80% | 0% | 0.80 | 186 |
| 0.70 | +/-56/255 (22%) | 93% | 0% | 0.93 | 169 |
| 1.00 | +/-80/255 (31%) | 100% | 0% | 1.00 | 163 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 137 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 133 |
| 0.20 | 112px effective (2x downscale) | 80% | 0% | 0.80 | 199 |
| 0.30 | 74px effective (3x downscale) | 80% | 0% | 0.80 | 190 |
| 0.50 | 56px effective (4x downscale) | 53% | 0% | 0.53 | 259 |
| 0.70 | 44px effective (5x downscale) | 0% | 0% | 0.00 | 280 |
| 1.00 | 28px effective (8x downscale) | 0% | 0% | 0.00 | 280 |

### Task 9: *pick up the orange juice and place it in the basket*

- **LatencyStressor**: 100% baseline, max deg 100%, bp=0.50 (100ms)
- **DropoutStressor**: 100% baseline, max deg 100%, bp=0.50 (50% dropout)
- **PhysicsShiftStressor**: 100% baseline, max deg 100%, bp=0.10 (mass 0.9-1.1x, fric 0.9-1.1x)
- **EmbodimentStressor**: 100% baseline, max deg 100%, bp=0.10 (links 0.99-1.01x, gains 0.97-1.03x)
- **LongHorizonDriftStressor**: 100% baseline, max deg 27%, robust
- **ImageNoiseStressor**: 100% baseline, max deg 20%, robust
- **OcclusionStressor**: 100% baseline, max deg 20%, robust
- **BrightnessShiftStressor**: 100% baseline, max deg 20%, robust
- **ResolutionStressor**: 100% baseline, max deg 87%, bp=0.50 (56px effective)

#### LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 122 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 132 |
| 0.20 | 40ms (2 steps) | 100% | 0% | 1.00 | 146 |
| 0.30 | 60ms (3 steps) | 100% | 0% | 1.00 | 157 |
| 0.50 | 100ms (5 steps) | 33% | 0% | 0.33 | 258 |
| 0.70 | 140ms (7 steps) | 7% | 0% | 0.07 | 280 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 280 |

#### DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 129 |
| 0.10 | 10% drop prob | 93% | 0% | 0.93 | 136 |
| 0.20 | 20% drop prob | 73% | 0% | 0.73 | 181 |
| 0.30 | 30% drop prob | 73% | 0% | 0.73 | 196 |
| 0.50 | 50% drop prob | 47% | 0% | 0.47 | 257 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 280 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 280 |

#### PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 122 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 0% | 0% | 0.00 | 280 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 0% | 0% | 0.00 | 280 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 0% | 0% | 0.00 | 280 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 0% | 0% | 0.00 | 280 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 0% | 0% | 0.00 | 280 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 0% | 0% | 0.00 | 280 |

#### EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 122 |
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
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 124 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 122 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 100% | 0% | 1.00 | 125 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 100% | 0% | 1.00 | 130 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 80% | 0% | 0.80 | 162 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 73% | 0% | 0.73 | 177 |

#### ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 120 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 143 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 149 |
| 0.30 | std=15/255 (6%) | 100% | 0% | 1.00 | 148 |
| 0.50 | std=25/255 (10%) | 100% | 0% | 1.00 | 148 |
| 0.70 | std=35/255 (14%) | 93% | 0% | 0.93 | 175 |
| 1.00 | std=50/255 (20%) | 80% | 0% | 0.80 | 211 |

#### OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 123 |
| 0.10 | 1 patches, up to 3% each | 93% | 0% | 0.93 | 166 |
| 0.20 | 1 patches, up to 6% each | 80% | 0% | 0.80 | 180 |
| 0.30 | 1 patches, up to 9% each | 80% | 0% | 0.80 | 184 |
| 0.50 | 2 patches, up to 15% each | 87% | 0% | 0.87 | 166 |
| 0.70 | 3 patches, up to 21% each | 93% | 0% | 0.93 | 167 |
| 1.00 | 5 patches, up to 30% each | 93% | 0% | 0.93 | 165 |

#### BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 124 |
| 0.10 | +/-8/255 (3%) | 93% | 0% | 0.93 | 162 |
| 0.20 | +/-16/255 (6%) | 100% | 0% | 1.00 | 151 |
| 0.30 | +/-24/255 (9%) | 93% | 0% | 0.93 | 162 |
| 0.50 | +/-40/255 (16%) | 87% | 0% | 0.87 | 170 |
| 0.70 | +/-56/255 (22%) | 100% | 0% | 1.00 | 168 |
| 1.00 | +/-80/255 (31%) | 80% | 0% | 0.80 | 181 |

#### ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 128 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 124 |
| 0.20 | 112px effective (2x downscale) | 93% | 0% | 0.93 | 146 |
| 0.30 | 74px effective (3x downscale) | 73% | 0% | 0.73 | 183 |
| 0.50 | 56px effective (4x downscale) | 33% | 0% | 0.33 | 252 |
| 0.70 | 44px effective (5x downscale) | 20% | 0% | 0.20 | 256 |
| 1.00 | 28px effective (8x downscale) | 13% | 0% | 0.13 | 270 |


## Breakpoint Comparison

Average intensity at which success rate drops below 50%:

- **LatencyStressor**: libero_object: 0.62
- **DropoutStressor**: libero_object: 0.46
- **PhysicsShiftStressor**: libero_object: 0.10
- **EmbodimentStressor**: libero_object: 0.10
- **LongHorizonDriftStressor**: libero_object: robust
- **ImageNoiseStressor**: libero_object: 1.00
- **OcclusionStressor**: libero_object: robust
- **BrightnessShiftStressor**: libero_object: robust
- **ResolutionStressor**: libero_object: 0.64

---

*Report generated by Trace Robotics v0.1.0*
*https://tracerobotics.com*