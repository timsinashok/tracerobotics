# Trace Robotics — Robustness Report

**Policy:** Pi0PolicyAdapter  
**Task:** libero  
**Modalities:** vision, proprioception  
**Generated:** 2026-03-05 02:56
**Control frequency:** 50Hz (20ms per step)

---

## Executive Summary

- **LatencyStressor**: baseline 100% success, max degradation 100%, breakpoint at intensity 0.50 (100ms)
- **DropoutStressor**: baseline 98% success, max degradation 98%, breakpoint at intensity 0.70 (70% dropout)
- **PhysicsShiftStressor**: baseline 100% success, max degradation 12%, breakpoint at intensity none (robust)
- **EmbodimentStressor**: baseline 100% success, max degradation 0%, breakpoint at intensity none (robust)
- **LongHorizonDriftStressor**: baseline 100% success, max degradation 18%, breakpoint at intensity none (robust)
- **ImageNoiseStressor**: baseline 100% success, max degradation 2%, breakpoint at intensity none (robust)
- **OcclusionStressor**: baseline 100% success, max degradation 6%, breakpoint at intensity none (robust)
- **BrightnessShiftStressor**: baseline 100% success, max degradation 2%, breakpoint at intensity none (robust)
- **ResolutionStressor**: baseline 100% success, max degradation 96%, breakpoint at intensity 1.00 (28px effective)

## LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 80 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 90 |
| 0.20 | 40ms (2 steps) | 98% | 0% | 0.98 | 107 |
| 0.30 | 60ms (3 steps) | 80% | 0% | 0.80 | 158 |
| 0.50 | 100ms (5 steps) | 14% | 0% | 0.14 | 215 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 220 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 220 |

## DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 98% | 0% | 0.98 | 86 |
| 0.10 | 10% drop prob | 96% | 0% | 0.96 | 95 |
| 0.20 | 20% drop prob | 98% | 0% | 0.98 | 114 |
| 0.30 | 30% drop prob | 90% | 0% | 0.90 | 130 |
| 0.50 | 50% drop prob | 50% | 0% | 0.50 | 183 |
| 0.70 | 70% drop prob | 4% | 0% | 0.04 | 218 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 220 |

## PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 83 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 100% | 0% | 1.00 | 82 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 100% | 0% | 1.00 | 82 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 100% | 0% | 1.00 | 82 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 98% | 0% | 0.98 | 84 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 100% | 0% | 1.00 | 83 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 88% | 0% | 0.88 | 99 |

## EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 81 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 100% | 0% | 1.00 | 86 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 100% | 0% | 1.00 | 84 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 100% | 0% | 1.00 | 86 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 100% | 0% | 1.00 | 86 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 100% | 0% | 1.00 | 89 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 100% | 0% | 1.00 | 93 |

## LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 86 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 100% | 0% | 1.00 | 82 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 100% | 0% | 1.00 | 80 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 100% | 0% | 1.00 | 81 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 100% | 0% | 1.00 | 85 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 88% | 0% | 0.88 | 103 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 82% | 0% | 0.82 | 120 |

## ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 80 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 84 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 81 |
| 0.30 | std=15/255 (6%) | 100% | 0% | 1.00 | 76 |
| 0.50 | std=25/255 (10%) | 100% | 0% | 1.00 | 79 |
| 0.70 | std=35/255 (14%) | 100% | 0% | 1.00 | 80 |
| 1.00 | std=50/255 (20%) | 98% | 0% | 0.98 | 86 |

## OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 82 |
| 0.10 | 1 patches, up to 3% each | 100% | 0% | 1.00 | 85 |
| 0.20 | 1 patches, up to 6% each | 100% | 0% | 1.00 | 84 |
| 0.30 | 1 patches, up to 9% each | 100% | 0% | 1.00 | 83 |
| 0.50 | 2 patches, up to 15% each | 100% | 0% | 1.00 | 84 |
| 0.70 | 3 patches, up to 21% each | 98% | 0% | 0.98 | 87 |
| 1.00 | 5 patches, up to 30% each | 94% | 0% | 0.94 | 98 |

## BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 79 |
| 0.10 | +/-8/255 (3%) | 100% | 0% | 1.00 | 80 |
| 0.20 | +/-16/255 (6%) | 100% | 0% | 1.00 | 81 |
| 0.30 | +/-24/255 (9%) | 100% | 0% | 1.00 | 82 |
| 0.50 | +/-40/255 (16%) | 98% | 0% | 0.98 | 82 |
| 0.70 | +/-56/255 (22%) | 100% | 0% | 1.00 | 81 |
| 1.00 | +/-80/255 (31%) | 100% | 0% | 1.00 | 83 |

## ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 83 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 80 |
| 0.20 | 112px effective (2x downscale) | 100% | 0% | 1.00 | 81 |
| 0.30 | 74px effective (3x downscale) | 100% | 0% | 1.00 | 86 |
| 0.50 | 56px effective (4x downscale) | 72% | 0% | 0.72 | 145 |
| 0.70 | 44px effective (5x downscale) | 78% | 0% | 0.78 | 133 |
| 1.00 | 28px effective (8x downscale) | 4% | 0% | 0.04 | 218 |

## Breakpoints

The intensity at which success rate drops below 50%:

- **LatencyStressor**: fails at intensity **0.50** (100ms)
- **DropoutStressor**: fails at intensity **0.70** (70% dropout)
- **PhysicsShiftStressor**: no breakpoint detected (robust)
- **EmbodimentStressor**: no breakpoint detected (robust)
- **LongHorizonDriftStressor**: no breakpoint detected (robust)
- **ImageNoiseStressor**: no breakpoint detected (robust)
- **OcclusionStressor**: no breakpoint detected (robust)
- **BrightnessShiftStressor**: no breakpoint detected (robust)
- **ResolutionStressor**: fails at intensity **1.00** (28px effective)

---

*Report generated by Trace Robotics v0.1.0*
*https://tracerobotics.com*