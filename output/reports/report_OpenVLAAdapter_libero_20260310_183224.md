# Trace Robotics — Robustness Report

**Policy:** OpenVLAAdapter  
**Task:** libero  
**Modalities:** vision, proprioception  
**Generated:** 2026-03-11 02:01
**Control frequency:** 50Hz (20ms per step)

---

## Executive Summary

- **LatencyStressor**: baseline 100% success, max degradation 100%, breakpoint at intensity 0.30 (60ms)
- **DropoutStressor**: baseline 100% success, max degradation 100%, breakpoint at intensity 0.30 (30% dropout)
- **PhysicsShiftStressor**: baseline 100% success, max degradation 14%, breakpoint at intensity none (robust)
- **EmbodimentStressor**: baseline 100% success, max degradation 0%, breakpoint at intensity none (robust)
- **LongHorizonDriftStressor**: baseline 100% success, max degradation 88%, breakpoint at intensity 0.30 (obs std 0.3 @step100)
- **ImageNoiseStressor**: baseline 100% success, max degradation 0%, breakpoint at intensity none (robust)
- **OcclusionStressor**: baseline 100% success, max degradation 2%, breakpoint at intensity none (robust)
- **BrightnessShiftStressor**: baseline 100% success, max degradation 2%, breakpoint at intensity none (robust)
- **ResolutionStressor**: baseline 100% success, max degradation 64%, breakpoint at intensity 1.00 (28px effective)

## LatencyStressor

| Intensity | Real-World Delay | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0ms (0 steps) | 100% | 0% | 1.00 | 75 |
| 0.10 | 20ms (1 steps) | 100% | 0% | 1.00 | 81 |
| 0.20 | 40ms (2 steps) | 86% | 0% | 0.86 | 106 |
| 0.30 | 60ms (3 steps) | 34% | 0% | 0.34 | 180 |
| 0.50 | 100ms (5 steps) | 0% | 0% | 0.00 | 220 |
| 0.70 | 140ms (7 steps) | 0% | 0% | 0.00 | 220 |
| 1.00 | 200ms (10 steps) | 0% | 0% | 0.00 | 220 |

## DropoutStressor

| Intensity | Drop Probability | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | 0% drop prob | 100% | 0% | 1.00 | 75 |
| 0.10 | 10% drop prob | 72% | 0% | 0.72 | 119 |
| 0.20 | 20% drop prob | 56% | 0% | 0.56 | 145 |
| 0.30 | 30% drop prob | 28% | 0% | 0.28 | 187 |
| 0.50 | 50% drop prob | 10% | 0% | 0.10 | 211 |
| 0.70 | 70% drop prob | 0% | 0% | 0.00 | 220 |
| 1.00 | 100% drop prob | 0% | 0% | 0.00 | 220 |

## PhysicsShiftStressor

| Intensity | Physics Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 75 |
| 0.10 | mass 0.9-1.1x, fric 0.9-1.1x | 100% | 0% | 1.00 | 75 |
| 0.20 | mass 0.9-1.2x, fric 0.9-1.1x | 100% | 0% | 1.00 | 75 |
| 0.30 | mass 0.8-1.3x, fric 0.8-1.1x | 100% | 0% | 1.00 | 75 |
| 0.50 | mass 0.8-1.5x, fric 0.7-1.2x | 100% | 0% | 1.00 | 75 |
| 0.70 | mass 0.7-1.7x, fric 0.5-1.4x | 100% | 0% | 1.00 | 75 |
| 1.00 | mass 0.5-2.0x, fric 0.3-1.5x | 86% | 0% | 0.86 | 95 |

## EmbodimentStressor

| Intensity | Embodiment Perturbation | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------------------|-------------|-------------|------------|-----------|
| 0.00 | nominal | 100% | 0% | 1.00 | 75 |
| 0.10 | links 0.99-1.01x, gains 0.97-1.03x | 100% | 0% | 1.00 | 75 |
| 0.20 | links 0.98-1.02x, gains 0.94-1.06x | 100% | 0% | 1.00 | 75 |
| 0.30 | links 0.97-1.03x, gains 0.91-1.09x | 100% | 0% | 1.00 | 75 |
| 0.50 | links 0.95-1.05x, gains 0.85-1.15x | 100% | 0% | 1.00 | 76 |
| 0.70 | links 0.93-1.07x, gains 0.79-1.21x | 100% | 0% | 1.00 | 77 |
| 1.00 | links 0.90-1.10x, gains 0.70-1.30x | 100% | 0% | 1.00 | 77 |

## LongHorizonDriftStressor

| Intensity | Drift Magnitude | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------------|-------------|-------------|------------|-----------|
| 0.00 | no drift | 100% | 0% | 1.00 | 75 |
| 0.10 | obs noise 0.10, act noise 0.05 @step100 | 86% | 0% | 0.86 | 104 |
| 0.20 | obs noise 0.20, act noise 0.10 @step100 | 56% | 0% | 0.56 | 152 |
| 0.30 | obs noise 0.30, act noise 0.15 @step100 | 38% | 0% | 0.38 | 176 |
| 0.50 | obs noise 0.50, act noise 0.25 @step100 | 18% | 0% | 0.18 | 200 |
| 0.70 | obs noise 0.70, act noise 0.35 @step100 | 14% | 0% | 0.14 | 205 |
| 1.00 | obs noise 1.00, act noise 0.50 @step100 | 12% | 0% | 0.12 | 206 |

## ImageNoiseStressor

| Intensity | Noise Level | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|-------------|------------|-----------|
| 0.00 | std=0/255 (0%) | 100% | 0% | 1.00 | 75 |
| 0.10 | std=5/255 (2%) | 100% | 0% | 1.00 | 75 |
| 0.20 | std=10/255 (4%) | 100% | 0% | 1.00 | 75 |
| 0.30 | std=15/255 (6%) | 100% | 0% | 1.00 | 75 |
| 0.50 | std=25/255 (10%) | 100% | 0% | 1.00 | 75 |
| 0.70 | std=35/255 (14%) | 100% | 0% | 1.00 | 75 |
| 1.00 | std=50/255 (20%) | 100% | 0% | 1.00 | 75 |

## OcclusionStressor

| Intensity | Occlusion | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-----------|-------------|-------------|------------|-----------|
| 0.00 | none | 100% | 0% | 1.00 | 75 |
| 0.10 | 1 patches, up to 3% each | 100% | 0% | 1.00 | 75 |
| 0.20 | 1 patches, up to 6% each | 100% | 0% | 1.00 | 75 |
| 0.30 | 1 patches, up to 9% each | 100% | 0% | 1.00 | 75 |
| 0.50 | 2 patches, up to 15% each | 100% | 0% | 1.00 | 75 |
| 0.70 | 3 patches, up to 21% each | 100% | 0% | 1.00 | 75 |
| 1.00 | 5 patches, up to 30% each | 98% | 0% | 0.98 | 79 |

## BrightnessShiftStressor

| Intensity | Brightness Shift | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|------------------|-------------|-------------|------------|-----------|
| 0.00 | +/-0/255 (0%) | 100% | 0% | 1.00 | 75 |
| 0.10 | +/-8/255 (3%) | 100% | 0% | 1.00 | 75 |
| 0.20 | +/-16/255 (6%) | 98% | 0% | 0.98 | 78 |
| 0.30 | +/-24/255 (9%) | 100% | 0% | 1.00 | 75 |
| 0.50 | +/-40/255 (16%) | 100% | 0% | 1.00 | 75 |
| 0.70 | +/-56/255 (22%) | 98% | 0% | 0.98 | 78 |
| 1.00 | +/-80/255 (31%) | 100% | 0% | 1.00 | 75 |

## ResolutionStressor

| Intensity | Effective Resolution | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|----------------------|-------------|-------------|------------|-----------|
| 0.00 | 224px (native) | 100% | 0% | 1.00 | 75 |
| 0.10 | 224px (native) | 100% | 0% | 1.00 | 75 |
| 0.20 | 112px effective (2x downscale) | 100% | 0% | 1.00 | 76 |
| 0.30 | 74px effective (3x downscale) | 100% | 0% | 1.00 | 77 |
| 0.50 | 56px effective (4x downscale) | 100% | 0% | 1.00 | 76 |
| 0.70 | 44px effective (5x downscale) | 98% | 0% | 0.98 | 80 |
| 1.00 | 28px effective (8x downscale) | 36% | 0% | 0.36 | 168 |

## Breakpoints

The intensity at which success rate drops below 50%:

- **LatencyStressor**: fails at intensity **0.30** (60ms)
- **DropoutStressor**: fails at intensity **0.30** (30% dropout)
- **PhysicsShiftStressor**: no breakpoint detected (robust)
- **EmbodimentStressor**: no breakpoint detected (robust)
- **LongHorizonDriftStressor**: fails at intensity **0.30** (obs std 0.3 @step100)
- **ImageNoiseStressor**: no breakpoint detected (robust)
- **OcclusionStressor**: no breakpoint detected (robust)
- **BrightnessShiftStressor**: no breakpoint detected (robust)
- **ResolutionStressor**: fails at intensity **1.00** (28px effective)

---

*Report generated by Trace Robotics v0.1.0*
*https://tracerobotics.com*