# Trace Robotics — Robustness Report

**Policy:** Pi0PolicyAdapter  
**Task:** libero  
**Modalities:** vision, proprioception  
**Generated:** 2026-03-05 02:56

---

## Executive Summary

- **LatencyStressor**: baseline 100% success, max degradation 100%, breakpoint at intensity 0.50
- **DropoutStressor**: baseline 98% success, max degradation 98%, breakpoint at intensity 0.70
- **PhysicsShiftStressor**: baseline 100% success, max degradation 12%, breakpoint at intensity none (robust)
- **EmbodimentStressor**: baseline 100% success, max degradation 0%, breakpoint at intensity none (robust)
- **LongHorizonDriftStressor**: baseline 100% success, max degradation 18%, breakpoint at intensity none (robust)
- **ImageNoiseStressor**: baseline 100% success, max degradation 2%, breakpoint at intensity none (robust)
- **OcclusionStressor**: baseline 100% success, max degradation 6%, breakpoint at intensity none (robust)
- **BrightnessShiftStressor**: baseline 100% success, max degradation 2%, breakpoint at intensity none (robust)
- **ResolutionStressor**: baseline 100% success, max degradation 96%, breakpoint at intensity 1.00

## LatencyStressor

| Intensity | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|------------|-----------|
| 0.00 | 100% | 0% | 1.00 | 80 |
| 0.10 | 100% | 0% | 1.00 | 90 |
| 0.20 | 98% | 0% | 0.98 | 107 |
| 0.30 | 80% | 0% | 0.80 | 158 |
| 0.50 | 14% | 0% | 0.14 | 215 |
| 0.70 | 0% | 0% | 0.00 | 220 |
| 1.00 | 0% | 0% | 0.00 | 220 |

## DropoutStressor

| Intensity | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|------------|-----------|
| 0.00 | 98% | 0% | 0.98 | 86 |
| 0.10 | 96% | 0% | 0.96 | 95 |
| 0.20 | 98% | 0% | 0.98 | 114 |
| 0.30 | 90% | 0% | 0.90 | 130 |
| 0.50 | 50% | 0% | 0.50 | 183 |
| 0.70 | 4% | 0% | 0.04 | 218 |
| 1.00 | 0% | 0% | 0.00 | 220 |

## PhysicsShiftStressor

| Intensity | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|------------|-----------|
| 0.00 | 100% | 0% | 1.00 | 83 |
| 0.10 | 100% | 0% | 1.00 | 82 |
| 0.20 | 100% | 0% | 1.00 | 82 |
| 0.30 | 100% | 0% | 1.00 | 82 |
| 0.50 | 98% | 0% | 0.98 | 84 |
| 0.70 | 100% | 0% | 1.00 | 83 |
| 1.00 | 88% | 0% | 0.88 | 99 |

## EmbodimentStressor

| Intensity | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|------------|-----------|
| 0.00 | 100% | 0% | 1.00 | 81 |
| 0.10 | 100% | 0% | 1.00 | 86 |
| 0.20 | 100% | 0% | 1.00 | 84 |
| 0.30 | 100% | 0% | 1.00 | 86 |
| 0.50 | 100% | 0% | 1.00 | 86 |
| 0.70 | 100% | 0% | 1.00 | 89 |
| 1.00 | 100% | 0% | 1.00 | 93 |

## LongHorizonDriftStressor

| Intensity | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|------------|-----------|
| 0.00 | 100% | 0% | 1.00 | 86 |
| 0.10 | 100% | 0% | 1.00 | 82 |
| 0.20 | 100% | 0% | 1.00 | 80 |
| 0.30 | 100% | 0% | 1.00 | 81 |
| 0.50 | 100% | 0% | 1.00 | 85 |
| 0.70 | 88% | 0% | 0.88 | 103 |
| 1.00 | 82% | 0% | 0.82 | 120 |

## ImageNoiseStressor

| Intensity | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|------------|-----------|
| 0.00 | 100% | 0% | 1.00 | 80 |
| 0.10 | 100% | 0% | 1.00 | 84 |
| 0.20 | 100% | 0% | 1.00 | 81 |
| 0.30 | 100% | 0% | 1.00 | 76 |
| 0.50 | 100% | 0% | 1.00 | 79 |
| 0.70 | 100% | 0% | 1.00 | 80 |
| 1.00 | 98% | 0% | 0.98 | 86 |

## OcclusionStressor

| Intensity | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|------------|-----------|
| 0.00 | 100% | 0% | 1.00 | 82 |
| 0.10 | 100% | 0% | 1.00 | 85 |
| 0.20 | 100% | 0% | 1.00 | 84 |
| 0.30 | 100% | 0% | 1.00 | 83 |
| 0.50 | 100% | 0% | 1.00 | 84 |
| 0.70 | 98% | 0% | 0.98 | 87 |
| 1.00 | 94% | 0% | 0.94 | 98 |

## BrightnessShiftStressor

| Intensity | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|------------|-----------|
| 0.00 | 100% | 0% | 1.00 | 79 |
| 0.10 | 100% | 0% | 1.00 | 80 |
| 0.20 | 100% | 0% | 1.00 | 81 |
| 0.30 | 100% | 0% | 1.00 | 82 |
| 0.50 | 98% | 0% | 0.98 | 82 |
| 0.70 | 100% | 0% | 1.00 | 81 |
| 1.00 | 100% | 0% | 1.00 | 83 |

## ResolutionStressor

| Intensity | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|------------|-----------|
| 0.00 | 100% | 0% | 1.00 | 83 |
| 0.10 | 100% | 0% | 1.00 | 80 |
| 0.20 | 100% | 0% | 1.00 | 81 |
| 0.30 | 100% | 0% | 1.00 | 86 |
| 0.50 | 72% | 0% | 0.72 | 145 |
| 0.70 | 78% | 0% | 0.78 | 133 |
| 1.00 | 4% | 0% | 0.04 | 218 |

## Breakpoints

The intensity at which success rate drops below 50%:

- **LatencyStressor**: fails at intensity **0.50**
- **DropoutStressor**: fails at intensity **0.70**
- **PhysicsShiftStressor**: no breakpoint detected (robust)
- **EmbodimentStressor**: no breakpoint detected (robust)
- **LongHorizonDriftStressor**: no breakpoint detected (robust)
- **ImageNoiseStressor**: no breakpoint detected (robust)
- **OcclusionStressor**: no breakpoint detected (robust)
- **BrightnessShiftStressor**: no breakpoint detected (robust)
- **ResolutionStressor**: fails at intensity **1.00**

---

*Report generated by Trace Robotics v0.1.0*
*https://tracerobotics.com*