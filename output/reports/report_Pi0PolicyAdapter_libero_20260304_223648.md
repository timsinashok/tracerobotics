# Trace Robotics — Robustness Report

**Policy:** Pi0PolicyAdapter  
**Task:** libero  
**Modalities:** vision, proprioception  
**Generated:** 2026-03-05 04:53

---

## Executive Summary

- **LatencyStressor**: baseline 98% success, max degradation 98%, breakpoint at intensity 0.50
- **DropoutStressor**: baseline 100% success, max degradation 100%, breakpoint at intensity 0.70
- **PhysicsShiftStressor**: baseline 100% success, max degradation 14%, breakpoint at intensity none (robust)
- **EmbodimentStressor**: baseline 100% success, max degradation 2%, breakpoint at intensity none (robust)
- **LongHorizonDriftStressor**: baseline 100% success, max degradation 22%, breakpoint at intensity none (robust)
- **ImageNoiseStressor**: baseline 100% success, max degradation 0%, breakpoint at intensity none (robust)
- **OcclusionStressor**: baseline 98% success, max degradation 0%, breakpoint at intensity none (robust)
- **BrightnessShiftStressor**: baseline 98% success, max degradation 0%, breakpoint at intensity none (robust)
- **ResolutionStressor**: baseline 100% success, max degradation 98%, breakpoint at intensity 1.00

## LatencyStressor

| Intensity | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|------------|-----------|
| 0.00 | 98% | 0% | 0.98 | 86 |
| 0.10 | 98% | 0% | 0.98 | 96 |
| 0.20 | 100% | 0% | 1.00 | 102 |
| 0.30 | 78% | 0% | 0.78 | 147 |
| 0.50 | 12% | 0% | 0.12 | 217 |
| 0.70 | 0% | 0% | 0.00 | 220 |
| 1.00 | 0% | 0% | 0.00 | 220 |

## DropoutStressor

| Intensity | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|------------|-----------|
| 0.00 | 100% | 0% | 1.00 | 83 |
| 0.10 | 100% | 0% | 1.00 | 92 |
| 0.20 | 96% | 0% | 0.96 | 110 |
| 0.30 | 84% | 0% | 0.84 | 136 |
| 0.50 | 60% | 0% | 0.60 | 178 |
| 0.70 | 8% | 0% | 0.08 | 215 |
| 1.00 | 0% | 0% | 0.00 | 220 |

## PhysicsShiftStressor

| Intensity | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|------------|-----------|
| 0.00 | 100% | 0% | 1.00 | 83 |
| 0.10 | 100% | 0% | 1.00 | 80 |
| 0.20 | 100% | 0% | 1.00 | 84 |
| 0.30 | 100% | 0% | 1.00 | 80 |
| 0.50 | 100% | 0% | 1.00 | 83 |
| 0.70 | 100% | 0% | 1.00 | 89 |
| 1.00 | 86% | 0% | 0.86 | 105 |

## EmbodimentStressor

| Intensity | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|------------|-----------|
| 0.00 | 100% | 0% | 1.00 | 83 |
| 0.10 | 100% | 0% | 1.00 | 82 |
| 0.20 | 98% | 0% | 0.98 | 87 |
| 0.30 | 100% | 0% | 1.00 | 85 |
| 0.50 | 100% | 0% | 1.00 | 86 |
| 0.70 | 100% | 0% | 1.00 | 90 |
| 1.00 | 98% | 0% | 0.98 | 96 |

## LongHorizonDriftStressor

| Intensity | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|------------|-----------|
| 0.00 | 100% | 0% | 1.00 | 83 |
| 0.10 | 100% | 0% | 1.00 | 85 |
| 0.20 | 100% | 0% | 1.00 | 79 |
| 0.30 | 100% | 0% | 1.00 | 79 |
| 0.50 | 100% | 0% | 1.00 | 84 |
| 0.70 | 92% | 0% | 0.92 | 96 |
| 1.00 | 78% | 0% | 0.78 | 122 |

## ImageNoiseStressor

| Intensity | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|------------|-----------|
| 0.00 | 100% | 0% | 1.00 | 80 |
| 0.10 | 100% | 0% | 1.00 | 82 |
| 0.20 | 100% | 0% | 1.00 | 78 |
| 0.30 | 100% | 0% | 1.00 | 78 |
| 0.50 | 100% | 0% | 1.00 | 78 |
| 0.70 | 100% | 0% | 1.00 | 78 |
| 1.00 | 100% | 0% | 1.00 | 85 |

## OcclusionStressor

| Intensity | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|------------|-----------|
| 0.00 | 98% | 0% | 0.98 | 85 |
| 0.10 | 100% | 0% | 1.00 | 80 |
| 0.20 | 98% | 0% | 0.98 | 88 |
| 0.30 | 100% | 0% | 1.00 | 81 |
| 0.50 | 100% | 0% | 1.00 | 80 |
| 0.70 | 100% | 0% | 1.00 | 86 |
| 1.00 | 98% | 0% | 0.98 | 89 |

## BrightnessShiftStressor

| Intensity | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|------------|-----------|
| 0.00 | 98% | 0% | 0.98 | 84 |
| 0.10 | 98% | 0% | 0.98 | 84 |
| 0.20 | 100% | 0% | 1.00 | 82 |
| 0.30 | 98% | 0% | 0.98 | 86 |
| 0.50 | 100% | 0% | 1.00 | 81 |
| 0.70 | 100% | 0% | 1.00 | 82 |
| 1.00 | 100% | 0% | 1.00 | 82 |

## ResolutionStressor

| Intensity | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|------------|-----------|
| 0.00 | 100% | 0% | 1.00 | 82 |
| 0.10 | 100% | 0% | 1.00 | 81 |
| 0.20 | 100% | 0% | 1.00 | 81 |
| 0.30 | 98% | 0% | 0.98 | 87 |
| 0.50 | 70% | 0% | 0.70 | 150 |
| 0.70 | 68% | 0% | 0.68 | 144 |
| 1.00 | 2% | 0% | 0.02 | 219 |

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