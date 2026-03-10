# Trace Robotics — Robustness Report

**Policy:** OpenVLAAdapter  
**Task:** libero  
**Modalities:** vision, proprioception  
**Generated:** 2026-03-11 02:01

---

## Executive Summary

- **LatencyStressor**: baseline 100% success, max degradation 100%, breakpoint at intensity 0.30
- **DropoutStressor**: baseline 100% success, max degradation 100%, breakpoint at intensity 0.30
- **PhysicsShiftStressor**: baseline 100% success, max degradation 14%, breakpoint at intensity none (robust)
- **EmbodimentStressor**: baseline 100% success, max degradation 0%, breakpoint at intensity none (robust)
- **LongHorizonDriftStressor**: baseline 100% success, max degradation 88%, breakpoint at intensity 0.30
- **ImageNoiseStressor**: baseline 100% success, max degradation 0%, breakpoint at intensity none (robust)
- **OcclusionStressor**: baseline 100% success, max degradation 2%, breakpoint at intensity none (robust)
- **BrightnessShiftStressor**: baseline 100% success, max degradation 2%, breakpoint at intensity none (robust)
- **ResolutionStressor**: baseline 100% success, max degradation 64%, breakpoint at intensity 1.00

## LatencyStressor

| Intensity | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|------------|-----------|
| 0.00 | 100% | 0% | 1.00 | 75 |
| 0.10 | 100% | 0% | 1.00 | 81 |
| 0.20 | 86% | 0% | 0.86 | 106 |
| 0.30 | 34% | 0% | 0.34 | 180 |
| 0.50 | 0% | 0% | 0.00 | 220 |
| 0.70 | 0% | 0% | 0.00 | 220 |
| 1.00 | 0% | 0% | 0.00 | 220 |

## DropoutStressor

| Intensity | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|------------|-----------|
| 0.00 | 100% | 0% | 1.00 | 75 |
| 0.10 | 72% | 0% | 0.72 | 119 |
| 0.20 | 56% | 0% | 0.56 | 145 |
| 0.30 | 28% | 0% | 0.28 | 187 |
| 0.50 | 10% | 0% | 0.10 | 211 |
| 0.70 | 0% | 0% | 0.00 | 220 |
| 1.00 | 0% | 0% | 0.00 | 220 |

## PhysicsShiftStressor

| Intensity | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|------------|-----------|
| 0.00 | 100% | 0% | 1.00 | 75 |
| 0.10 | 100% | 0% | 1.00 | 75 |
| 0.20 | 100% | 0% | 1.00 | 75 |
| 0.30 | 100% | 0% | 1.00 | 75 |
| 0.50 | 100% | 0% | 1.00 | 75 |
| 0.70 | 100% | 0% | 1.00 | 75 |
| 1.00 | 86% | 0% | 0.86 | 95 |

## EmbodimentStressor

| Intensity | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|------------|-----------|
| 0.00 | 100% | 0% | 1.00 | 75 |
| 0.10 | 100% | 0% | 1.00 | 75 |
| 0.20 | 100% | 0% | 1.00 | 75 |
| 0.30 | 100% | 0% | 1.00 | 75 |
| 0.50 | 100% | 0% | 1.00 | 76 |
| 0.70 | 100% | 0% | 1.00 | 77 |
| 1.00 | 100% | 0% | 1.00 | 77 |

## LongHorizonDriftStressor

| Intensity | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|------------|-----------|
| 0.00 | 100% | 0% | 1.00 | 75 |
| 0.10 | 86% | 0% | 0.86 | 104 |
| 0.20 | 56% | 0% | 0.56 | 152 |
| 0.30 | 38% | 0% | 0.38 | 176 |
| 0.50 | 18% | 0% | 0.18 | 200 |
| 0.70 | 14% | 0% | 0.14 | 205 |
| 1.00 | 12% | 0% | 0.12 | 206 |

## ImageNoiseStressor

| Intensity | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|------------|-----------|
| 0.00 | 100% | 0% | 1.00 | 75 |
| 0.10 | 100% | 0% | 1.00 | 75 |
| 0.20 | 100% | 0% | 1.00 | 75 |
| 0.30 | 100% | 0% | 1.00 | 75 |
| 0.50 | 100% | 0% | 1.00 | 75 |
| 0.70 | 100% | 0% | 1.00 | 75 |
| 1.00 | 100% | 0% | 1.00 | 75 |

## OcclusionStressor

| Intensity | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|------------|-----------|
| 0.00 | 100% | 0% | 1.00 | 75 |
| 0.10 | 100% | 0% | 1.00 | 75 |
| 0.20 | 100% | 0% | 1.00 | 75 |
| 0.30 | 100% | 0% | 1.00 | 75 |
| 0.50 | 100% | 0% | 1.00 | 75 |
| 0.70 | 100% | 0% | 1.00 | 75 |
| 1.00 | 98% | 0% | 0.98 | 79 |

## BrightnessShiftStressor

| Intensity | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|------------|-----------|
| 0.00 | 100% | 0% | 1.00 | 75 |
| 0.10 | 100% | 0% | 1.00 | 75 |
| 0.20 | 98% | 0% | 0.98 | 78 |
| 0.30 | 100% | 0% | 1.00 | 75 |
| 0.50 | 100% | 0% | 1.00 | 75 |
| 0.70 | 98% | 0% | 0.98 | 78 |
| 1.00 | 100% | 0% | 1.00 | 75 |

## ResolutionStressor

| Intensity | Success Rate | Catastrophic | Avg Reward | Avg Steps |
|-----------|-------------|-------------|------------|-----------|
| 0.00 | 100% | 0% | 1.00 | 75 |
| 0.10 | 100% | 0% | 1.00 | 75 |
| 0.20 | 100% | 0% | 1.00 | 76 |
| 0.30 | 100% | 0% | 1.00 | 77 |
| 0.50 | 100% | 0% | 1.00 | 76 |
| 0.70 | 98% | 0% | 0.98 | 80 |
| 1.00 | 36% | 0% | 0.36 | 168 |

## Breakpoints

The intensity at which success rate drops below 50%:

- **LatencyStressor**: fails at intensity **0.30**
- **DropoutStressor**: fails at intensity **0.30**
- **PhysicsShiftStressor**: no breakpoint detected (robust)
- **EmbodimentStressor**: no breakpoint detected (robust)
- **LongHorizonDriftStressor**: fails at intensity **0.30**
- **ImageNoiseStressor**: no breakpoint detected (robust)
- **OcclusionStressor**: no breakpoint detected (robust)
- **BrightnessShiftStressor**: no breakpoint detected (robust)
- **ResolutionStressor**: fails at intensity **1.00**

---

*Report generated by Trace Robotics v0.1.0*
*https://tracerobotics.com*