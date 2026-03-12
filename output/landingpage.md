Trace Robotics
Manifesto
Evaluation
Philosophy
Get in Touch

Evaluation Infrastructure
Robustness Testing for Robot Foundation Models.
We tested Pi0.5 and OpenVLA-OFT — two leading vision-language-action models — under identical deployment stressors. Both achieve near-perfect success in ideal conditions. Both collapse at just 100ms of control latency — a delay common in production robot stacks. Trace finds these failures before deployment.

Request Early Access
Request Sample Report
01 — The Evaluation Gap
Robotics is entering the foundation model era, but evaluation has not kept pace with capability.

Hidden Brittleness
Multimodal policies often perform well in demos but degrade sharply under sensor latency, physics mismatches, or embodiment variation.

Fragmented Benchmarks
Internal tests are cherry-picked.
Simulators are not standardized.
Robotics lacks a standardized robustness benchmark similar to RobustBench.
We provide the neutral diagnostic layer to turn hidden failures into actionable insight.

02 — What We Measure
Action Latency & Noise
Stress-testing policies against real-world delays (50-200ms) and sensor dropouts to ensure deployment reliability.

Physics Shifts
Validating behavior across friction, mass, and damping variations to prevent brittleness in novel environments.

Embodiment Perturbation
Ensuring policies generalize to slight mechanical variations, joint limit changes, and arm length differences.

Robustness Reporting
Executive-grade analysis identifying exactly where policies break, with catastrophic failure rates and recovery metrics.

/// Diagnostic Layer

03 — Our View
Robotics is advancing faster than our ability to evaluate it. Trace exists to close that gap.
We do not build robots. We make them reliable.

04 — Early Findings
We evaluated Pi0.5 (Physical Intelligence) and OpenVLA-OFT (Stanford) — two architecturally distinct VLAs — under 9 deployment stressors across 630 episodes each.

Key findings:

Both models break at the same latency threshold (intensity 0.30 = 100ms) despite having completely different architectures, action chunking strategies, and inference pipelines. This suggests latency fragility is a systemic property of current VLA policies, not a model-specific bug.

OpenVLA is robust to 5 of 9 stressors (image noise, occlusion, brightness, embodiment, physics shifts — all 100% at max intensity) but fragile to latency, observation dropout, and long-horizon drift.

Latency vs Success Rate (Multi-Model)

Pi0.5:    100% → 0% between intensity 0.30–0.50
OpenVLA:  100% → 0% between intensity 0.30–0.50

Infrastructure delays — not model capability — may be the primary deployment bottleneck.

Request a sample robustness report
05 — Who This Is For
Foundation model labs
Validate robustness before releasing checkpoints.

Robot companies
Test third-party models before deploying on your hardware.

Safety & compliance teams
Generate robustness evidence for regulatory review.

Research Driven
Researchers building evaluation infrastructure. We evaluate VLA policies under real-world stressors.

Diagnostic First
Architecting for failure discovery, not just success rates.

Backed by Builders
Supported by early pioneers in AI infrastructure.

TRACE
"Physical intelligence is inevitable.
The infrastructure behind it is not."

Early Access

We have evaluated Pi0.5 and OpenVLA-OFT across 9 stressors with over 1,200 simulation episodes. If you are building robot foundation models and want an independent robustness evaluation, reach out.

Get in Touch
2026
© 2026 Trace Robotics Inc.

System Status: Operational