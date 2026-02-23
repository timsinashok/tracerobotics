# Trace Robotics - Development Rules

## Project Overview

Trace Robotics is a stress-testing and evaluation platform for multimodal robot foundation policies.
Input: a policy checkpoint. Output: a robustness report showing non-obvious failure modes under stress.

## Architecture

The codebase lives in `trace/` and follows this structure:

- `trace/policy_adapter/` — Wraps policy checkpoints into a unified `act(obs) -> action` interface
- `trace/task_spec/` — Defines simulation environments, success conditions, and episode configs
- `trace/stressor_engine/` — Parameterized stressors (latency, dropout, physics shift, embodiment, drift)
- `trace/runner/` — Episode execution and parameter sweep orchestration
- `trace/metrics/` — Per-step collection and cross-episode aggregation
- `trace/report/` — Auto-generated markdown/PDF robustness reports
- `configs/` — YAML task and sweep configurations
- `tests/` — pytest test suite
- `scripts/` — CLI entry points

## Tech Stack

- **Python 3.11+**
- **MuJoCo** (via `mujoco` Python bindings) as the primary simulator
- **NumPy** for numerical operations
- **PyYAML** for configuration
- **pytest** for testing
- **Matplotlib** for plots in reports
- **Jinja2** for report templating

## Development Rules

### Code Style
- Use type hints on all function signatures
- Use `abc.ABC` and `@abstractmethod` for base classes
- Keep modules focused — one responsibility per file
- Use dataclasses or Pydantic for structured data (prefer dataclasses for internal types)
- Prefer composition over inheritance

### Naming
- Snake_case for files, functions, variables
- PascalCase for classes
- ALL_CAPS for constants
- Prefix abstract base classes with `Base` (e.g., `BaseStressor`, `BaseTask`)

### Design Principles
- Every stressor must be parameterized, sweepable, and reproducible
- All randomness must be seeded and reproducible
- Headless execution only — no UI code
- Keep interfaces minimal — easy to add new tasks/stressors/adapters
- Metrics track *comparative degradation*, not absolute numbers
- The product is the report, not the code

### Testing
- Run tests with: `pytest tests/`
- Every new stressor, task, or adapter needs tests
- Use fixtures for shared MuJoCo environments
- Tests must be deterministic (seeded)

### Configuration
- Task configs go in `configs/tasks/`
- Sweep configs go in `configs/sweeps/`
- Use YAML for all configuration files
- Never hardcode parameters that should be configurable

### What NOT to build
- No UI / dashboard / frontend
- No locomotion, humanoid, or navigation support
- No multi-simulator support yet (MuJoCo only)
- No public benchmarks or leaderboards
- No custom simulator code — use MuJoCo as-is
