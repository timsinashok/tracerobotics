# Testing Guide — Trace Robotics

## Quick Start

Run all tests:

```bash
pytest tests/
```

> **Note:** If you hit an `ImportError` related to `logfire` or `opentelemetry`, disable the plugin:
>
> ```bash
> pytest tests/ -p no:logfire
> ```
>
> This is an environment-level conflict, not a project issue. Alternatively, set up a virtual environment (see below).

## Setting Up a Virtual Environment (Recommended)

To isolate project dependencies from your system Python:

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
pip install -e ".[dev]"
```

Once activated, `pytest` will work without the `-p no:logfire` workaround.

## Test Suite Overview

| Module | File | Tests | What it covers |
|---|---|---|---|
| Policy Adapter | `tests/test_policy_adapter.py` | 4 | `RandomPolicy` output shape, bounds, reproducibility, metadata |
| Metrics | `tests/test_metrics.py` | 5 | `EpisodeMetrics` construction, `SweepAggregator` aggregation, breakpoint detection, max degradation |
| Stressors | `tests/test_stressors.py` | 8 | Latency, Dropout, LongHorizonDrift, PhysicsShift, Embodiment stressors — passthrough, intensity, drift behavior |
| Report | `tests/test_report.py` | 1 | Markdown report file generation |
| **Total** | | **18** | |

## Running Specific Tests

```bash
# Run a single test file
pytest tests/test_stressors.py

# Run a single test class
pytest tests/test_stressors.py::TestLatencyStressor

# Run a single test function
pytest tests/test_stressors.py::TestLatencyStressor::test_zero_intensity_passthrough

# Run tests matching a keyword
pytest tests/ -k "dropout"
```

## Useful Options

```bash
# Verbose output (already default via pyproject.toml)
pytest tests/ -v

# Show print statements and stdout
pytest tests/ -s

# Stop on first failure
pytest tests/ -x

# Run with coverage report
pytest tests/ --cov=trace --cov-report=term-missing

# Run only previously failed tests
pytest tests/ --lf
```

## Shared Fixtures (conftest.py)

Defined in `tests/conftest.py`:

| Fixture | Type | Description |
|---|---|---|
| `random_policy` | `RandomPolicy` | 7-DOF random policy, seed=42 |
| `dummy_observation` | `dict[str, np.ndarray]` | 64x64 RGB image + 14-dim proprioception, seed=0 |
| `default_stressor_config` | `StressorConfig` | Name="test_stressor", intensity=0.5, seed=42 |

## Writing New Tests

### Rules

1. **Every new stressor, task, or adapter must have tests**
2. **All tests must be deterministic** — seed all randomness
3. **Use shared fixtures** from `conftest.py` where possible
4. **Follow the existing pattern** — group related tests in a class

### Template

```python
"""Tests for <module>."""

import numpy as np
import pytest

from trace.<module>.your_class import YourClass


class TestYourClass:
    """Tests for YourClass."""

    def test_basic_behavior(self, random_policy, dummy_observation):
        # Arrange
        ...

        # Act
        result = ...

        # Assert
        assert result.shape == expected_shape

    def test_edge_case(self):
        # Seed everything for reproducibility
        rng = np.random.default_rng(123)
        ...
```

### Stressor Test Checklist

When adding a new stressor, include tests for:

- [ ] Zero intensity passes through actions/observations unchanged
- [ ] High intensity produces measurable effect
- [ ] Stressor is reproducible with same seed
- [ ] Stressor works with the standard `dummy_observation` fixture

## CI Notes

No CI pipeline is configured yet. Tests are run locally with:

```bash
pytest tests/
```

Linting and type checking are available via:

```bash
ruff check trace/ tests/
mypy trace/
```
