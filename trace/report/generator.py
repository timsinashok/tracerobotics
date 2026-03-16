"""Report generator — produces markdown robustness reports from sweep results.

All stressor intensities are mapped to real-world deployment units using the
formulas from the stressor implementations and the default sweep parameters.
This makes reports immediately interpretable without needing to understand the
internal [0,1] intensity scale.
"""

import datetime
import math
from dataclasses import dataclass, field
from pathlib import Path

from trace.metrics.aggregator import SweepResult
from trace.policy_adapter.base import PolicyMetadata


# ---------------------------------------------------------------------------
# TaskSweepResults — groups sweep results per task for multi-task reports
# ---------------------------------------------------------------------------

@dataclass
class TaskSweepResults:
    """Sweep results for a single task within a multi-task evaluation."""
    suite_name: str
    task_id: int
    task_label: str
    language_instruction: str
    sweep_results: list[SweepResult]


# ---------------------------------------------------------------------------
# Real-world unit mapping for each stressor.
# ---------------------------------------------------------------------------

def _latency_real(intensity: float, params: dict) -> str:
    max_steps = params.get("max_delay_steps", 10)
    steps = int(intensity * max_steps)
    ms = steps * 20
    return f"{ms}ms ({steps} steps)"

def _latency_bp(intensity: float, params: dict) -> str:
    max_steps = params.get("max_delay_steps", 10)
    steps = int(intensity * max_steps)
    ms = steps * 20
    return f"{ms}ms"

def _dropout_real(intensity: float, params: dict) -> str:
    pct = intensity * 100
    return f"{pct:.0f}% drop prob"

def _dropout_bp(intensity: float, params: dict) -> str:
    pct = intensity * 100
    return f"{pct:.0f}% dropout"

def _physics_real(intensity: float, params: dict) -> str:
    if intensity == 0.0:
        return "nominal"
    mass_lo, mass_hi = params.get("mass_range", [0.5, 2.0])
    fric_lo, fric_hi = params.get("friction_range", [0.3, 1.5])
    m_lo = 1.0 + intensity * (mass_lo - 1.0)
    m_hi = 1.0 + intensity * (mass_hi - 1.0)
    f_lo = 1.0 + intensity * (fric_lo - 1.0)
    f_hi = 1.0 + intensity * (fric_hi - 1.0)
    return f"mass {m_lo:.1f}-{m_hi:.1f}x, fric {f_lo:.1f}-{f_hi:.1f}x"

def _physics_bp(intensity: float, params: dict) -> str:
    return _physics_real(intensity, params)

def _embodiment_real(intensity: float, params: dict) -> str:
    if intensity == 0.0:
        return "nominal"
    ll_lo, ll_hi = params.get("link_length_range", [0.9, 1.1])
    g_lo, g_hi = params.get("gain_range", [0.7, 1.3])
    l_lo = 1.0 + intensity * (ll_lo - 1.0)
    l_hi = 1.0 + intensity * (ll_hi - 1.0)
    ga_lo = 1.0 + intensity * (g_lo - 1.0)
    ga_hi = 1.0 + intensity * (g_hi - 1.0)
    return f"links {l_lo:.2f}-{l_hi:.2f}x, gains {ga_lo:.2f}-{ga_hi:.2f}x"

def _embodiment_bp(intensity: float, params: dict) -> str:
    return _embodiment_real(intensity, params)

def _drift_real(intensity: float, params: dict) -> str:
    if intensity == 0.0:
        return "no drift"
    obs_g = params.get("obs_noise_growth", 0.01)
    act_g = params.get("action_noise_growth", 0.005)
    obs_std = obs_g * intensity * 100
    act_std = act_g * intensity * 100
    return f"obs noise {obs_std:.2f}, act noise {act_std:.2f} @step100"

def _drift_bp(intensity: float, params: dict) -> str:
    obs_g = params.get("obs_noise_growth", 0.01)
    obs_std = obs_g * intensity * 100
    return f"obs std {obs_std:.1f} @step100"

def _image_noise_real(intensity: float, params: dict) -> str:
    max_std = params.get("max_noise_std", 50.0)
    std = intensity * max_std
    pct = std / 255 * 100
    return f"std={std:.0f}/255 ({pct:.0f}%)"

def _image_noise_bp(intensity: float, params: dict) -> str:
    max_std = params.get("max_noise_std", 50.0)
    std = intensity * max_std
    return f"noise std={std:.0f}"

def _occlusion_real(intensity: float, params: dict) -> str:
    if intensity == 0.0:
        return "none"
    max_p = params.get("max_patches", 5)
    max_f = params.get("max_patch_frac", 0.3)
    n = max(1, int(intensity * max_p))
    frac = intensity * max_f * 100
    return f"{n} patches, up to {frac:.0f}% each"

def _occlusion_bp(intensity: float, params: dict) -> str:
    return _occlusion_real(intensity, params)

def _brightness_real(intensity: float, params: dict) -> str:
    max_s = params.get("max_shift", 80.0)
    shift = intensity * max_s
    pct = shift / 255 * 100
    return f"+/-{shift:.0f}/255 ({pct:.0f}%)"

def _brightness_bp(intensity: float, params: dict) -> str:
    max_s = params.get("max_shift", 80.0)
    shift = intensity * max_s
    return f"+/-{shift:.0f} px"

def _resolution_real(intensity: float, params: dict) -> str:
    max_f = params.get("max_downscale_factor", 8)
    factor = max(1, int(1 + intensity * (max_f - 1)))
    if factor <= 1:
        return "224px (native)"
    eff = 224 // factor
    return f"{eff}px effective ({factor}x downscale)"

def _resolution_bp(intensity: float, params: dict) -> str:
    max_f = params.get("max_downscale_factor", 8)
    factor = max(1, int(1 + intensity * (max_f - 1)))
    eff = 224 // factor
    return f"{eff}px effective"


_REAL_UNIT_MAP: dict[str, dict] = {
    "LatencyStressor": {"label": "Real-World Delay", "fn": _latency_real, "bp_fn": _latency_bp},
    "DropoutStressor": {"label": "Drop Probability", "fn": _dropout_real, "bp_fn": _dropout_bp},
    "PhysicsShiftStressor": {"label": "Physics Perturbation", "fn": _physics_real, "bp_fn": _physics_bp},
    "EmbodimentStressor": {"label": "Embodiment Perturbation", "fn": _embodiment_real, "bp_fn": _embodiment_bp},
    "LongHorizonDriftStressor": {"label": "Drift Magnitude", "fn": _drift_real, "bp_fn": _drift_bp},
    "ImageNoiseStressor": {"label": "Noise Level", "fn": _image_noise_real, "bp_fn": _image_noise_bp},
    "OcclusionStressor": {"label": "Occlusion", "fn": _occlusion_real, "bp_fn": _occlusion_bp},
    "BrightnessShiftStressor": {"label": "Brightness Shift", "fn": _brightness_real, "bp_fn": _brightness_bp},
    "ResolutionStressor": {"label": "Effective Resolution", "fn": _resolution_real, "bp_fn": _resolution_bp},
}


def _get_real_unit(stressor_name: str, intensity: float, params: dict) -> str:
    entry = _REAL_UNIT_MAP.get(stressor_name)
    if entry is None:
        return ""
    return entry["fn"](intensity, params)


def _get_real_unit_label(stressor_name: str) -> str | None:
    entry = _REAL_UNIT_MAP.get(stressor_name)
    return entry["label"] if entry else None


def _get_breakpoint_real(stressor_name: str, intensity: float, params: dict) -> str:
    entry = _REAL_UNIT_MAP.get(stressor_name)
    if entry is None:
        return f"{intensity:.2f}"
    return entry["bp_fn"](intensity, params)


_DEFAULT_STRESSOR_PARAMS: dict[str, dict] = {
    "LatencyStressor": {"max_delay_steps": 10},
    "DropoutStressor": {"mode": "zero", "noise_scale": 0.1},
    "PhysicsShiftStressor": {"mass_range": [0.5, 2.0], "friction_range": [0.3, 1.5], "damping_range": [0.5, 2.0]},
    "EmbodimentStressor": {"link_length_range": [0.9, 1.1], "joint_limit_range": [0.85, 1.0], "gain_range": [0.7, 1.3]},
    "LongHorizonDriftStressor": {"obs_noise_growth": 0.01, "action_noise_growth": 0.005},
    "ImageNoiseStressor": {"max_noise_std": 50.0},
    "OcclusionStressor": {"max_patches": 5, "max_patch_frac": 0.3, "fill_value": 0},
    "BrightnessShiftStressor": {"max_shift": 80.0},
    "ResolutionStressor": {"max_downscale_factor": 8},
}


class ReportGenerator:
    """Generates markdown robustness reports from evaluation results."""

    def __init__(self, output_dir: str = "output/reports") -> None:
        self.output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # Single-task report (backwards compatible)
    # ------------------------------------------------------------------

    def generate(
        self,
        policy_meta: PolicyMetadata,
        task_name: str,
        sweep_results: list[SweepResult],
        filepath: str | None = None,
    ) -> str:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if filepath is not None:
            filepath = Path(filepath)
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{policy_meta.name}_{task_name}_{timestamp}.md"
            filepath = self.output_dir / filename

        sections = [
            self._header(policy_meta, task_name),
            self._summary(sweep_results),
        ]
        for result in sweep_results:
            sections.append(self._stressor_section(result))
        sections.append(self._breakpoints(sweep_results))
        sections.append(self._footer())

        content = "\n\n".join(sections)
        filepath.write_text(content)
        return str(filepath)

    # ------------------------------------------------------------------
    # Multi-task report
    # ------------------------------------------------------------------

    def generate_multi_task(
        self,
        policy_meta: PolicyMetadata,
        task_results: list[TaskSweepResults],
        filepath: str | None = None,
    ) -> str:
        """Generate a unified report across multiple tasks/suites."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if filepath is not None:
            filepath = Path(filepath)
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.output_dir / f"report_{policy_meta.name}_multi_{timestamp}.md"

        # Group by suite
        suites: dict[str, list[TaskSweepResults]] = {}
        for tr in task_results:
            suites.setdefault(tr.suite_name, []).append(tr)

        sections = [
            self._multi_header(policy_meta, task_results, suites),
            self._cross_suite_summary(suites),
        ]

        # Per-suite sections
        for suite_name, suite_tasks in suites.items():
            sections.append(self._suite_section(suite_name, suite_tasks))

        sections.append(self._cross_suite_breakpoints(suites))
        sections.append(self._footer())

        content = "\n\n".join(sections)
        filepath.write_text(content)
        return str(filepath)

    # ------------------------------------------------------------------
    # Multi-task report building blocks
    # ------------------------------------------------------------------

    def _multi_header(
        self, meta: PolicyMetadata, task_results: list[TaskSweepResults],
        suites: dict[str, list[TaskSweepResults]],
    ) -> str:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        suite_names = ", ".join(suites.keys())
        task_ids = sorted({tr.task_id for tr in task_results})
        return (
            f"# Trace Robotics — Multi-Task Robustness Report\n\n"
            f"**Policy:** {meta.name}  \n"
            f"**Suites:** {suite_names}  \n"
            f"**Task IDs:** {task_ids}  \n"
            f"**Total tasks evaluated:** {len(task_results)}  \n"
            f"**Modalities:** {', '.join(meta.modalities)}  \n"
            f"**Generated:** {now}  \n"
            f"**Control frequency:** 50Hz (20ms per step)\n\n"
            f"---"
        )

    def _cross_suite_summary(self, suites: dict[str, list[TaskSweepResults]]) -> str:
        """Summary table: per-suite average baseline and breakpoints for each stressor."""
        lines = ["## Cross-Suite Summary\n"]

        # Collect all stressor names (from the first task that has results)
        stressor_names = []
        for suite_tasks in suites.values():
            for tr in suite_tasks:
                if tr.sweep_results:
                    stressor_names = [r.stressor_name for r in tr.sweep_results]
                    break
            if stressor_names:
                break

        if not stressor_names:
            lines.append("*No stressor results available yet.*")
            return "\n".join(lines)

        # Header
        suite_cols = " | ".join(suites.keys())
        lines.append(f"| Stressor | {suite_cols} |")
        lines.append("|" + "---|" * (len(suites) + 1))

        # One row per stressor: show "baseline% -> breakpoint" for each suite
        for s_name in stressor_names:
            cells = []
            for suite_name, suite_tasks in suites.items():
                baselines = []
                breakpoints = []
                for tr in suite_tasks:
                    for r in tr.sweep_results:
                        if r.stressor_name == s_name:
                            baselines.append(r.baseline_success_rate())
                            if r.breakpoint_intensity is not None:
                                breakpoints.append(r.breakpoint_intensity)
                if baselines:
                    avg_bl = sum(baselines) / len(baselines)
                    if breakpoints:
                        avg_bp = sum(breakpoints) / len(breakpoints)
                        cells.append(f"{avg_bl:.0%} (bp={avg_bp:.2f})")
                    else:
                        cells.append(f"{avg_bl:.0%} (robust)")
                else:
                    cells.append("--")
            lines.append(f"| {s_name} | {' | '.join(cells)} |")

        return "\n".join(lines)

    def _suite_section(self, suite_name: str, suite_tasks: list[TaskSweepResults]) -> str:
        """Detailed per-suite section with per-task results."""
        lines = [f"## {suite_name}\n"]

        for tr in suite_tasks:
            lines.append(f"### Task {tr.task_id}: *{tr.language_instruction}*\n")

            if not tr.sweep_results:
                lines.append("*Pending...*\n")
                continue

            # Compact summary for this task
            for r in tr.sweep_results:
                baseline = r.baseline_success_rate()
                degradation = r.max_degradation()
                bp = r.breakpoint_intensity
                params = _DEFAULT_STRESSOR_PARAMS.get(r.stressor_name, {})
                if bp is not None:
                    real = _get_breakpoint_real(r.stressor_name, bp, params)
                    bp_str = f"bp={bp:.2f} ({real})"
                else:
                    bp_str = "robust"
                lines.append(
                    f"- **{r.stressor_name}**: {baseline:.0%} baseline, "
                    f"max deg {degradation:.0%}, {bp_str}"
                )
            lines.append("")

            # Detailed table for each stressor
            for r in tr.sweep_results:
                lines.append(self._stressor_section(r))
                lines.append("")

        return "\n".join(lines)

    def _cross_suite_breakpoints(self, suites: dict[str, list[TaskSweepResults]]) -> str:
        """Breakpoint comparison across all suites."""
        lines = ["## Breakpoint Comparison\n"]
        lines.append("Average intensity at which success rate drops below 50%:\n")

        # Collect stressor names
        stressor_names = []
        for suite_tasks in suites.values():
            for tr in suite_tasks:
                if tr.sweep_results:
                    stressor_names = [r.stressor_name for r in tr.sweep_results]
                    break
            if stressor_names:
                break

        for s_name in stressor_names:
            suite_bps = []
            for suite_name, suite_tasks in suites.items():
                bps = []
                for tr in suite_tasks:
                    for r in tr.sweep_results:
                        if r.stressor_name == s_name and r.breakpoint_intensity is not None:
                            bps.append(r.breakpoint_intensity)
                if bps:
                    suite_bps.append(f"{suite_name}: {sum(bps)/len(bps):.2f}")
                else:
                    suite_bps.append(f"{suite_name}: robust")
            lines.append(f"- **{s_name}**: {', '.join(suite_bps)}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Shared building blocks (used by both single and multi-task)
    # ------------------------------------------------------------------

    def _header(self, meta: PolicyMetadata, task_name: str) -> str:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        return (
            f"# Trace Robotics — Robustness Report\n\n"
            f"**Policy:** {meta.name}  \n"
            f"**Task:** {task_name}  \n"
            f"**Modalities:** {', '.join(meta.modalities)}  \n"
            f"**Generated:** {now}  \n"
            f"**Control frequency:** 50Hz (20ms per step)\n\n"
            f"---"
        )

    def _summary(self, results: list[SweepResult]) -> str:
        lines = ["## Executive Summary\n"]
        for r in results:
            baseline = r.baseline_success_rate()
            degradation = r.max_degradation()
            bp = r.breakpoint_intensity
            params = _DEFAULT_STRESSOR_PARAMS.get(r.stressor_name, {})
            if bp is not None:
                real = _get_breakpoint_real(r.stressor_name, bp, params)
                bp_str = f"{bp:.2f} ({real})"
            else:
                bp_str = "none (robust)"
            lines.append(
                f"- **{r.stressor_name}**: baseline {baseline:.0%} success, "
                f"max degradation {degradation:.0%}, "
                f"breakpoint at intensity {bp_str}"
            )
        return "\n".join(lines)

    def _stressor_section(self, result: SweepResult) -> str:
        params = _DEFAULT_STRESSOR_PARAMS.get(result.stressor_name, {})
        real_label = _get_real_unit_label(result.stressor_name)

        lines = [f"#### {result.stressor_name}\n"]

        if real_label:
            lines.append(f"| Intensity | {real_label} | Success Rate | Catastrophic | Avg Reward | Avg Steps |")
            lines.append("|-----------|" + "-" * (len(real_label) + 2) + "|-------------|-------------|------------|-----------|")
        else:
            lines.append("| Intensity | Success Rate | Catastrophic | Avg Reward | Avg Steps |")
            lines.append("|-----------|-------------|-------------|------------|-----------|")

        for s in result.intensity_stats:
            if real_label:
                real_val = _get_real_unit(result.stressor_name, s.intensity, params)
                lines.append(
                    f"| {s.intensity:.2f} | {real_val} | {s.success_rate:.0%} | "
                    f"{s.catastrophic_failure_rate:.0%} | "
                    f"{s.mean_reward:.2f} | {s.mean_steps:.0f} |"
                )
            else:
                lines.append(
                    f"| {s.intensity:.2f} | {s.success_rate:.0%} | "
                    f"{s.catastrophic_failure_rate:.0%} | "
                    f"{s.mean_reward:.2f} | {s.mean_steps:.0f} |"
                )

        return "\n".join(lines)

    def _breakpoints(self, results: list[SweepResult]) -> str:
        lines = ["## Breakpoints\n"]
        lines.append("The intensity at which success rate drops below 50%:\n")
        for r in results:
            bp = r.breakpoint_intensity
            params = _DEFAULT_STRESSOR_PARAMS.get(r.stressor_name, {})
            if bp is not None:
                real = _get_breakpoint_real(r.stressor_name, bp, params)
                lines.append(f"- **{r.stressor_name}**: fails at intensity **{bp:.2f}** ({real})")
            else:
                lines.append(f"- **{r.stressor_name}**: no breakpoint detected (robust)")
        return "\n".join(lines)

    def _footer(self) -> str:
        return (
            "---\n\n"
            "*Report generated by Trace Robotics v0.1.0*\n"
            "*https://tracerobotics.com*"
        )
