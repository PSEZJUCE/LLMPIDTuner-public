from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from llmpidtuner.config import CaseConfig
from llmpidtuner.models import PIDParams


@dataclass(frozen=True)
class FrequencyPlant:
    """FOPDT or SOPDT process used for exact-delay frequency-domain comparison."""

    system: Literal["first_order", "second_order"]
    k: float
    time_delay: float
    t: float | None = None
    tau1: float | None = None
    tau2: float | None = None

    @classmethod
    def from_case(cls, case: CaseConfig) -> "FrequencyPlant":
        if case.system == "first_order" and case.first_order is not None:
            return cls(
                system="first_order",
                k=case.first_order.k,
                t=case.first_order.t,
                time_delay=case.simulation.time_delay,
            )
        if case.system == "second_order" and case.second_order is not None:
            return cls(
                system="second_order",
                k=case.second_order.k,
                tau1=case.second_order.tau1,
                tau2=case.second_order.tau2,
                time_delay=case.simulation.time_delay,
            )
        raise ValueError("Frequency comparison requires a single first_order or second_order case.")

    @property
    def characteristic_time(self) -> float:
        if self.system == "second_order":
            return max(math.sqrt(float(self.tau1 or 1.0) * float(self.tau2 or 1.0)), 1e-12)
        return max(float(self.t or 1.0), 1e-12)


@dataclass(frozen=True)
class FrequencySummary:
    label: str
    pid: PIDParams
    phase_margin_deg: float | None
    critical_crossover_frequency: float | None
    all_crossovers: list[dict[str, float]]

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "pid": asdict(self.pid),
            "phase_margin_deg": self.phase_margin_deg,
            "critical_crossover_frequency": self.critical_crossover_frequency,
            "all_crossovers": self.all_crossovers,
        }


def compare_case_frequency_response(
    case: CaseConfig,
    base_pid: PIDParams,
    grpo_pid: PIDParams,
    output_path: str | Path,
    *,
    base_label: str = "Base model",
    grpo_label: str = "GRPO model",
    summary_path: str | Path | None = None,
    points: int = 4096,
    show_titles: bool = True,
) -> tuple[Path, Path | None, list[FrequencySummary]]:
    """Plot exact-delay Bode and Nyquist responses for two supplied PID controllers."""

    return compare_frequency_response(
        FrequencyPlant.from_case(case),
        base_pid,
        grpo_pid,
        output_path,
        base_label=base_label,
        grpo_label=grpo_label,
        summary_path=summary_path,
        points=points,
        show_titles=show_titles,
    )


def compare_frequency_response(
    plant: FrequencyPlant,
    base_pid: PIDParams,
    grpo_pid: PIDParams,
    output_path: str | Path,
    *,
    base_label: str = "Base model",
    grpo_label: str = "GRPO model",
    summary_path: str | Path | None = None,
    points: int = 4096,
    show_titles: bool = True,
) -> tuple[Path, Path | None, list[FrequencySummary]]:
    """Compare two PID controllers without Pade approximation of the time delay."""

    _validate_plant(plant)
    controllers = [(base_label, base_pid, "#e45756"), (grpo_label, grpo_pid, "#2f80ed")]
    omega = _adaptive_frequency_grid(plant, points)
    responses = [_frequency_response(plant, pid, omega) for _, pid, _ in controllers]
    summaries = [
        _summarize_frequency_response(plant, label, pid, omega, loop)
        for (label, pid, _), loop in zip(controllers, responses, strict=True)
    ]

    figure_path = Path(output_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_frequency_comparison(
        plant,
        omega,
        controllers,
        responses,
        summaries,
        figure_path,
        show_titles=show_titles,
    )

    written_summary: Path | None = None
    if summary_path is not None:
        written_summary = Path(summary_path)
        written_summary.parent.mkdir(parents=True, exist_ok=True)
        written_summary.write_text(
            json.dumps(
                {
                    "plant": asdict(plant),
                    "delay_evaluation": "exact_exp_minus_j_omega_theta",
                    "frequency_range_rad_per_s": [float(omega[0]), float(omega[-1])],
                    "frequency_points": int(len(omega)),
                    "controllers": [summary.as_dict() for summary in summaries],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return figure_path, written_summary, summaries


def _frequency_response(plant: FrequencyPlant, pid: PIDParams, omega: np.ndarray) -> np.ndarray:
    s = 1j * omega
    controller = pid.kp + pid.ki / s + pid.kd * s
    if plant.system == "second_order":
        process = plant.k / ((1.0 + s * float(plant.tau1)) * (1.0 + s * float(plant.tau2)))
    else:
        process = plant.k / (1.0 + s * float(plant.t))
    return controller * process * np.exp(-s * plant.time_delay)


def _adaptive_frequency_grid(plant: FrequencyPlant, points: int) -> np.ndarray:
    if points < 64:
        raise ValueError("points must be at least 64.")
    characteristic_time = plant.characteristic_time
    return np.logspace(
        math.log10(1.0e-5 / characteristic_time),
        math.log10(1.0e5 / characteristic_time),
        int(points),
    )


def _summarize_frequency_response(
    plant: FrequencyPlant,
    label: str,
    pid: PIDParams,
    omega: np.ndarray,
    loop: np.ndarray,
) -> FrequencySummary:
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(loop), np.finfo(float).tiny))
    crossovers = _unit_gain_crossovers(plant, pid, omega, magnitude_db)
    critical = min(crossovers, key=lambda item: item["phase_margin_deg"]) if crossovers else None
    return FrequencySummary(
        label=label,
        pid=pid,
        phase_margin_deg=None if critical is None else float(critical["phase_margin_deg"]),
        critical_crossover_frequency=None
        if critical is None
        else float(critical["frequency_rad_per_s"]),
        all_crossovers=crossovers,
    )


def _loop_magnitude_db(plant: FrequencyPlant, pid: PIDParams, omega: float) -> float:
    loop = _frequency_response(plant, pid, np.asarray([omega], dtype=np.float64))[0]
    return float(20.0 * math.log10(max(abs(loop), np.finfo(float).tiny)))


def _exact_loop_phase_deg(plant: FrequencyPlant, pid: PIDParams, omega: float) -> float:
    controller_phase = math.atan2(pid.kd * omega - pid.ki / omega, pid.kp)
    if plant.system == "second_order":
        process_phase = -math.atan(omega * float(plant.tau1)) - math.atan(omega * float(plant.tau2))
    else:
        process_phase = -math.atan(omega * float(plant.t))
    return float(math.degrees(controller_phase + process_phase - omega * plant.time_delay))


def _unit_gain_crossovers(
    plant: FrequencyPlant,
    pid: PIDParams,
    omega: np.ndarray,
    magnitude_db: np.ndarray,
) -> list[dict[str, float]]:
    log_omega = np.log(omega)
    crossings = np.flatnonzero(
        (magnitude_db[:-1] == 0.0) | (magnitude_db[:-1] * magnitude_db[1:] < 0.0)
    )
    result: list[dict[str, float]] = []
    for index in crossings:
        left, right = float(log_omega[index]), float(log_omega[index + 1])
        left_value = float(magnitude_db[index])
        for _ in range(64):
            middle = 0.5 * (left + right)
            magnitude = _loop_magnitude_db(plant, pid, math.exp(middle))
            if abs(magnitude) <= 1e-10:
                left = right = middle
                break
            if (left_value <= 0.0 <= magnitude) or (magnitude <= 0.0 <= left_value):
                right = middle
            else:
                left, left_value = middle, magnitude
        log_crossover = 0.5 * (left + right)
        crossover = math.exp(log_crossover)
        result.append(
            {
                "frequency_rad_per_s": float(crossover),
                "phase_margin_deg": float(180.0 + _exact_loop_phase_deg(plant, pid, crossover)),
            }
        )
    return result


def _plot_frequency_comparison(
    plant: FrequencyPlant,
    omega: np.ndarray,
    controllers: list[tuple[str, PIDParams, str]],
    responses: list[np.ndarray],
    summaries: list[FrequencySummary],
    output_path: Path,
    *,
    show_titles: bool,
) -> None:
    figure = plt.figure(figsize=(12, 7.2))
    grid = figure.add_gridspec(2, 2, width_ratios=(1.15, 1.0), hspace=0.08, wspace=0.30)
    magnitude_axis = figure.add_subplot(grid[0, 0])
    phase_axis = figure.add_subplot(grid[1, 0], sharex=magnitude_axis)
    nyquist_axis = figure.add_subplot(grid[:, 1])

    for (label, _, color), loop, summary in zip(controllers, responses, summaries, strict=True):
        magnitude_db = 20.0 * np.log10(np.maximum(np.abs(loop), np.finfo(float).tiny))
        phase_deg = np.degrees(np.unwrap(np.angle(loop)))
        magnitude_axis.semilogx(omega, magnitude_db, color=color, linewidth=1.6, label=label)
        phase_axis.semilogx(omega, phase_deg, color=color, linewidth=1.6, label=label)
        nyquist_axis.plot(
            loop.real, loop.imag, color=color, linewidth=1.6, label=f"{label} (+omega)"
        )
        nyquist_axis.plot(
            loop.real, -loop.imag, color=color, linewidth=1.0, linestyle="--", alpha=0.8
        )

        if summary.critical_crossover_frequency is not None:
            magnitude_axis.axvline(
                summary.critical_crossover_frequency,
                color=color,
                linewidth=0.8,
                linestyle=":",
                alpha=0.9,
            )

    magnitude_axis.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--")
    phase_axis.axhline(-180.0, color="#555555", linewidth=0.8, linestyle="--")
    magnitude_axis.set_ylabel("Magnitude (dB)")
    phase_axis.set_ylabel("Phase (deg)")
    phase_axis.set_xlabel("Frequency (rad/s)")
    if show_titles:
        magnitude_axis.set_title(_title_for_plant(plant))
    magnitude_axis.tick_params(labelbottom=False)
    for axis in (magnitude_axis, phase_axis):
        axis.grid(True, which="both", alpha=0.25)
        axis.margins(x=0)

    nyquist_axis.plot(-1.0, 0.0, "o", color="#222222", markersize=5, label="Critical point (-1, 0)")
    nyquist_axis.axhline(0.0, color="#888888", linewidth=0.7)
    nyquist_axis.axvline(0.0, color="#888888", linewidth=0.7)
    if show_titles:
        nyquist_axis.set_title("Nyquist Plot (Exact Delay)")
    nyquist_axis.set_xlabel("Real")
    nyquist_axis.set_ylabel("Imaginary")
    nyquist_axis.grid(True, alpha=0.25)
    nyquist_axis.set_aspect("equal", adjustable="datalim")

    labels = []
    for summary in summaries:
        pm = "N/A" if summary.phase_margin_deg is None else f"{summary.phase_margin_deg:.1f} deg"
        labels.append(f"{summary.label}: PM={pm}")
    magnitude_axis.legend(frameon=False, loc="best", title="\n".join(labels))
    nyquist_axis.legend(frameon=False, loc="best")
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _title_for_plant(plant: FrequencyPlant) -> str:
    if plant.system == "second_order":
        return (
            "SOPDT Bode Plot: "
            f"K={plant.k:.4g}, tau1={float(plant.tau1):.4g}, "
            f"tau2={float(plant.tau2):.4g}, delay={plant.time_delay:.4g}"
        )
    return f"FOPDT Bode Plot: K={plant.k:.4g}, T={float(plant.t):.4g}, delay={plant.time_delay:.4g}"


def _validate_plant(plant: FrequencyPlant) -> None:
    if plant.k <= 0.0:
        raise ValueError("Frequency comparison currently requires plant gain K > 0.")
    if plant.time_delay < 0.0:
        raise ValueError("time_delay must be non-negative.")
    if plant.system == "first_order" and (plant.t is None or plant.t <= 0.0):
        raise ValueError("A first-order plant requires t > 0.")
    if plant.system == "second_order" and (
        plant.tau1 is None or plant.tau2 is None or plant.tau1 <= 0.0 or plant.tau2 <= 0.0
    ):
        raise ValueError("A second-order plant requires tau1 > 0 and tau2 > 0.")
