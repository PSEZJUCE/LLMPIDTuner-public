from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from llmpidtuner.models import PIDParams
from llmpidtuner.simulation import SimulationResult


def plot_pid_comparison(
    initial_result: SimulationResult,
    final_result: SimulationResult,
    initial_pid: PIDParams,
    final_pid: PIDParams,
    setpoint: float,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(
        initial_result.time,
        initial_result.output,
        label=_label("Initial PID", initial_pid, initial_result.iae),
        linewidth=1.5,
    )

    same_pid = initial_pid == final_pid
    if same_pid:
        title = "PID Response"
    else:
        plt.plot(
            final_result.time,
            final_result.output,
            label=_label("Final PID", final_pid, final_result.iae),
            linewidth=1.5,
        )
        title = "PID Tuning Comparison"

    plt.axhline(setpoint, color="black", linestyle="--", linewidth=1.0, label="Setpoint")
    plt.xlabel("Time (s)")
    plt.ylabel("System Output")
    plt.title(title)
    axis = plt.gca()
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.margins(x=0, y=0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _label(prefix: str, pid: PIDParams, iae: float) -> str:
    return f"{prefix}: Kp={pid.kp:.3f}, Ki={pid.ki:.3f}, Kd={pid.kd:.3f}, IAE={iae:.2f}"
