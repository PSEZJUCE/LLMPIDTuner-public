from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from llmpidtuner.models import PIDParams
from llmpidtuner.simulation import SimulationResult


def write_simulation_files(
    result: SimulationResult,
    pid: PIDParams,
    output_dir: str | Path,
    stage_label: str | None = None,
) -> tuple[Path, Path]:
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    suffix = f"_{stage_label}" if stage_label else ""
    parameter_file = folder / f"parameter_PID_IAE{suffix}.txt"
    curve_file = folder / f"value_curve{suffix}.txt"

    parameter_file.write_text(
        f"Kp={pid.kp:.3f}, Ki={pid.ki:.3f}, Kd={pid.kd:.3f}, IAE={result.iae:.2f}",
        encoding="utf-8",
    )
    with curve_file.open("w", encoding="utf-8") as file:
        file.write("Results Array (Time, Setpoint, Output):\n")
        np.savetxt(
            file,
            result.results_array,
            fmt="%.2f %.5f %.5f",
            header="Time Setpoint Output",
            comments="",
        )
    return parameter_file, curve_file


def write_results_excel(records: list[dict[str, object]], path: str | Path) -> None:
    if not records:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_excel(output_path, index=False)
