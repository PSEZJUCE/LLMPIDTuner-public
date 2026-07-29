from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from llmpidtuner.metrics import normalize_prompt_variant
from llmpidtuner.models import FirstOrderPlant, PIDParams, SecondOrderPlant, SimulationSettings


@dataclass(frozen=True)
class CaseConfig:
    name: str
    system: str
    mode: str = "dry_run"
    output_dir: str = "runs"
    initial_pid: PIDParams = field(default_factory=lambda: PIDParams(1.0, 0.1, 0.01))
    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    first_order: FirstOrderPlant | None = None
    second_order: SecondOrderPlant | None = None
    batch: dict[str, Any] | None = None
    max_iterations: int = 2
    success_overshoot: float = 15.0
    success_steady_state_error: float = 1.0
    demonstration_path: str | None = None
    control_style: str = "balanced"
    demonstration: dict[str, Any] | None = None
    prompt_variant: str = "full"
    imc: dict[str, Any] | None = None
    llm_profile: str | None = None
    llm: dict[str, Any] | None = None
    resume: bool = False


def load_case_config(path: str | Path) -> CaseConfig:
    case_path = Path(path)
    data = yaml.safe_load(case_path.read_text(encoding="utf-8"))

    initial_pid_data = data.get("initial_pid", {})
    simulation_data = data.get("simulation", {})
    plant_data = data.get("plant", {})
    demonstration = data.get("demonstration")
    prompt_variant = normalize_prompt_variant(data.get("prompt_variant"))
    if demonstration and demonstration.get("method") == "frozen":
        demonstration_variant = normalize_prompt_variant(
            demonstration.get("prompt_variant", prompt_variant)
        )
        if demonstration_variant != prompt_variant:
            raise ValueError(
                "Case prompt_variant must match the frozen demonstration prompt_variant: "
                f"{prompt_variant!r} != {demonstration_variant!r}."
            )

    first_order = None
    second_order = None
    if data["system"] == "first_order":
        first_order = FirstOrderPlant(k=float(plant_data["k"]), t=float(plant_data["t"]))
    elif data["system"] == "second_order":
        second_order = SecondOrderPlant(
            k=float(plant_data["k"]),
            tau1=float(plant_data["tau1"]),
            tau2=float(plant_data["tau2"]),
        )
    elif data["system"] not in {"first_order_batch", "second_order_batch"}:
        raise ValueError(f"Unsupported system: {data['system']}")

    return CaseConfig(
        name=case_path.stem,
        system=data["system"],
        mode=data.get("mode", "dry_run"),
        output_dir=data.get("output_dir", "runs"),
        initial_pid=PIDParams(
            kp=float(initial_pid_data.get("kp", 1.0)),
            ki=float(initial_pid_data.get("ki", 0.1)),
            kd=float(initial_pid_data.get("kd", 0.01)),
        ),
        simulation=SimulationSettings(
            setpoint=float(simulation_data.get("setpoint", 1.0)),
            sim_time=float(simulation_data.get("sim_time", 4000.0)),
            num_points=int(simulation_data.get("num_points", 40001)),
            time_delay=float(simulation_data.get("time_delay", 20.0)),
            max_abs_output=None
            if simulation_data.get("max_abs_output") is None
            else float(simulation_data["max_abs_output"]),
        ),
        first_order=first_order,
        second_order=second_order,
        batch=data.get("batch"),
        max_iterations=int(data.get("max_iterations", 2)),
        success_overshoot=float(data.get("success", {}).get("overshoot", 15.0)),
        success_steady_state_error=float(data.get("success", {}).get("steady_state_error", 1.0)),
        demonstration_path=data.get("demonstration_path"),
        demonstration=demonstration,
        prompt_variant=prompt_variant,
        imc=data.get("imc"),
        control_style=str(data.get("control_style", "balanced")),
        llm_profile=data.get("llm_profile") or data.get("model_profile"),
        llm=data.get("llm"),
    )
