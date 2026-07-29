from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from llmpidtuner.demonstrations import imc_pid_tuning_first_order, imc_pid_tuning_second_order
from llmpidtuner.metrics import calculate_response_metrics
from llmpidtuner.experiment_protocol import imc_pid_for_style
from llmpidtuner.models import (
    FirstOrderPlant, PIDParams, ResponseMetrics, SecondOrderPlant, SimulationSettings
)
from llmpidtuner.simulation import FirstOrderDelaySimulator, SecondOrderDelaySimulator


@dataclass(frozen=True)
class PlantSpec:
    plant_type: str
    k: float
    t: float | None = None
    tau1: float | None = None
    tau2: float | None = None
    time_delay: float = 1.0
    setpoint: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "PlantSpec":
        return PlantSpec(
            plant_type=str(data["plant_type"]),
            k=float(data["k"]),
            t=None if data.get("t") is None else float(data["t"]),
            tau1=None if data.get("tau1") is None else float(data["tau1"]),
            tau2=None if data.get("tau2") is None else float(data["tau2"]),
            time_delay=float(data.get("time_delay", 1.0)),
            setpoint=float(data.get("setpoint", 1.0)),
        )


@dataclass
class TrainingSimulationResult:
    time: np.ndarray
    setpoint: np.ndarray
    output: np.ndarray
    control: np.ndarray
    errors: np.ndarray
    metrics: ResponseMetrics


def sample_plant(
    rng: np.random.Generator,
    second_order_prob: float = 0.5,
    first_k_range: tuple[float, float] = (0.2, 0.9),
    first_t_range: tuple[float, float] = (100.0, 600.0),
    second_k_range: tuple[float, float] = (0.1, 3.0),
    tau_range: tuple[float, float] = (0.1, 100.0),
    time_delay: float = 1.0,
    setpoint: float = 1.0,
) -> PlantSpec:
    if rng.random() < second_order_prob:
        return PlantSpec(
            plant_type="second_order",
            k=float(rng.uniform(*second_k_range)),
            tau1=float(rng.uniform(*tau_range)),
            tau2=float(rng.uniform(*tau_range)),
            time_delay=float(time_delay),
            setpoint=float(setpoint),
        )
    return PlantSpec(
        plant_type="first_order",
        k=float(rng.uniform(*first_k_range)),
        t=float(rng.uniform(*first_t_range)),
        time_delay=float(time_delay),
        setpoint=float(setpoint),
    )


def imc_pid_for_plant(
    plant: PlantSpec,
    lambda_value: float = 10.0,
    control_style: str | None = None,
) -> PIDParams:
    if control_style:
        model_plant = (
            SecondOrderPlant(
                plant.k,
                float(plant.tau1 or 1.0),
                float(plant.tau2 or 1.0),
            )
            if plant.plant_type == "second_order"
            else FirstOrderPlant(plant.k, float(plant.t or 1.0))
        )
        return imc_pid_for_style(model_plant, plant.time_delay, control_style)
    if plant.plant_type == "second_order":
        return imc_pid_tuning_second_order(
            plant.k,
            float(plant.tau1 or 1.0),
            float(plant.tau2 or 1.0),
            time_delay=plant.time_delay,
            lambda_value=lambda_value,
        )
    return imc_pid_tuning_first_order(
        plant.k,
        float(plant.t or 1.0),
        time_delay=plant.time_delay,
        lambda_value=lambda_value,
    )


def simulate_pid(
    plant: PlantSpec,
    pid: PIDParams,
    settings: SimulationSettings,
) -> TrainingSimulationResult:
    plant_settings = SimulationSettings(
        setpoint=plant.setpoint,
        sim_time=settings.sim_time,
        num_points=settings.num_points,
        time_delay=plant.time_delay,
        max_abs_output=settings.max_abs_output,
    )
    if plant.plant_type == "second_order":
        raw = SecondOrderDelaySimulator(
            SecondOrderPlant(plant.k, float(plant.tau1 or 1.0), float(plant.tau2 or 1.0)),
            pid,
            plant_settings,
        ).run()
    else:
        raw = FirstOrderDelaySimulator(
            FirstOrderPlant(plant.k, float(plant.t or 1.0)),
            pid,
            plant_settings,
        ).run()

    setpoint = np.full_like(raw.output, plant.setpoint, dtype=np.float64)
    metrics = calculate_response_metrics(
        raw.time,
        raw.output,
        raw.control_signal,
        raw.errors,
        raw.iae,
        setpoint=plant.setpoint,
        time_delay=plant.time_delay,
        finite=raw.finite,
    )
    return TrainingSimulationResult(
        time=raw.time,
        setpoint=setpoint,
        output=raw.output,
        control=raw.control_signal,
        errors=raw.errors,
        metrics=metrics,
    )




def plant_description(plant: PlantSpec) -> str:
    if plant.plant_type == "second_order":
        return (
            "Plant: second-order plus dead-time process, "
            f"K={plant.k:.6g}, tau1={float(plant.tau1 or 0.0):.6g}, "
            f"tau2={float(plant.tau2 or 0.0):.6g}, dead_time={plant.time_delay:.6g}."
        )
    return (
        "Plant: first-order plus dead-time process, "
        f"K={plant.k:.6g}, T={float(plant.t or 0.0):.6g}, dead_time={plant.time_delay:.6g}."
    )


def metrics_description(metrics: ResponseMetrics) -> str:
    settling = "not settled" if not math.isfinite(metrics.settling_time) else f"{metrics.settling_time:.3g} s"
    return (
        f"IAE={metrics.iae:.6g}, overshoot={metrics.overshoot_pct:.3g}%, "
        f"steady_state_error={metrics.steady_state_error_pct:.3g}%, "
        f"settling_time={settling}, oscillations={metrics.oscillation_count}."
    )
