from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from decimal import Decimal, localcontext
from typing import Any, Literal

import numpy as np
from scipy.stats import qmc

from llmpidtuner.demonstrations import (
    imc_pid_tuning_first_order,
    imc_pid_tuning_second_order,
)
from llmpidtuner.metrics import calculate_response_metrics
from llmpidtuner.models import (
    FirstOrderPlant,
    PIDParams,
    ResponseMetrics,
    SecondOrderPlant,
    SimulationSettings,
)
from llmpidtuner.simulation import FirstOrderDelaySimulator, SecondOrderDelaySimulator


PROTOCOL_ID = "perturbed_imc_delay_stratified_v1"
SYSTEM_TYPES = ("first_order", "second_order")
STYLES = ("aggressive", "balanced", "conservative")
FAULT_TYPES = (
    "p_high",
    "p_low",
    "i_high",
    "i_low_or_off",
    "i_high_d_high",
    "p_high_d_low",
    "p_high_i_high",
    "p_low_i_high",
    "i_high_d_low",
    "p_low_i_high_d_low",
)
DELAY_BANDS: dict[str, tuple[float, float]] = {
    "low": (0.0, 0.05),
    "medium": (0.05, 0.20),
    "delay_dominant": (0.20, 0.50),
    "strong_delay": (0.50, 0.80),
}
DELAY_QUOTA = (
    "low",
    "low",
    "low",
    "low",
    "medium",
    "medium",
    "medium",
    "delay_dominant",
    "delay_dominant",
    "strong_delay",
)
SEVERITY_QUOTA = (
    "mild",
    "mild",
    "mild",
    "moderate",
    "moderate",
    "moderate",
    "moderate",
    "moderate",
    "severe",
    "severe",
)
# Pair gentle perturbations with harder delays and severe perturbations with
# easier delays. Each fault still receives the accepted 4/3/2/1 delay quota.
DELAY_BY_SEVERITY_SLOT = (
    "strong_delay",
    "delay_dominant",
    "medium",
    "delay_dominant",
    "medium",
    "medium",
    "low",
    "low",
    "low",
    "low",
)
FAULT_SEVERITY_SCHEDULES = {
    "p_high": ("mild",) * 5 + ("moderate",) * 3 + ("severe",) * 2,
    "i_high": ("mild",) * 5 + ("moderate",) * 3 + ("severe",) * 2,
    "p_low": ("moderate",) * 8 + ("severe",) * 2,
    "i_low_or_off": ("moderate",) * 8 + ("severe",) * 2,
    "p_low_i_high": ("moderate",) * 8 + ("severe",) * 2,
}
DEMONSTRATION_DELAY_BY_FAULT = (
    "strong_delay",
    "delay_dominant",
    "medium",
    "medium",
    "medium",
    "delay_dominant",
    "low",
    "low",
    "low",
    "low",
)
STYLE_MULTIPLIERS = {"aggressive": 0.8, "balanced": 2.0, "conservative": 5.0}
DEFAULT_SIMULATION = SimulationSettings(
    setpoint=1.0,
    sim_time=4000.0,
    num_points=40001,
    time_delay=1.0,
    max_abs_output=3.0,
)


@dataclass(frozen=True)
class ProtocolCase:
    case_id: str
    system: str
    plant: FirstOrderPlant | SecondOrderPlant
    time_delay: float
    rho: float
    initial_pid: PIDParams
    reference_pid: PIDParams
    fault_type: str
    severity: str
    perturbation: dict[str, float]
    initial_metrics: ResponseMetrics
    reference_metrics: ResponseMetrics
    seed: int
    candidate_index: int
    perturbation_attempt: int

    def as_dict(self) -> dict[str, Any]:
        plant = asdict(self.plant)
        return {
            "case_id": self.case_id,
            "system": self.system,
            "plant": plant,
            "time_delay": self.time_delay,
            "rho": self.rho,
            "initial_pid": asdict(self.initial_pid),
            "reference_pid": asdict(self.reference_pid),
            "fault_type": self.fault_type,
            "severity": self.severity,
            "perturbation": dict(self.perturbation),
            "initial_metrics": self.initial_metrics.as_dict(),
            "reference_metrics": self.reference_metrics.as_dict(),
            "provenance": {
                "protocol_id": PROTOCOL_ID,
                "seed": self.seed,
                "candidate_index": self.candidate_index,
                "perturbation_attempt": self.perturbation_attempt,
                "case_hash": self.case_hash,
            },
        }

    @property
    def case_hash(self) -> str:
        payload = {
            "system": self.system,
            "plant": asdict(self.plant),
            "time_delay": self.time_delay,
            "initial_pid": asdict(self.initial_pid),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def control_time_constant(plant: FirstOrderPlant | SecondOrderPlant) -> float:
    if isinstance(plant, FirstOrderPlant):
        return float(plant.t)
    return float(plant.tau1 + plant.tau2)


def lambda_for_style(
    plant: FirstOrderPlant | SecondOrderPlant,
    time_delay: float,
    style: str = "balanced",
) -> float:
    if style not in STYLE_MULTIPLIERS:
        raise ValueError(f"Unsupported control style: {style!r}")
    base = max(float(time_delay), 0.1 * control_time_constant(plant))
    return STYLE_MULTIPLIERS[style] * base


def imc_pid_for_style(
    plant: FirstOrderPlant | SecondOrderPlant,
    time_delay: float,
    style: str = "balanced",
) -> PIDParams:
    lambda_value = lambda_for_style(plant, time_delay, style)
    if isinstance(plant, FirstOrderPlant):
        pid = imc_pid_tuning_first_order(
            plant.k,
            plant.t,
            time_delay=time_delay,
            lambda_value=lambda_value,
        )
    else:
        pid = imc_pid_tuning_second_order(
            plant.k,
            plant.tau1,
            plant.tau2,
            time_delay=time_delay,
            lambda_value=lambda_value,
        )
    return _serialize_pid(pid)


def dimensionless_pid(
    plant: FirstOrderPlant | SecondOrderPlant,
    pid: PIDParams,
) -> tuple[float, float, float]:
    tc = control_time_constant(plant)
    return plant.k * pid.kp, plant.k * pid.ki * tc, plant.k * pid.kd / tc


def simulate_protocol_case(
    plant: FirstOrderPlant | SecondOrderPlant,
    pid: PIDParams,
    time_delay: float,
    simulation: SimulationSettings = DEFAULT_SIMULATION,
) -> tuple[Any, ResponseMetrics]:
    settings = SimulationSettings(
        setpoint=simulation.setpoint,
        sim_time=simulation.sim_time,
        num_points=simulation.num_points,
        time_delay=float(time_delay),
        max_abs_output=simulation.max_abs_output,
    )
    simulator = (
        FirstOrderDelaySimulator(plant, pid, settings)
        if isinstance(plant, FirstOrderPlant)
        else SecondOrderDelaySimulator(plant, pid, settings)
    )
    result = simulator.run()
    metrics = calculate_response_metrics(
        result.time,
        result.output,
        result.control_signal,
        result.errors,
        result.iae,
        setpoint=settings.setpoint,
        time_delay=settings.time_delay,
        finite=result.finite,
    )
    return result, metrics


def initial_case_is_acceptable(
    initial: ResponseMetrics,
    reference: ResponseMetrics,
    *,
    minimum_iae_ratio: float = 1.5,
) -> bool:
    if not initial.finite or not reference.finite or reference.iae <= 0:
        return False
    return bool(
        not initial.converged()
        and initial.iae / reference.iae >= minimum_iae_ratio
        and max(abs(initial.max_value), abs(initial.min_value)) <= 3.0
    )


def severity_from_iae_ratio(ratio: float) -> str:
    if ratio < 2.0:
        return "mild"
    if ratio < 5.0:
        return "moderate"
    return "severe"


def generate_protocol_cases(
    system: Literal["first_order", "second_order"],
    count: int,
    seed: int,
    *,
    purpose: str,
    require_all_style_targets: bool = False,
    required_target_style: str | None = None,
    simulation: SimulationSettings = DEFAULT_SIMULATION,
    excluded_hashes: set[str] | None = None,
    max_candidates: int = 200000,
    slot_offset: int = 0,
) -> list[ProtocolCase]:
    if system not in SYSTEM_TYPES:
        raise ValueError(f"Unsupported system: {system!r}")
    if count <= 0:
        raise ValueError("count must be positive")
    if required_target_style is not None and required_target_style not in STYLES:
        raise ValueError(f"Unsupported control style: {required_target_style!r}")
    excluded_hashes = excluded_hashes or set()
    dimensions = 4 if system == "first_order" else 5
    candidates = qmc.LatinHypercube(d=dimensions, seed=seed).random(max_candidates)
    accepted: list[ProtocolCase] = []
    accepted_hashes: set[str] = set()

    for slot in range(count):
        global_slot = slot_offset + slot
        fault_type = FAULT_TYPES[global_slot % len(FAULT_TYPES)]
        if "demonstration" in purpose:
            quota_slot = global_slot % len(SEVERITY_QUOTA)
            severity = SEVERITY_QUOTA[quota_slot]
            delay_band = DEMONSTRATION_DELAY_BY_FAULT[global_slot % len(FAULT_TYPES)]
        else:
            occurrence = (global_slot // len(FAULT_TYPES)) % 10
            severity_schedule = FAULT_SEVERITY_SCHEDULES.get(
                fault_type,
                ("mild",) * 4 + ("moderate",) * 4 + ("severe",) * 2,
            )
            severity = severity_schedule[occurrence]
            delay_band = DELAY_BY_SEVERITY_SLOT[occurrence]
        found: ProtocolCase | None = None

        for candidate_index in range(slot, max_candidates):
            values = candidates[candidate_index]
            plant = _plant_from_unit(system, values)
            delay = _delay_from_unit(plant, delay_band, values[-1])
            if delay is None:
                continue
            time_delay, _ = delay
            plant = _serialize_plant(plant)
            time_delay = _sig(time_delay, 4)
            rho = time_delay / control_time_constant(plant)
            reference_pid = imc_pid_for_style(plant, time_delay, "balanced")
            _, reference_metrics = simulate_protocol_case(
                plant, reference_pid, time_delay, simulation
            )
            if require_all_style_targets and not _all_style_targets_are_valid(
                plant, time_delay, simulation
            ):
                continue

            if required_target_style is not None and not _style_target_is_valid(
                plant, time_delay, required_target_style, simulation
            ):
                continue
            for perturbation_attempt, factors in enumerate(_perturbation_candidates(fault_type)):
                initial_pid = _serialize_pid(_apply_perturbation(reference_pid, factors))
                _, initial_metrics = simulate_protocol_case(
                    plant, initial_pid, time_delay, simulation
                )
                case = ProtocolCase(
                    case_id=f"{system}_{global_slot + 1:06d}",
                    system=system,
                    plant=plant,
                    time_delay=time_delay,
                    rho=rho,
                    initial_pid=initial_pid,
                    reference_pid=reference_pid,
                    fault_type=fault_type,
                    severity=severity,
                    perturbation=factors,
                    initial_metrics=initial_metrics,
                    reference_metrics=reference_metrics,
                    seed=seed,
                    candidate_index=candidate_index,
                    perturbation_attempt=perturbation_attempt,
                )
                if case.case_hash in excluded_hashes or case.case_hash in accepted_hashes:
                    continue
                if severity_from_iae_ratio(initial_metrics.iae / reference_metrics.iae) != severity:
                    continue
                if not initial_case_is_acceptable(initial_metrics, reference_metrics):
                    continue
                found = case
                break
            if found is not None:
                break

        if found is None:
            raise RuntimeError(
                f"Unable to construct {purpose} case {slot + 1}/{count} "
                f"for {system}, fault={fault_type}, severity={severity}, delay={delay_band}."
            )
        accepted.append(found)
        accepted_hashes.add(found.case_hash)

    return accepted


def protocol_manifest(
    cases: list[ProtocolCase],
    *,
    seed: int,
    purpose: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "purpose": purpose,
        "seed": seed,
        "count": len(cases),
        "simulation": asdict(DEFAULT_SIMULATION),
        "process_ranges": {
            "first_order": {"k": [0.2, 0.9], "t": [100.0, 600.0]},
            "second_order": {
                "k": [0.1, 3.0],
                "tau_slow": [10.0, 100.0],
                "tau_fast_ratio": [0.1, 1.0],
            },
        },
        "delay": {
            "absolute_seconds": [1.0, 200.0],
            "rho_max": 0.8,
            "bands": DELAY_BANDS,
            "quota_per_10": list(DELAY_QUOTA),
        },
        "fault_types": list(FAULT_TYPES),
        "severity_definition": {
            "mild": [1.5, 2.0],
            "moderate": [2.0, 5.0],
            "severe": [5.0, None],
            "authority": "initial_IAE / balanced_IMC_IAE",
        },
        "severity_counts": {
            level: sum(case.severity == level for case in cases)
            for level in ("mild", "moderate", "severe")
        },
        "fault_severity_schedules": {
            fault: list(schedule) for fault, schedule in FAULT_SEVERITY_SCHEDULES.items()
        },
        "case_hashes": [case.case_hash for case in cases],
    }


def _plant_from_unit(
    system: str,
    values: np.ndarray,
) -> FirstOrderPlant | SecondOrderPlant:
    if system == "first_order":
        return FirstOrderPlant(
            k=0.2 + float(values[0]) * 0.7,
            t=100.0 + float(values[1]) * 500.0,
        )
    tau_slow = 10.0 + float(values[1]) * 90.0
    ratio = 0.1 + float(values[2]) * 0.9
    return SecondOrderPlant(
        k=0.1 + float(values[0]) * 2.9,
        tau1=tau_slow * ratio,
        tau2=tau_slow,
    )


def _delay_from_unit(
    plant: FirstOrderPlant | SecondOrderPlant,
    band: str,
    unit_value: float,
) -> tuple[float, float] | None:
    low, high = DELAY_BANDS[band]
    tc = control_time_constant(plant)
    effective_low = max(low, 1.0 / tc)
    effective_high = min(high, 200.0 / tc, 0.8)
    if effective_low > effective_high:
        return None
    rho = effective_low + float(unit_value) * (effective_high - effective_low)
    return max(1.0, min(200.0, rho * tc)), rho


def _perturbation_candidates(fault_type: str) -> list[dict[str, float]]:
    """Return a deterministic strength sweep for one PID fault direction."""

    candidates: list[dict[str, float]] = []
    for strength in _geometric_strengths(1.01, 100.0, 160):
        high = _sig(float(strength), 15)
        low = _sig(1.0 / high, 15)
        factors = {"kp": 1.0, "ki": 1.0, "kd": 1.0}
        if fault_type == "p_high":
            factors["kp"] = high
        elif fault_type == "p_low":
            factors["kp"] = low
        elif fault_type == "i_high":
            factors["ki"] = high
        elif fault_type == "i_low_or_off":
            factors["ki"] = low
        elif fault_type == "i_high_d_high":
            factors.update(ki=high, kd=_sig(float(np.sqrt(high)), 15))
        elif fault_type == "p_high_d_low":
            factors.update(kp=high, kd=low)
        elif fault_type == "p_high_i_high":
            factors.update(kp=high, ki=high)
        elif fault_type == "p_low_i_high":
            factors.update(kp=low, ki=high)
        elif fault_type == "i_high_d_low":
            factors.update(ki=high, kd=low)
        elif fault_type == "p_low_i_high_d_low":
            factors.update(kp=low, ki=high, kd=low)
        else:
            raise ValueError(f"Unsupported PID fault type: {fault_type!r}")
        candidates.append({name: _sig(value, 15) for name, value in factors.items()})

    if fault_type in {"i_low_or_off", "p_high_d_low"}:
        off = dict(candidates[-1])
        off["ki" if fault_type == "i_low_or_off" else "kd"] = 0.0
        candidates.append(off)
    return candidates


def _geometric_strengths(start: float, stop: float, count: int) -> list[float]:
    """Generate a platform-independent geometric sequence for frozen assets."""

    if start <= 0 or stop <= 0:
        raise ValueError("Geometric sequence endpoints must be positive.")
    if count < 2:
        raise ValueError("Geometric sequence requires at least two values.")
    with localcontext() as context:
        context.prec = 50
        start_decimal = Decimal(str(start))
        stop_decimal = Decimal(str(stop))
        log_step = (stop_decimal.ln() - start_decimal.ln()) / Decimal(count - 1)
        values = [start_decimal * (log_step * Decimal(index)).exp() for index in range(count)]
        values[0] = start_decimal
        values[-1] = stop_decimal
    return [float(value) for value in values]


def _apply_perturbation(pid: PIDParams, factors: dict[str, float]) -> PIDParams:
    return PIDParams(
        kp=max(1e-12, pid.kp * factors["kp"]),
        ki=max(0.0, pid.ki * factors["ki"]),
        kd=max(0.0, pid.kd * factors["kd"]),
    )


def _all_style_targets_are_valid(
    plant: FirstOrderPlant | SecondOrderPlant,
    time_delay: float,
    simulation: SimulationSettings,
) -> bool:
    metrics = []
    for style in STYLES:
        pid = _serialize_pid(imc_pid_for_style(plant, time_delay, style))
        _, result = simulate_protocol_case(plant, pid, time_delay, simulation)
        if not result.converged():
            return False
        metrics.append(result)
    return metrics[0].settling_time <= metrics[1].settling_time <= metrics[2].settling_time


def _serialize_pid(pid: PIDParams) -> PIDParams:
    return PIDParams(_sig(pid.kp, 6), _sig(pid.ki, 6), _sig(pid.kd, 6))


def _style_target_is_valid(
    plant: FirstOrderPlant | SecondOrderPlant,
    time_delay: float,
    style: str,
    simulation: SimulationSettings,
) -> bool:
    pid = _serialize_pid(imc_pid_for_style(plant, time_delay, style))
    _, result = simulate_protocol_case(plant, pid, time_delay, simulation)
    return result.converged()


def _serialize_plant(
    plant: FirstOrderPlant | SecondOrderPlant,
) -> FirstOrderPlant | SecondOrderPlant:
    if isinstance(plant, FirstOrderPlant):
        return FirstOrderPlant(_sig(plant.k, 6), _sig(plant.t, 6))
    tau1, tau2 = sorted((_sig(plant.tau1, 6), _sig(plant.tau2, 6)))
    return SecondOrderPlant(_sig(plant.k, 6), tau1, tau2)


def _sig(value: float, digits: int) -> float:
    if value == 0:
        return 0.0
    return float(f"{float(value):.{digits}g}")
