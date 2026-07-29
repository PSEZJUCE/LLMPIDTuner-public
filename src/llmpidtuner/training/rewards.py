from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from llmpidtuner.models import PIDParams, SimulationSettings
from llmpidtuner.training.data import PromptSample
from llmpidtuner.training.simulation import PlantSpec, ResponseMetrics, simulate_pid


NUMBER_PATTERN = r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?"


@dataclass
class RewardResult:
    reward: float
    parsed_pid: PIDParams | None
    metrics: ResponseMetrics | None
    components: dict[str, float]
    weights: dict[str, float]
    analysis: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "reward": float(self.reward),
            "parsed_pid": None
            if self.parsed_pid is None
            else {"kp": self.parsed_pid.kp, "ki": self.parsed_pid.ki, "kd": self.parsed_pid.kd},
            "metrics": None if self.metrics is None else self.metrics.as_dict(),
            "components": {key: float(value) for key, value in self.components.items()},
            "weights": {key: float(value) for key, value in self.weights.items()},
            "analysis": self.analysis,
        }


def parse_pid(text: str) -> PIDParams | None:
    if not text:
        return None
    normalized = (
        text.replace("；", ";").replace("，", ",").replace("：", ":").replace("\u3000", " ").strip()
    )
    kp = _find_named_value(normalized, ("Kp", "K_p", "P"))
    ki = _find_named_value(normalized, ("Ki", "K_i", "I"))
    kd = _find_named_value(normalized, ("Kd", "K_d", "D"))
    if kp is not None and ki is not None and kd is not None:
        pid = PIDParams(kp, ki, kd)
        return pid if all(math.isfinite(x) for x in (pid.kp, pid.ki, pid.kd)) else None

    numbers = re.findall(NUMBER_PATTERN, normalized)
    if 3 <= len(numbers) <= 5 and len(normalized) <= 120:
        try:
            pid = PIDParams(float(numbers[0]), float(numbers[1]), float(numbers[2]))
        except ValueError:
            return None
        return pid if all(math.isfinite(x) for x in (pid.kp, pid.ki, pid.kd)) else None
    return None


def format_compliance_score(text: str) -> float:
    if parse_pid(text) is None:
        return 0.0
    normalized = text.strip().replace("；", ";").replace("，", ",").replace("：", ":")
    canonical_patterns = [
        rf"^\s*Kp\s*=\s*{NUMBER_PATTERN}\s*,\s*Ki\s*=\s*{NUMBER_PATTERN}\s*,\s*Kd\s*=\s*{NUMBER_PATTERN}\s*$",
        rf"^\s*P\s*[:=]\s*{NUMBER_PATTERN}\s*;\s*I\s*[:=]\s*{NUMBER_PATTERN}\s*;\s*D\s*[:=]\s*{NUMBER_PATTERN}\s*$",
    ]
    if any(re.match(pattern, normalized, flags=re.IGNORECASE) for pattern in canonical_patterns):
        return 1.0
    return 0.7 if len(normalized) <= 120 else 0.4


def normalize_group_advantages(
    rewards: Sequence[float],
    group_size: int,
    eps: float = 1e-6,
) -> list[float]:
    if group_size <= 0:
        raise ValueError("group_size must be positive.")
    if len(rewards) % group_size != 0:
        raise ValueError("reward count must be a multiple of group_size.")
    values = np.asarray(rewards, dtype=np.float64).reshape(-1, group_size)
    means = values.mean(axis=1, keepdims=True)
    stds = values.std(axis=1, keepdims=True)
    advantages = np.where(stds > eps, (values - means) / np.maximum(stds, eps), 0.0)
    return advantages.reshape(-1).astype(np.float32).tolist()


def group_reward_stats(rewards: Sequence[float], group_size: int) -> list[dict[str, float]]:
    values = np.asarray(rewards, dtype=np.float64).reshape(-1, group_size)
    return [
        {
            "mean": float(row.mean()),
            "std": float(row.std()),
            "min": float(row.min()),
            "max": float(row.max()),
        }
        for row in values
    ]


def _controller_phase_radians(pid: PIDParams, omega: float) -> float:
    return float(np.arctan2(pid.kd * omega - pid.ki / omega, pid.kp))


def _loop_phase_radians(pid: PIDParams, plant: PlantSpec, omega: float) -> float:
    controller_phase = _controller_phase_radians(pid, omega)
    delay_phase = -omega * plant.time_delay
    if plant.plant_type == "second_order":
        return (
            controller_phase
            - np.arctan(omega * float(plant.tau1 or 1.0))
            - np.arctan(omega * float(plant.tau2 or 1.0))
            + delay_phase
        )
    return controller_phase - np.arctan(omega * float(plant.t or 1.0)) + delay_phase


def performance_reward(
    overshoot: float,
    settling_time: float,
    steady_state_error: float,
    settling_reference: float,
) -> float:
    if overshoot <= 5.0:
        overshoot_reward = 1.0
    elif overshoot <= 15.0:
        overshoot_reward = 1.0 - (overshoot - 5.0) / 10.0
    elif overshoot <= 30.0:
        overshoot_reward = -0.5 * (overshoot - 15.0) / 15.0
    else:
        overshoot_reward = -0.5

    settling_ratio = settling_time / max(settling_reference, 1e-12)
    settling_reward = (
        1.0
        if settling_ratio <= 1.0
        else 1.0 - 1.3 * (settling_ratio - 1.0)
        if settling_ratio <= 2.0
        else -0.3
    )
    ss_reward = (
        1.0
        if steady_state_error <= 0.1
        else 1.0 - (steady_state_error - 0.1) / 0.9
        if steady_state_error <= 1.0
        else -0.5 * (steady_state_error - 1.0)
        if steady_state_error <= 2.0
        else -0.5
    )
    return float(
        np.clip(
            (5.0 / 12.0) * overshoot_reward + (1.0 / 3.0) * settling_reward + 0.25 * ss_reward,
            -1.0,
            1.0,
        )
    )


def robustness_reward(pid: PIDParams, plant: PlantSpec) -> float:
    sensitivity = abs(pid.kd) / (abs(pid.kp) + 1e-8) + abs(pid.ki) / (abs(pid.kp) + 1e-8)
    robust = (
        1.0
        if 0.1 <= sensitivity <= 2.0
        else 1.0 - (sensitivity - 2.0) / 3.0
        if sensitivity <= 5.0
        else -0.5
    )
    gain_robustness = 1.0 / (1.0 + abs(plant.k * pid.kp - 1.0))
    return float(np.clip(0.7 * robust + 0.3 * gain_robustness, -1.0, 1.0))


def _find_named_value(text: str, names: tuple[str, ...]) -> float | None:
    joined = "|".join(re.escape(name) for name in names)
    match = re.search(
        rf"(?<![A-Za-z0-9_])(?:{joined})(?![A-Za-z0-9_])\s*[:=]\s*({NUMBER_PATTERN})",
        text,
        re.IGNORECASE,
    )
    return None if match is None else float(match.group(1))


def _performance_metrics(
    baseline: ResponseMetrics,
    metrics: ResponseMetrics,
    simulation: SimulationSettings,
    plant: PlantSpec,
) -> dict[str, float | bool]:
    if baseline.iae and math.isfinite(baseline.iae) and baseline.iae > 0:
        iae_improvement = (baseline.iae - metrics.iae) / max(baseline.iae, 1e-8)
    else:
        iae_improvement = 0.0
    settling_time = (
        metrics.settling_time if math.isfinite(metrics.settling_time) else simulation.sim_time
    )
    tc = _characteristic_time(plant)
    lambda_balanced = 2.0 * max(plant.time_delay, 0.1 * tc)
    settling_reference = plant.time_delay - math.log(0.05) * lambda_balanced
    return {
        "iae_improvement": float(iae_improvement),
        "overshoot": float(metrics.overshoot_pct),
        "settling_time": float(settling_time),
        "settling_reference": float(settling_reference),
        "steady_state_error": float(metrics.steady_state_error_pct),
        "task_success": metrics.converged(),
    }


# Revised Routh-Hurwitz / exact-delay reward implementation.
@dataclass(frozen=True)
class RewardConfig:
    schema_version: int = 2
    invalid_reward: float = -1.0
    min_reward: float = -1.0
    max_reward: float = 1.0
    failed_branch_min: float = -0.5
    failed_branch_max: float = 0.0
    success_branch_min: float = 0.5
    success_branch_max: float = 1.0
    stability_weight: float = 0.35
    performance_weight: float = 0.35
    format_weight: float = 0.05
    iae_weight: float = 0.15
    regularization_weight: float = 0.10
    pade_order: int = 1
    routh_relative_tolerance: float = 1.0e-9
    controller_zero_tolerance: float = 1.0e-12
    phase_log10_min: float = -5.0
    phase_log10_max: float = 5.0
    phase_grid_points: int = 2048
    phase_bisection_iterations: int = 64
    gain_reference_imc_lambda: float = 10.0
    gain_calibration_seed: int = 3407
    gain_calibration_samples: int = 1024
    gain_calibration_quantile: float = 0.05
    gain_floor_min: float = 0.01
    gain_multiplier_p: float = 4.0
    gain_multiplier_i: float = 4.0
    gain_multiplier_d: float = 6.0


@dataclass(frozen=True)
class GainReference:
    p: float
    i: float
    d: float
    seed: int
    samples: int
    quantile: float
    floor_min: float
    imc_lambda: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "p": self.p,
            "i": self.i,
            "d": self.d,
            "seed": self.seed,
            "samples": self.samples,
            "quantile": self.quantile,
            "floor_min": self.floor_min,
            "imc_lambda": self.imc_lambda,
        }


def _basic_pid_constraints(pid: PIDParams) -> bool:
    return bool(
        math.isfinite(pid.kp)
        and math.isfinite(pid.ki)
        and math.isfinite(pid.kd)
        and pid.kp > 0.0
        and pid.ki >= 0.0
        and pid.kd >= 0.0
    )


def _characteristic_time(plant: PlantSpec) -> float:
    if plant.plant_type == "second_order":
        return max(float(plant.tau1 or 1.0) + float(plant.tau2 or 1.0), 1e-12)
    return max(float(plant.t or 1.0), 1e-12)


def _dimensionless_gains(pid: PIDParams, plant: PlantSpec) -> tuple[float, float, float]:
    tc = _characteristic_time(plant)
    return plant.k * pid.kp, plant.k * pid.ki * tc, plant.k * pid.kd / tc


def calibrate_gain_reference(
    config: RewardConfig, *, second_order_prob: float, time_delay: float, setpoint: float
) -> GainReference:
    """Calibrate deterministic IMC reference floors without simulations."""
    from llmpidtuner.training.simulation import imc_pid_for_plant, sample_plant

    rng = np.random.default_rng(config.gain_calibration_seed)
    values = []
    for _ in range(max(1, config.gain_calibration_samples)):
        plant = sample_plant(
            rng, second_order_prob=second_order_prob, time_delay=time_delay, setpoint=setpoint
        )
        values.append(
            _dimensionless_gains(
                imc_pid_for_plant(plant, lambda_value=config.gain_reference_imc_lambda), plant
            )
        )
    floors = np.maximum(
        np.quantile(np.asarray(values, dtype=np.float64), config.gain_calibration_quantile, axis=0),
        max(config.gain_floor_min, 1e-12),
    )
    return GainReference(
        p=float(floors[0]),
        i=float(floors[1]),
        d=float(floors[2]),
        seed=config.gain_calibration_seed,
        samples=config.gain_calibration_samples,
        quantile=config.gain_calibration_quantile,
        floor_min=config.gain_floor_min,
        imc_lambda=config.gain_reference_imc_lambda,
    )


def characteristic_polynomial(
    pid: PIDParams, plant: PlantSpec, config: RewardConfig
) -> tuple[list[float], dict[str, Any]]:
    """Use exact zero-delay dynamics and [1/1] Pade only for delayed Routh tests."""
    if config.pade_order != 1:
        raise ValueError("Only first-order [1/1] Pade is supported.")
    if plant.k <= 0.0:
        raise ValueError("Routh safety analysis requires positive process gain.")
    k, ki_active = plant.k, pid.ki > config.controller_zero_tolerance
    delayed = plant.time_delay > config.controller_zero_tolerance

    if plant.plant_type == "first_order":
        t = max(float(plant.t or 1.0), config.controller_zero_tolerance)
        if delayed:
            d = plant.time_delay / 2.0
            coeffs = (
                [
                    t * d - k * d * pid.kd,
                    t + d + k * (pid.kd - d * pid.kp),
                    1.0 + k * pid.kp - k * d * pid.ki,
                    k * pid.ki,
                ]
                if ki_active
                else [t * d - k * d * pid.kd, t + d + k * (pid.kd - d * pid.kp), 1.0 + k * pid.kp]
            )
        else:
            coeffs = (
                [t + k * pid.kd, 1.0 + k * pid.kp, k * pid.ki]
                if ki_active
                else [t + k * pid.kd, 1.0 + k * pid.kp]
            )
    elif plant.plant_type == "second_order":
        t1 = max(float(plant.tau1 or 1.0), config.controller_zero_tolerance)
        t2 = max(float(plant.tau2 or 1.0), config.controller_zero_tolerance)
        b2, b1 = t1 * t2, t1 + t2
        if delayed:
            d = plant.time_delay / 2.0
            coeffs = (
                [
                    b2 * d,
                    b2 + b1 * d - k * d * pid.kd,
                    b1 + d + k * (pid.kd - d * pid.kp),
                    1.0 + k * (pid.kp - d * pid.ki),
                    k * pid.ki,
                ]
                if ki_active
                else [
                    b2 * d,
                    b2 + b1 * d - k * d * pid.kd,
                    b1 + d + k * (pid.kd - d * pid.kp),
                    1.0 + k * pid.kp,
                ]
            )
        else:
            coeffs = (
                [b2, b1 + k * pid.kd, 1.0 + k * pid.kp, k * pid.ki]
                if ki_active
                else [b2, b1 + k * pid.kd, 1.0 + k * pid.kp]
            )
    else:
        raise ValueError(f"Unsupported plant type: {plant.plant_type}")
    return [float(value) for value in coeffs], {
        "plant_type": plant.plant_type,
        "controller_structure": "PID"
        if ki_active
        else ("PD" if pid.kd > config.controller_zero_tolerance else "P"),
        "delay_model": "pade_1_1" if delayed else "exact_no_delay",
        "degree": len(coeffs) - 1,
    }


def _routh_result(
    status: str, reason: str | None, table: np.ndarray, tolerance: float, coefficients: np.ndarray
) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "tolerance": float(tolerance),
        "normalized_coefficients": [float(value) for value in coefficients],
        "first_column": [float(value) for value in table[:, 0]],
        "table": [[float(value) for value in row] for row in table],
    }


def routh_hurwitz(coefficients: Sequence[float], relative_tolerance: float) -> dict[str, Any]:
    coeffs = np.asarray(coefficients, dtype=np.float64)
    if coeffs.ndim != 1 or len(coeffs) < 2 or not np.all(np.isfinite(coeffs)):
        return {"status": "indeterminate", "reason": "invalid_coefficients", "first_column": []}
    raw_tolerance = max(0.0, relative_tolerance) * max(1.0, float(np.max(np.abs(coeffs))))
    if coeffs[0] <= raw_tolerance:
        return {
            "status": "marginal" if abs(coeffs[0]) <= raw_tolerance else "unstable",
            "reason": "nonpositive_or_nearzero_leading_coefficient",
            "tolerance": float(raw_tolerance),
            "first_column": [float(coeffs[0])],
        }

    normalized = coeffs / coeffs[0]
    tolerance = max(0.0, relative_tolerance) * max(1.0, float(np.max(np.abs(normalized))))
    degree, cols = len(normalized) - 1, (len(normalized) + 1) // 2
    table = np.zeros((degree + 1, cols), dtype=np.float64)
    table[0, : len(normalized[::2])] = normalized[::2]
    table[1, : len(normalized[1::2])] = normalized[1::2]

    for row in range(2):
        if np.all(np.abs(table[row]) <= tolerance):
            return _routh_result("indeterminate", "all_zero_row", table, tolerance, normalized)
        if abs(table[row, 0]) <= tolerance:
            return _routh_result("marginal", "zero_first_column", table, tolerance, normalized)
    for row in range(2, degree + 1):
        previous, two_above = table[row - 1], table[row - 2]
        if np.all(np.abs(previous) <= tolerance):
            return _routh_result("indeterminate", "all_zero_row", table, tolerance, normalized)
        if abs(previous[0]) <= tolerance:
            return _routh_result("marginal", "zero_first_column", table, tolerance, normalized)
        table[row, :-1] = (previous[0] * two_above[1:] - two_above[0] * previous[1:]) / previous[0]
        if not np.all(np.isfinite(table[row])):
            return _routh_result(
                "indeterminate", "nonfinite_routh_table", table, tolerance, normalized
            )
        if np.all(np.abs(table[row]) <= tolerance):
            return _routh_result("indeterminate", "all_zero_row", table, tolerance, normalized)
        if abs(table[row, 0]) <= tolerance:
            return _routh_result("marginal", "zero_first_column", table, tolerance, normalized)

    status = "stable" if np.all(table[:, 0] > tolerance) else "unstable"
    return _routh_result(
        status,
        None if status == "stable" else "first_column_sign_change",
        table,
        tolerance,
        normalized,
    )


def _log_loop_magnitude(pid: PIDParams, plant: PlantSpec, omega: float) -> float:
    if omega <= 0.0 or plant.k <= 0.0:
        return math.nan
    controller = math.hypot(pid.kp, pid.kd * omega - pid.ki / omega)
    if plant.plant_type == "second_order":
        t1, t2 = float(plant.tau1 or 1.0), float(plant.tau2 or 1.0)
        process = plant.k / math.sqrt((1.0 + (omega * t1) ** 2) * (1.0 + (omega * t2) ** 2))
    else:
        process = plant.k / math.sqrt(1.0 + (omega * float(plant.t or 1.0)) ** 2)
    return math.log(controller * process)


def _refine_crossover(
    pid: PIDParams, plant: PlantSpec, left: float, right: float, iterations: int
) -> float:
    left_value = _log_loop_magnitude(pid, plant, math.exp(left))
    for _ in range(max(1, iterations)):
        middle = (left + right) / 2.0
        middle_value = _log_loop_magnitude(pid, plant, math.exp(middle))
        if abs(middle_value) <= 1e-12:
            return math.exp(middle)
        if (left_value <= 0.0 <= middle_value) or (middle_value <= 0.0 <= left_value):
            right = middle
        else:
            left, left_value = middle, middle_value
    return math.exp((left + right) / 2.0)


def _open_loop_phase_margin(
    pid: PIDParams, plant: PlantSpec, config: RewardConfig | None = None
) -> tuple[float | None, float | None, dict[str, Any]]:
    """Minimum phase margin for exact process delay at all refined crossovers."""
    config = config or RewardConfig()
    tc = _characteristic_time(plant)
    lower = 10.0**config.phase_log10_min / tc
    upper = 10.0**config.phase_log10_max / tc
    omega = np.logspace(math.log10(lower), math.log10(upper), max(32, config.phase_grid_points))
    values = np.asarray([_log_loop_magnitude(pid, plant, float(w)) for w in omega])
    finite = np.isfinite(values)
    crossed = np.flatnonzero(
        finite[:-1] & finite[1:] & ((values[:-1] == 0.0) | (values[:-1] * values[1:] < 0.0))
    )
    base = {
        "frequency_range": [float(lower), float(upper)],
        "grid_points": int(len(omega)),
        "crossover_count": int(len(crossed)),
        "delay_model": "exact",
    }
    if not len(crossed):
        return (
            None,
            None,
            {**base, "reason": "No unity-gain crossover in the adaptive search range."},
        )

    margins = []
    for index in crossed:
        wc = _refine_crossover(
            pid,
            plant,
            math.log(float(omega[index])),
            math.log(float(omega[index + 1])),
            config.phase_bisection_iterations,
        )
        phase = _loop_phase_radians(pid, plant, wc)
        margins.append((180.0 + float(np.degrees(phase)), wc, float(np.degrees(phase))))
    margin, crossover, phase = min(margins, key=lambda item: item[0])
    return (
        margin,
        crossover,
        {
            **base,
            "all_crossovers": [
                {"phase_margin_deg": float(pm), "frequency": float(wc), "loop_phase_deg": float(p)}
                for pm, wc, p in margins
            ],
            "controller_phase_deg": float(np.degrees(_controller_phase_radians(pid, crossover))),
            "loop_phase_deg": phase,
        },
    )


def gain_regularization_reward(
    pid: PIDParams,
    plant: PlantSpec,
    config: RewardConfig,
    reference: GainReference | None,
) -> tuple[float, dict[str, Any]]:
    del reference
    from llmpidtuner.training.simulation import imc_pid_for_plant

    imc = imc_pid_for_plant(plant, control_style="balanced")
    refs = np.maximum(
        np.asarray(_dimensionless_gains(imc, plant)),
        config.gain_floor_min,
    )
    lambda_balanced = 2.0 * max(
        plant.time_delay,
        0.1 * _characteristic_time(plant),
    )
    plant_reference = GainReference(
        p=float(refs[0]),
        i=float(refs[1]),
        d=float(refs[2]),
        seed=0,
        samples=1,
        quantile=0.0,
        floor_min=config.gain_floor_min,
        imc_lambda=lambda_balanced,
    )
    gains = _dimensionless_gains(pid, plant)
    limits = (
        config.gain_multiplier_p * plant_reference.p,
        config.gain_multiplier_i * plant_reference.i,
        config.gain_multiplier_d * plant_reference.d,
    )
    excess = [
        max(0.0, math.log((gain + 1e-12) / (limit + 1e-12)))
        for gain, limit in zip(gains, limits, strict=True)
    ]
    return -min(1.0, sum(value * value for value in excess)), {
        "dimensionless_gains": {"p": gains[0], "i": gains[1], "d": gains[2]},
        "reference": plant_reference.as_dict(),
        "limits": {"p": limits[0], "i": limits[1], "d": limits[2]},
        "log_excess": {"p": excess[0], "i": excess[1], "d": excess[2]},
    }


def _weights_from_config(config: RewardConfig) -> dict[str, float]:
    raw = {
        "stability": config.stability_weight,
        "performance": config.performance_weight,
        "format": config.format_weight,
        "iae": config.iae_weight,
        "regularization": config.regularization_weight,
    }
    total = sum(max(0.0, float(value)) for value in raw.values())
    if total <= 0.0:
        raise ValueError("At least one reward weight must be positive.")
    return {key: max(0.0, float(value)) / total for key, value in raw.items()}


def stability_proxy(
    pid: PIDParams, plant: PlantSpec, config: RewardConfig | None = None
) -> tuple[float, bool, dict[str, Any]]:
    config = config or RewardConfig()
    if not _basic_pid_constraints(pid):
        return -1.0, False, {"safety_reason": "invalid_pid_sign"}
    try:
        coeffs, characteristic = characteristic_polynomial(pid, plant, config)
        routh = routh_hurwitz(coeffs, config.routh_relative_tolerance)
        phase_margin, crossover, phase_analysis = _open_loop_phase_margin(pid, plant, config)
        pm_safe = phase_margin is None or phase_margin > 0.0
        safe = routh["status"] == "stable" and pm_safe
        if not safe:
            reward, reason = (
                -1.0,
                "phase_margin_nonpositive" if not pm_safe else f"routh_{routh['status']}",
            )
        elif phase_margin is None:
            reward, reason = 0.0, None
        elif phase_margin > 45.0:
            reward, reason = 1.0, None
        elif phase_margin > 30.0:
            reward, reason = 0.5 + (phase_margin - 30.0) / 30.0, None
        else:
            reward, reason = phase_margin / 60.0, None
        return (
            float(reward),
            bool(safe),
            {
                "safety_reason": reason,
                "pade_order": config.pade_order,
                "characteristic": characteristic,
                "characteristic_coeffs": coeffs,
                "routh": routh,
                "phase_margin": phase_margin,
                "gain_crossover_frequency": crossover,
                "phase_analysis": phase_analysis,
            },
        )
    except Exception as exc:
        return -1.0, False, {"safety_reason": "analysis_error", "error": str(exc)}


def comprehensive_pid_reward(
    pid: PIDParams,
    plant: PlantSpec,
    prev_iae: float,
    new_iae: float,
    performance_metrics: dict[str, float | bool],
    format_score: float,
    config: RewardConfig,
    reference: GainReference | None = None,
) -> tuple[float, dict[str, Any]]:
    weights = _weights_from_config(config)
    stability_reward, safe, stability_analysis = stability_proxy(pid, plant, config)
    performance = performance_reward(
        performance_metrics["overshoot"],
        performance_metrics["settling_time"],
        performance_metrics["steady_state_error"],
        performance_metrics["settling_reference"],
    )
    iae = math.tanh(((prev_iae - new_iae) / max(abs(prev_iae), 1e-8)) * 2.0)
    regularization, regularization_analysis = gain_regularization_reward(
        pid, plant, config, reference
    )
    robustness = robustness_reward(pid, plant)
    quality = float(
        np.clip(
            weights["stability"] * stability_reward
            + weights["performance"] * performance
            + weights["format"] * format_score
            + weights["iae"] * iae
            + weights["regularization"] * regularization,
            -1.0,
            1.0,
        )
    )
    task_success = bool(performance_metrics["task_success"])
    if not safe:
        total = config.invalid_reward
        reward_branch = "unsafe"
    else:
        branch_min, branch_max, reward_branch = (
            (config.success_branch_min, config.success_branch_max, "success")
            if task_success
            else (config.failed_branch_min, config.failed_branch_max, "stable_failed")
        )
        total = branch_min + 0.5 * (quality + 1.0) * (branch_max - branch_min)
    return float(np.clip(total, config.min_reward, config.max_reward)), {
        "is_safe": safe,
        "task_success": task_success,
        "reward_branch": reward_branch,
        "quality_score": quality,
        "stability_reward": stability_reward,
        "performance_reward": performance,
        "format_reward": format_score,
        "iae_reward": iae,
        "regularization_reward": regularization,
        "robustness_reward": robustness,
        "weights_used": weights,
        "stability_analysis": stability_analysis,
        "regularization_analysis": regularization_analysis,
    }


def _safety_failure_result(
    config: RewardConfig,
    weights: dict[str, float],
    parsed: PIDParams | None,
    metrics: ResponseMetrics | None,
    format_score: float,
    error: str,
    details: dict[str, Any],
) -> RewardResult:
    return RewardResult(
        reward=config.invalid_reward,
        parsed_pid=parsed,
        metrics=metrics,
        weights=weights,
        components={
            "stability": -1.0,
            "performance": -1.0,
            "format": format_score,
            "iae": -1.0,
            "regularization": -1.0,
            "robustness": -1.0,
        },
        analysis={"error": error, **details},
    )


def evaluate_completion(
    completion: str,
    sample: PromptSample,
    simulation: SimulationSettings,
    reward_config: RewardConfig,
    gain_reference: GainReference | None = None,
    *,
    completion_hit_length_limit: bool = False,
) -> RewardResult:
    parsed = parse_pid(completion)
    weights = _weights_from_config(reward_config)
    format_score = 0.0 if completion_hit_length_limit else format_compliance_score(completion)
    completion_details = {
        "completion": completion[:240],
        "completion_hit_length_limit": completion_hit_length_limit,
    }
    if parsed is None:
        return _safety_failure_result(
            reward_config,
            weights,
            None,
            None,
            format_score,
            "Could not parse PID gains from completion.",
            completion_details,
        )
    if not _basic_pid_constraints(parsed):
        return _safety_failure_result(
            reward_config,
            weights,
            parsed,
            None,
            format_score,
            "PID gains must satisfy Kp > 0, Ki >= 0, and Kd >= 0.",
            completion_details,
        )

    simulated = simulate_pid(sample.plant, parsed, simulation)
    if not simulated.metrics.finite:
        return _safety_failure_result(
            reward_config,
            weights,
            parsed,
            simulated.metrics,
            format_score,
            "Simulation became non-finite or exceeded max_abs_output.",
            {
                "baseline_metrics": sample.current_metrics.as_dict(),
                "candidate_metrics": simulated.metrics.as_dict(),
                **completion_details,
            },
        )
    performance_metrics = _performance_metrics(
        sample.current_metrics, simulated.metrics, simulation, sample.plant
    )
    reward, analysis = comprehensive_pid_reward(
        parsed,
        sample.plant,
        sample.current_metrics.iae,
        simulated.metrics.iae,
        performance_metrics,
        format_score,
        reward_config,
        gain_reference,
    )
    analysis.update(
        {
            "baseline_metrics": sample.current_metrics.as_dict(),
            "candidate_metrics": simulated.metrics.as_dict(),
            "performance_metrics": performance_metrics,
            **completion_details,
        }
    )
    return RewardResult(
        reward=reward,
        parsed_pid=parsed,
        metrics=simulated.metrics,
        components={
            "stability": float(analysis["stability_reward"]),
            "performance": float(analysis["performance_reward"]),
            "format": float(analysis["format_reward"]),
            "iae": float(analysis["iae_reward"]),
            "regularization": float(analysis["regularization_reward"]),
            "robustness": float(analysis["robustness_reward"]),
        },
        weights={key: float(value) for key, value in analysis["weights_used"].items()},
        analysis=analysis,
    )
