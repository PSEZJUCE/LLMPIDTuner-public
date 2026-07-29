from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from llmpidtuner.models import PerformanceMetrics, ResponseMetrics


PROMPT_VARIANTS = frozenset({"full", "kpi3", "numeric8"})


def normalize_prompt_variant(value: str | None) -> str:
    variant = str(value or "full").strip().lower()
    if variant not in PROMPT_VARIANTS:
        choices = ", ".join(sorted(PROMPT_VARIANTS))
        raise ValueError(f"Unsupported prompt_variant {value!r}; expected one of: {choices}.")
    return variant


@dataclass(frozen=True)
class ResponseFeatures:
    """One structured response diagnosis shared by every prompt renderer."""

    overshoot_pct: float
    overshoot_description: str
    oscillation_count: int
    oscillation_description: str
    attenuation_ratio: float
    oscillation_period_s: float
    time_to_63_2_s: float | None
    settling_time_s: float | None
    steady_state_error_pct: float
    steady_state_error_description: str


def calculate_performance_metrics(output: np.ndarray, setpoint: float) -> PerformanceMetrics:
    values = np.asarray(output, dtype=np.float64)
    scale = max(abs(float(setpoint)), 1e-12)
    overshoot = max(0.0, (float(np.max(values)) - setpoint) / scale * 100.0)
    tail = values[-min(100, len(values)) :]
    steady_state_error = abs(float(np.mean(tail)) - setpoint) / scale * 100.0
    return PerformanceMetrics(float(overshoot), float(steady_state_error))


def calculate_response_metrics(
    time: np.ndarray,
    output: np.ndarray,
    control: np.ndarray,
    errors: np.ndarray,
    iae: float,
    *,
    setpoint: float = 1.0,
    time_delay: float = 0.0,
    finite: bool = True,
) -> ResponseMetrics:
    """Calculate the canonical metrics used by evaluation, SFT, and GRPO."""

    time = np.asarray(time, dtype=np.float64)
    output = np.asarray(output, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    finite = bool(
        finite
        and len(output)
        and np.all(np.isfinite(output))
        and np.all(np.isfinite(control))
        and np.all(np.isfinite(errors))
        and math.isfinite(float(iae))
    )
    if not finite:
        return ResponseMetrics.failed()

    scale = max(abs(float(setpoint)), 1e-12)
    max_value = float(np.max(output))
    min_value = float(np.min(output))
    final_value = float(output[-1])
    overshoot = max(0.0, (max_value - setpoint) / scale * 100.0)

    tail = output[-min(100, len(output)) :]
    steady_state_error = abs(float(np.mean(tail)) - setpoint) / scale * 100.0

    tolerance = 0.05 * scale
    outside = np.abs(output - setpoint) > tolerance
    settling_time = math.inf
    if not outside[-1]:
        indices = np.flatnonzero(outside)
        settling_index = int(indices[-1] + 1) if len(indices) else 0
        if settling_index < len(time):
            settling_time = float(time[settling_index])

    time_to_63 = math.inf
    after_delay = np.flatnonzero(time >= time_delay)
    if len(after_delay):
        candidate = after_delay[
            output[after_delay] >= 0.632 * setpoint
            if setpoint >= 0
            else output[after_delay] <= 0.632 * setpoint
        ]
        if len(candidate):
            time_to_63 = max(0.0, float(time[int(candidate[0])] - time_delay))

    return ResponseMetrics(
        iae=float(iae),
        overshoot_pct=float(overshoot),
        settling_time=float(settling_time),
        steady_state_error_pct=float(steady_state_error),
        oscillation_count=_count_setpoint_crossings(output, setpoint, tolerance),
        time_to_63_2_after_delay=float(time_to_63),
        final_value=final_value,
        max_value=max_value,
        min_value=min_value,
        control_rms=float(np.sqrt(np.mean(np.square(control)))),
        finite=True,
        settled=math.isfinite(settling_time),
    )


def _count_setpoint_crossings(output: np.ndarray, setpoint: float, tolerance: float) -> int:
    centered = output - setpoint
    significant = centered[np.abs(centered) > tolerance]
    if len(significant) < 2:
        return 0
    signs = np.sign(significant)
    return int(np.count_nonzero(signs[1:] != signs[:-1]))


class ControlSystemAnalysis:
    """Extract and render curve diagnostics using the canonical metric definitions."""

    def __init__(self, curve_file: str, time_delay: float = 0.0) -> None:
        self.filename = curve_file
        self.df = pd.read_csv(curve_file, sep=" ", skiprows=2, names=["Time", "Setpoint", "Output"])
        self.time = self.df["Time"]
        self.setpoint = self.df["Setpoint"]
        self.output = self.df["Output"]
        self.set_value = self.setpoint.iloc[-1]
        self.overshoot = 0.0
        self.time_delay = float(time_delay)

    @classmethod
    def from_arrays(
        cls,
        time: np.ndarray,
        setpoint: np.ndarray,
        output: np.ndarray,
        filename: str = "<array>",
        time_delay: float = 0.0,
    ) -> "ControlSystemAnalysis":
        instance = cls.__new__(cls)
        instance.filename = filename
        instance.df = pd.DataFrame({"Time": time, "Setpoint": setpoint, "Output": output})
        instance.time = instance.df["Time"]
        instance.setpoint = instance.df["Setpoint"]
        instance.output = instance.df["Output"]
        instance.set_value = instance.setpoint.iloc[-1]
        instance.overshoot = 0.0
        instance.time_delay = float(time_delay)
        return instance

    def calculate_overshoot(self) -> str:
        max_value = self.output.max()
        self.overshoot = max(0.0, (max_value - self.set_value) / abs(self.set_value) * 100) if self.set_value != 0 else 0
        if self.overshoot <= 15:
            return "The overshoot is normal."
        if 16 <= self.overshoot <= 30:
            return "The overshoot is relatively large."
        if 31 <= self.overshoot < 100:
            return "The overshoot is severe."
        return "The overshoot is extremely severe, reaching 100% or more."

    def calculate_oscillation_count(self) -> int:
        tolerance = 0.05 * max(abs(float(self.set_value)), 1e-12)
        return _count_setpoint_crossings(self.output.to_numpy(), float(self.set_value), tolerance)

    @staticmethod
    def describe_oscillations(damped_oscillations: int) -> str:
        if damped_oscillations <= 1:
            return "The oscillation behavior is normal."
        if 2 <= damped_oscillations <= 3:
            return "The curve shows noticeable oscillations."
        return "Oscillation is very violent and the curve fluctuates greatly."

    def calculate_oscillations(self) -> str:
        return self.describe_oscillations(self.calculate_oscillation_count())

    def calculate_attenuation_and_period(self) -> tuple[float, float]:
        scale = max(abs(float(self.set_value)), 1e-12)
        peaks, _ = find_peaks(
            self.output.to_numpy(), prominence=0.01 * scale
        )
        if len(peaks) > 1:
            first_deviation = self.output.iloc[peaks[0]] - self.set_value
            second_deviation = self.output.iloc[peaks[1]] - self.set_value
            denominator = max(abs(float(first_deviation)), 1e-12)
            attenuation_ratio = abs(float(second_deviation)) / denominator
            oscillation_period = self.time.iloc[peaks[1]] - self.time.iloc[peaks[0]]
        else:
            attenuation_ratio = 0
            oscillation_period = 0
        return float(attenuation_ratio), float(oscillation_period)

    def calculate_time_constant(self) -> float | None:
        for t, o in zip(self.time, self.output):
            if t < self.time_delay:
                continue
            if o >= 0.632 * self.set_value:
                return max(0.0, float(t) - self.time_delay)
        return None

    def calculate_settling_time(self) -> float | None:
        for i in range(len(self.output)):
            if abs(self.output.iloc[i] - self.set_value) <= 0.05 * abs(self.set_value):
                remaining_output = self.output.iloc[i:]
                if all(abs(remaining_output - self.set_value) <= 0.05 * abs(self.set_value)):
                    return float(self.time.iloc[i])
        return None

    def calculate_steady_state_error(self) -> float:
        steady_state_values = self.output[-min(100, len(self.output)) :]
        steady_state_average = steady_state_values.mean()
        if self.set_value == 0:
            return 0.0
        return float(abs(steady_state_average - self.set_value) / abs(self.set_value) * 100)

    def extract_features(self) -> ResponseFeatures:
        overshoot_description = self.calculate_overshoot()
        oscillation_count = self.calculate_oscillation_count()
        oscillation_description = self.describe_oscillations(oscillation_count)
        attenuation_ratio, oscillation_period = self.calculate_attenuation_and_period()
        time_constant = self.calculate_time_constant()
        settling_time = self.calculate_settling_time()
        steady_state_error = self.calculate_steady_state_error()
        if steady_state_error <= 1:
            error_description = (
                "The control performance is good; the Steady-State Error is within 1%, "
                f"its value is {steady_state_error:.4g}%."
            )
        elif 1 < steady_state_error <= 5:
            error_description = (
                "In the final stage, the curve did not converge to the steady-state value; "
                f"there is a residual error of {steady_state_error:.4g}%."
            )
        else:
            error_description = (
                "The system has a significant deviation from the steady-state value; "
                f"there is a large residual error of {steady_state_error:.4g}%."
            )

        return ResponseFeatures(
            overshoot_pct=float(self.overshoot),
            overshoot_description=overshoot_description,
            oscillation_count=oscillation_count,
            oscillation_description=oscillation_description,
            attenuation_ratio=attenuation_ratio,
            oscillation_period_s=oscillation_period,
            time_to_63_2_s=time_constant,
            settling_time_s=settling_time,
            steady_state_error_pct=steady_state_error,
            steady_state_error_description=error_description,
        )

    def generate_description(self, prompt_variant: str = "full") -> str:
        return render_response_features(self.extract_features(), prompt_variant)


def render_response_features(features: ResponseFeatures, prompt_variant: str = "full") -> str:
    variant = normalize_prompt_variant(prompt_variant)
    overshoot_line = (
        "Overshoot: After reaching the peak value, the curve shows an overshoot of "
        f"{features.overshoot_pct:.4g}%, indicating {features.overshoot_description}"
    )

    if variant == "kpi3":
        return f"\n{overshoot_line}\nSteady-State Error: {features.steady_state_error_description}\n"

    if variant == "numeric8":
        time_constant = _format_optional_metric(features.time_to_63_2_s)
        settling_time = _format_optional_metric(features.settling_time_s)
        return f"""
Response KPI vector:
Overshoot (%) = {features.overshoot_pct:.4g}
Oscillation count = {features.oscillation_count}
Attenuation ratio = {features.attenuation_ratio:.4g}
Oscillation period (s) = {features.oscillation_period_s:.4g}
Time to 63.2% of setpoint (s) = {time_constant}
Settling time (+/-5%) (s) = {settling_time}
Steady-state error (%) = {features.steady_state_error_pct:.4g}
"""

    if features.time_to_63_2_s is not None:
        time_constant_description = (
            f"The time to 63.2% after the observed dead time is {features.time_to_63_2_s:.4g} seconds, "
            "measured from the end of the observed dead-time interval."
        )
    else:
        time_constant_description = "The system response is very slow or almost non-responsive."

    if features.settling_time_s is not None:
        settling_description = (
            f"The settling time (tc) is {features.settling_time_s:.4g} seconds, "
            "indicating the time when the output enters within +/-5% of the steady-state "
            "value and does not leave again."
        )
    else:
        settling_description = (
            "The system output does not settle within +/-5% of the steady-state value."
        )

    return f"""
{overshoot_line}
Oscillations: The curve contains {features.oscillation_count} oscillations. {features.oscillation_description}
Attenuation Ratio: The attenuation ratio between the first and second peak deviations is {features.attenuation_ratio:.4g}.
Oscillation Period: The period of oscillation is {features.oscillation_period_s:.4g} seconds.
{time_constant_description}
{settling_description}
{features.steady_state_error_description}
"""


def _format_optional_metric(value: float | None) -> str:
    return "NaN" if value is None else f"{value:.4g}"
