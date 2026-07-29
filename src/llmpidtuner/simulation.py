from __future__ import annotations

import math
from typing import Any
from dataclasses import dataclass

import numpy as np

from llmpidtuner.models import FirstOrderPlant, PIDParams, SecondOrderPlant, SimulationSettings


@dataclass
class SimulationResult:
    time: np.ndarray
    output: np.ndarray
    control_signal: np.ndarray
    errors: np.ndarray
    results_array: np.ndarray
    iae: float

    finite: bool = True


def _should_stop(settings: SimulationSettings, output: float, error: float, control: float) -> bool:
    if not all(np.isfinite(value) for value in (output, error, control)):
        return True
    return settings.max_abs_output is not None and abs(output) > settings.max_abs_output

def _delayed_control(
    control: np.ndarray,
    *,
    known_through: int,
    delay_steps: float,
) -> float:
    """Return u(t-delay) using linear interpolation on the sampled control history."""

    target = float(known_through) - delay_steps
    if target < 0.0:
        return 0.0
    lower = int(np.floor(target))
    upper = min(lower + 1, known_through)
    fraction = target - lower
    return float((1.0 - fraction) * control[lower] + fraction * control[upper])


def _pid_update(
    pid: PIDParams,
    *,
    error: float,
    previous_error: float,
    integral: float,
    dt: float,
) -> tuple[float, float]:
    integral += error * dt
    derivative = (error - previous_error) / dt
    control = pid.kp * error + pid.ki * integral + pid.kd * derivative
    return float(control), float(integral)
try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised in minimal runtime installs
    def njit(*_args: object, **_kwargs: object):
        def decorator(function: Any) -> Any:
            return function

        return decorator


@njit(cache=True)
def _simulate_fopdt_kernel(
    k: float,
    t_constant: float,
    kp: float,
    ki: float,
    kd: float,
    setpoint: float,
    sim_time: float,
    num_points: int,
    time_delay: float,
    max_abs_output: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    dt = sim_time / (num_points - 1)
    delay_steps = time_delay / dt
    output = np.zeros(num_points)
    control = np.zeros(num_points)
    errors = np.zeros(num_points)
    errors[0] = setpoint
    control[0] = kp * errors[0]
    previous_error = errors[0]
    integral = 0.0
    finite = True

    for index in range(1, num_points):
        target = float(index - 1) - delay_steps
        delayed = 0.0
        if target >= 0.0:
            lower = int(math.floor(target))
            upper = min(lower + 1, index - 1)
            fraction = target - lower
            delayed = (1.0 - fraction) * control[lower] + fraction * control[upper]

        output[index] = output[index - 1] + (
            k * delayed - output[index - 1]
        ) * dt / t_constant
        error = setpoint - output[index]
        errors[index] = error
        integral += error * dt
        derivative = (error - previous_error) / dt
        control[index] = kp * error + ki * integral + kd * derivative

        invalid = (
            not math.isfinite(output[index])
            or not math.isfinite(error)
            or not math.isfinite(control[index])
            or (max_abs_output >= 0.0 and abs(output[index]) > max_abs_output)
        )
        if invalid:
            previous = max(0, index - 1)
            output[index:] = output[previous]
            control[index:] = control[previous]
            errors[index:] = errors[previous]
            finite = False
            break
        previous_error = error
    return output, control, errors, finite


@njit(cache=True)
def _simulate_sopdt_kernel(
    k: float,
    tau1: float,
    tau2: float,
    kp: float,
    ki: float,
    kd: float,
    setpoint: float,
    sim_time: float,
    num_points: int,
    time_delay: float,
    max_abs_output: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    dt = sim_time / (num_points - 1)
    delay_steps = time_delay / dt
    output = np.zeros(num_points)
    control = np.zeros(num_points)
    errors = np.zeros(num_points)
    errors[0] = setpoint
    control[0] = kp * errors[0]
    previous_error = errors[0]
    integral = 0.0
    x1 = 0.0
    x2 = 0.0
    finite = True

    for index in range(1, num_points):
        target = float(index - 1) - delay_steps
        delayed = 0.0
        if target >= 0.0:
            lower = int(math.floor(target))
            upper = min(lower + 1, index - 1)
            fraction = target - lower
            delayed = (1.0 - fraction) * control[lower] + fraction * control[upper]

        x1 += (k * delayed - x1) * dt / tau1
        x2 += (x1 - x2) * dt / tau2
        output[index] = x2
        error = setpoint - output[index]
        errors[index] = error
        integral += error * dt
        derivative = (error - previous_error) / dt
        control[index] = kp * error + ki * integral + kd * derivative

        invalid = (
            not math.isfinite(output[index])
            or not math.isfinite(error)
            or not math.isfinite(control[index])
            or (max_abs_output >= 0.0 and abs(output[index]) > max_abs_output)
        )
        if invalid:
            previous = max(0, index - 1)
            output[index:] = output[previous]
            control[index:] = control[previous]
            errors[index:] = errors[previous]
            finite = False
            break
        previous_error = error
    return output, control, errors, finite



def _freeze_trajectory(
    output: np.ndarray,
    control_signal: np.ndarray,
    errors: np.ndarray,
    index: int,
) -> None:
    previous = max(0, index - 1)
    output[index:] = output[previous]
    control_signal[index:] = control_signal[previous]
    errors[index:] = errors[previous]


class FirstOrderDelaySimulator:
    """Discrete FOPDT simulation with linearly interpolated dead time."""

    def __init__(
        self,
        plant: FirstOrderPlant,
        pid: PIDParams,
        settings: SimulationSettings,
    ) -> None:
        self.plant = plant
        self.pid = pid
        self.settings = settings
        self.dt = settings.sim_time / (settings.num_points - 1)
        self.time = np.linspace(0, settings.sim_time, settings.num_points)
        self.output = np.zeros(settings.num_points)
        self.control_signal = np.zeros(settings.num_points)
        self.errors = np.zeros(settings.num_points)
        self.errors[0] = settings.setpoint - self.output[0]
        self.integral = 0.0
        self.prev_error = self.errors[0]
        self.control_signal[0] = self.pid.kp * self.errors[0]
        self.delay_steps = settings.time_delay / self.dt

    def _first_order_system(self, u_delayed: float, y_prev: float) -> float:
        return y_prev + (self.plant.k * u_delayed - y_prev) * self.dt / self.plant.t


    def run(self) -> SimulationResult:
        self.output, self.control_signal, self.errors, finite = _simulate_fopdt_kernel(
            self.plant.k,
            self.plant.t,
            self.pid.kp,
            self.pid.ki,
            self.pid.kd,
            self.settings.setpoint,
            self.settings.sim_time,
            self.settings.num_points,
            self.settings.time_delay,
            -1.0
            if self.settings.max_abs_output is None
            else self.settings.max_abs_output,
        )
        results_array = _create_results_array(
            self.time, self.output, self.settings.setpoint, self.settings.num_points, self.settings.sim_time
        )
        iae = float(np.sum(np.abs(self.errors)) * self.dt)
        return SimulationResult(
            time=self.time,
            output=self.output,
            control_signal=self.control_signal,
            errors=self.errors,
            results_array=results_array,
            finite=finite,
            iae=iae,
        )


class SecondOrderDelaySimulator:
    """Discrete SOPDT simulation using the same PID and delay protocol as FOPDT."""

    def __init__(
        self,
        plant: SecondOrderPlant,
        pid: PIDParams,
        settings: SimulationSettings,
    ) -> None:
        self.plant = plant
        self.pid = pid
        self.settings = settings
        self.dt = settings.sim_time / (settings.num_points - 1)
        self.time = np.linspace(0, settings.sim_time, settings.num_points)
        self.output = np.zeros(settings.num_points)
        self.control_signal = np.zeros(settings.num_points)
        self.errors = np.zeros(settings.num_points)
        self.errors[0] = settings.setpoint - self.output[0]
        self.integral = 0.0
        self.prev_error = self.errors[0]
        self.control_signal[0] = self.pid.kp * self.errors[0]
        self.delay_steps = settings.time_delay / self.dt


    def run(self) -> SimulationResult:
        self.output, self.control_signal, self.errors, finite = _simulate_sopdt_kernel(
            self.plant.k,
            self.plant.tau1,
            self.plant.tau2,
            self.pid.kp,
            self.pid.ki,
            self.pid.kd,
            self.settings.setpoint,
            self.settings.sim_time,
            self.settings.num_points,
            self.settings.time_delay,
            -1.0
            if self.settings.max_abs_output is None
            else self.settings.max_abs_output,
        )
        results_array = _create_results_array(
            self.time, self.output, self.settings.setpoint, self.settings.num_points, self.settings.sim_time
        )
        iae = float(np.sum(np.abs(self.errors)) * self.dt)
        return SimulationResult(
            time=self.time,
            output=self.output,
            control_signal=self.control_signal,
            errors=self.errors,
            results_array=results_array,
            iae=iae,
            finite=finite,
        )


def _create_results_array(
    time: np.ndarray,
    output: np.ndarray,
    setpoint: float,
    num_points: int,
    sim_time: float,
) -> np.ndarray:
    results_array: list[list[float]] = []
    step = max(1, int(num_points / sim_time))
    for i in range(0, num_points, step):
        if i >= num_points:
            i = num_points - 1
        results_array.append([float(time[i]), float(setpoint), float(output[i])])

    if (results_array[-1][0] + 1e-6) < sim_time:
        results_array.append([float(sim_time), float(setpoint), float(output[-1])])
    return np.array(results_array)
