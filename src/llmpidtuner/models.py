from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PIDParams:
    kp: float
    ki: float
    kd: float


@dataclass(frozen=True)
class SimulationSettings:
    setpoint: float = 1.0
    sim_time: float = 4000.0
    num_points: int = 40001
    time_delay: float = 20.0
    max_abs_output: float | None = None


@dataclass(frozen=True)
class ResponseMetrics:
    iae: float
    overshoot_pct: float
    settling_time: float
    steady_state_error_pct: float
    oscillation_count: int
    time_to_63_2_after_delay: float
    final_value: float
    max_value: float
    min_value: float
    control_rms: float
    finite: bool
    settled: bool

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["time_constant"] = data.pop("time_to_63_2_after_delay")
        return {
            key: None if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in data.items()
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ResponseMetrics":
        def number(name: str, default: float = 0.0) -> float:
            value = data.get(name, default)
            return math.inf if value is None else float(value)

        time_63 = data.get("time_to_63_2_after_delay", data.get("time_constant", math.inf))
        return ResponseMetrics(
            iae=number("iae"),
            overshoot_pct=number("overshoot_pct"),
            settling_time=number("settling_time", math.inf),
            steady_state_error_pct=number("steady_state_error_pct"),
            oscillation_count=int(data.get("oscillation_count", 0)),
            time_to_63_2_after_delay=math.inf if time_63 is None else float(time_63),
            final_value=number("final_value"),
            max_value=number("max_value"),
            min_value=number("min_value"),
            control_rms=number("control_rms"),
            finite=bool(data.get("finite", True)),
            settled=bool(data.get("settled", False)),
        )

    @property
    def time_constant(self) -> float:
        """Compatibility alias for historical training artifacts."""

        return self.time_to_63_2_after_delay
    @property
    def overshoot(self) -> float:
        return self.overshoot_pct

    @property
    def steady_state_error(self) -> float:
        return self.steady_state_error_pct


    def converged(
        self,
        *,
        overshoot: float = 15.0,
        steady_state_error: float = 1.0,
    ) -> bool:
        return bool(
            self.finite
            and self.settled
            and self.overshoot_pct < overshoot
            and self.steady_state_error_pct < steady_state_error
        )

    @staticmethod
    def failed() -> "ResponseMetrics":
        return ResponseMetrics(
            iae=math.inf,
            overshoot_pct=math.inf,
            settling_time=math.inf,
            steady_state_error_pct=math.inf,
            oscillation_count=0,
            time_to_63_2_after_delay=math.inf,
            final_value=math.inf,
            max_value=math.inf,
            min_value=-math.inf,
            control_rms=math.inf,
            finite=False,
            settled=False,
        )


@dataclass(frozen=True)
class FirstOrderPlant:
    k: float
    t: float


@dataclass(frozen=True)
class SecondOrderPlant:
    k: float
    tau1: float
    tau2: float


@dataclass(frozen=True)
class PerformanceMetrics:
    overshoot: float
    steady_state_error: float
