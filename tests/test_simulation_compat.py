import numpy as np

from llmpidtuner.models import FirstOrderPlant, PIDParams, SecondOrderPlant, SimulationSettings
from llmpidtuner.simulation import FirstOrderDelaySimulator, SecondOrderDelaySimulator


def test_protocol_grid_has_exact_point_one_second_step() -> None:
    settings = SimulationSettings()
    result = FirstOrderDelaySimulator(
        FirstOrderPlant(k=0.5, t=300.0),
        PIDParams(kp=1.0, ki=0.01, kd=0.1),
        settings,
    ).run()

    assert settings.sim_time == 4000.0
    assert settings.num_points == 40001
    assert len(result.time) == 40001
    assert np.isclose(result.time[1] - result.time[0], 0.1)


def test_fractional_delay_is_interpolated_and_derivative_kick_is_suppressed() -> None:
    settings = SimulationSettings(
        setpoint=1.0,
        sim_time=4.0,
        num_points=41,
        time_delay=1.05,
    )
    pid = PIDParams(kp=2.0, ki=0.0, kd=50.0)
    result = FirstOrderDelaySimulator(FirstOrderPlant(k=1.0, t=1.0), pid, settings).run()

    assert result.control_signal[0] == pid.kp
    assert np.all(result.output[:12] == 0.0)
    assert result.output[12] > 0.0


def test_fopdt_and_sopdt_share_finite_guard() -> None:
    settings = SimulationSettings(
        setpoint=1.0,
        sim_time=20.0,
        num_points=201,
        time_delay=1.0,
        max_abs_output=2.0,
    )
    pid = PIDParams(kp=100.0, ki=100.0, kd=100.0)
    first = FirstOrderDelaySimulator(FirstOrderPlant(1.0, 1.0), pid, settings).run()
    second = SecondOrderDelaySimulator(SecondOrderPlant(1.0, 1.0, 2.0), pid, settings).run()

    assert not first.finite
    assert not second.finite
