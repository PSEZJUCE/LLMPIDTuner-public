from llmpidtuner.experiment_protocol import (
    _geometric_strengths,
    generate_protocol_cases,
    imc_pid_for_style,
    simulate_protocol_case,
)
from llmpidtuner.models import SimulationSettings


def test_geometric_strengths_are_frozen_for_protocol_assets():
    strengths = _geometric_strengths(1.01, 100.0, 160)

    assert len(strengths) == 160
    assert strengths[0] == 1.01
    assert strengths[132] == 45.825918319452036
    assert strengths[155] == 89.08287985823134
    assert strengths[-1] == 100.0


def test_required_target_style_filters_nonconvergent_sft_candidates():
    settings = SimulationSettings(
        setpoint=1.0,
        sim_time=4000.0,
        num_points=40001,
        max_abs_output=3.0,
    )
    index = 5537
    case = generate_protocol_cases(
        "second_order",
        1,
        1_081_001 + index * 1009,
        purpose="sft_training",
        required_target_style="aggressive",
        simulation=settings,
        slot_offset=index,
        max_candidates=5000,
    )[0]
    target_pid = imc_pid_for_style(case.plant, case.time_delay, "aggressive")
    _, target_metrics = simulate_protocol_case(
        case.plant,
        target_pid,
        case.time_delay,
        settings,
    )

    assert target_metrics.converged()
