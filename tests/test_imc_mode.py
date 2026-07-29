from pathlib import Path

from llmpidtuner.config import CaseConfig, load_case_config
from llmpidtuner.models import FirstOrderPlant, PIDParams, SimulationSettings
from llmpidtuner.runner import run_case


def test_load_imc_case_config():
    config = load_case_config("cases/eval/first_order_100_imc.yaml")

    assert config.name == "first_order_100_imc"
    assert config.system == "first_order_batch"
    assert config.mode == "imc"
    assert config.llm_profile is None
    assert config.demonstration is None
    assert config.imc == {"style": "balanced"}
    assert config.batch == {
        "cases_path": (
            "cases/protocol/perturbed_imc_delay_stratified/sources/"
            "evaluation_first_order.yaml"
        )
    }


def test_imc_mode_writes_deterministic_result_without_prompt_files(tmp_path: Path):
    config = CaseConfig(
        name="tiny_imc",
        system="first_order",
        mode="imc",
        output_dir=str(tmp_path),
        initial_pid=PIDParams(kp=1.0, ki=0.1, kd=0.01),
        simulation=SimulationSettings(setpoint=1.0, sim_time=20.0, num_points=201, time_delay=1.0),
        first_order=FirstOrderPlant(k=0.65, t=112.0),
        imc={"lambda_value": 10},
    )

    output_dir = run_case(config)

    assert (output_dir / "parameter_PID_IAE.txt").exists()
    assert (output_dir / "value_curve.txt").exists()
    assert (output_dir / "parameter_PID_IAE_iteration_1.txt").exists()
    assert (output_dir / "value_curve_iteration_1.txt").exists()
    assert (output_dir / "pid_tuning_comparison.png").exists()
    assert (output_dir / "imc_metadata.txt").exists()
    assert not (output_dir / "initial_prompt.txt").exists()
    assert not (output_dir / "demonstration_prompt.txt").exists()
