from pathlib import Path

import pytest

from llmpidtuner.config import load_case_config
from llmpidtuner.demonstrations import generate_demonstration_from_spec
from llmpidtuner.experiment_protocol import PROTOCOL_ID


PROVIDERS = {
    "deepseek_v4_flash": "DS_Deepseek-V4-Flash",
    "qwen3_7_plus": "QWEN_qwen3.7-plus-2026-05-26",
}
PROMPTS = [
    ("", "balanced", "full"),
    ("_kpi3", "balanced", "kpi3"),
    ("_numeric8", "balanced", "numeric8"),
    ("_aggressive", "aggressive", "full"),
    ("_conservative", "conservative", "full"),
]


@pytest.mark.parametrize("system", ["first_order", "second_order"])
@pytest.mark.parametrize(("provider", "profile"), PROVIDERS.items())
@pytest.mark.parametrize(("suffix", "style", "variant"), PROMPTS)
def test_api_eval_case_matrix(
    system: str,
    provider: str,
    profile: str,
    suffix: str,
    style: str,
    variant: str,
) -> None:
    path = Path(f"cases/eval/{system}_100_{provider}{suffix}.yaml")
    config = load_case_config(path)

    assert config.name == path.stem
    assert config.system == f"{system}_batch"
    assert config.mode == "dry_run"
    assert config.llm_profile == profile
    assert config.control_style == style
    assert config.prompt_variant == variant
    assert config.max_iterations == 8
    assert config.simulation.sim_time == 4000
    assert config.simulation.num_points == 40001
    assert config.batch == {
        "cases_path": (
            "cases/protocol/perturbed_imc_delay_stratified/sources/"
            f"evaluation_{system}.yaml"
        )
    }
    assert config.demonstration is not None
    assert config.demonstration["protocol_id"] == PROTOCOL_ID
    assert config.demonstration["control_style"] == style
    assert config.demonstration["prompt_variant"] == variant
    text = generate_demonstration_from_spec(
        config.demonstration,
        initial_pid=config.initial_pid,
        simulation=config.simulation,
    )
    assert text is not None
    assert text.count("Experiment ") == 10


@pytest.mark.parametrize("system", ["first_order", "second_order"])
def test_imc_eval_case_uses_frozen_protocol_sources(system: str) -> None:
    config = load_case_config(Path(f"cases/eval/{system}_100_imc.yaml"))

    assert config.system == f"{system}_batch"
    assert config.mode == "imc"
    assert config.demonstration is None
    assert config.imc == {"style": "balanced"}
    assert config.batch == {
        "cases_path": (
            "cases/protocol/perturbed_imc_delay_stratified/sources/"
            f"evaluation_{system}.yaml"
        )
    }
