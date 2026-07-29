from pathlib import Path

import pytest

from llmpidtuner.config import load_case_config
from llmpidtuner.experiment_protocol import PROTOCOL_ID
from llmpidtuner.llm import LLMSettings
from llmpidtuner.runner import _write_llm_metadata


EVAL_CASES = [
    ("first_order", "full"),
    ("second_order", "full"),
    ("first_order", "kpi3"),
    ("second_order", "kpi3"),
    ("first_order", "numeric8"),
    ("second_order", "numeric8"),
]


def _case_path(system: str, variant: str) -> Path:
    suffix = "" if variant == "full" else f"_{variant}"
    return Path(f"cases/eval/{system}_100_base_qwen3_0p6b{suffix}.yaml")


@pytest.mark.parametrize(("system", "variant"), EVAL_CASES)
def test_base_prompt_ablation_case_matrix(system: str, variant: str):
    path = _case_path(system, variant)
    config = load_case_config(path)

    assert config.name == path.stem
    assert config.system == f"{system}_batch"
    assert config.llm_profile == "vLLM_qwen3-0.6b-base"
    assert config.prompt_variant == variant
    assert config.llm == {
        "temperature": 0.1,
        "top_p": 0.1,
        "enable_thinking": False,
        "max_tokens": 64,
        "seed": 42,
        "max_retries": 5,
    }
    assert config.batch == {
        "cases_path": (
            "cases/protocol/perturbed_imc_delay_stratified/sources/"
            f"evaluation_{system}.yaml"
        )
    }
    assert config.max_iterations == 8
    assert config.simulation.sim_time == 4000
    assert config.simulation.num_points == 40001
    assert config.demonstration["method"] == "frozen"
    assert config.demonstration["protocol_id"] == PROTOCOL_ID
    assert config.demonstration["control_style"] == "balanced"
    assert config.demonstration["prompt_variant"] == variant
    assert Path(config.demonstration["path"]).is_file()
    assert Path(config.demonstration["manifest_path"]).is_file()


def test_llm_metadata_records_request_seed(tmp_path):
    settings = LLMSettings(
        provider="vllm",
        profile="vLLM_qwen3-0.6b-base",
        model="qwen3-0.6b-base",
        enable_thinking=False,
        seed=42,
    )

    _write_llm_metadata(tmp_path, settings, "kpi3")

    metadata = (tmp_path / "llm_metadata.txt").read_text(encoding="utf-8")
    expected_body = '{"chat_template_kwargs": {"enable_thinking": false}}'
    assert f"thinking_request_body={expected_body}" in metadata
    assert "seed=42" in metadata
    assert "prompt_variant=kpi3" in metadata


def test_base_prompt_ablation_slurm_entry_covers_matrix_and_resume_modes():
    script = Path(
        "scripts/slurm/eval_qwen3_0p6b_base_prompt_ablation.sbatch"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --gres=gpu:1" in script
    assert 'WORKERS="${WORKERS:-10}"' in script
    assert 'EVAL_SCOPE="${EVAL_SCOPE:-all}"' in script
    assert "full-first-order)" in script
    assert "run-api-parallel" in script
    assert "--resume" in script
    assert "Resume protocol preflight passed." in script
    assert '"thinking_request_body": (' in script
    for system, variant in EVAL_CASES:
        assert _case_path(system, variant).as_posix() in script
