from llmpidtuner.config import load_case_config


def test_load_first_order_batch_case() -> None:
    config = load_case_config("cases/eval/first_order_100_deepseek_v4_flash.yaml")

    assert config.name == "first_order_100_deepseek_v4_flash"
    assert config.system == "first_order_batch"
    assert config.mode == "dry_run"
    assert config.llm_profile == "DS_Deepseek-V4-Flash"
    assert config.batch is not None
    assert config.batch["cases_path"].endswith("evaluation_first_order.yaml")
    assert config.control_style == "balanced"


def test_load_second_order_batch_case() -> None:
    config = load_case_config("cases/eval/second_order_100_deepseek_v4_flash.yaml")

    assert config.system == "second_order_batch"
    assert config.batch is not None
    assert config.batch["cases_path"].endswith("evaluation_second_order.yaml")
    assert config.simulation.sim_time == 4000.0
    assert config.simulation.num_points == 40001
    assert config.llm == {
        "temperature": 0.1,
        "top_p": 0.1,
        "enable_thinking": False,
        "max_tokens": 64,
        "max_retries": 5,
    }
