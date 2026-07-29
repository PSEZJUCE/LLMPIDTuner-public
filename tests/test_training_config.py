from pathlib import Path

import pytest

from llmpidtuner.training.artifacts import sha256_file, write_json_atomic
from llmpidtuner.training.config import (
    load_grpo_train_config,
    load_sft_train_config,
    load_training_data_config,
)


def test_canonical_training_configs_use_new_protocol() -> None:
    data = load_training_data_config("configs/data/pid_sft_messages_40k.yaml")
    sft = load_sft_train_config("configs/sft/qwen3_0p6b_pid.yaml")
    grpo = load_grpo_train_config("configs/grpo/qwen3_0p6b_pid.yaml")

    assert (data.first_order_count, data.second_order_count) == (20000, 20000)
    assert data.workers == 32
    assert len(data.excluded_plants_paths) == 6
    assert data.control_style == sft.control_style == grpo.control_style == "balanced"
    assert grpo.lora_dropout == 0.0
    assert grpo.num_generations == 4
    assert grpo.train_steps == 800
    assert len(grpo.validation_cases_paths) == 2
    assert grpo.validate_every == 50
    assert grpo.learning_rate == 1.0e-5
    assert grpo.min_learning_rate == 1.0e-6
    assert grpo.beta_kl == 0.10
    assert grpo.target_kl == 0.05
    assert grpo.reward.stability_weight == 0.35


def test_grpo_config_rejects_nonzero_lora_dropout(tmp_path: Path) -> None:
    path = tmp_path / "grpo.yaml"
    path.write_text(
        "model_name_or_path: model\noutput_dir: output\nlora_dropout: 0.05\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="lora_dropout must be 0"):
        load_grpo_train_config(path)


def test_grpo_config_rejects_nonbalanced_control_style(tmp_path: Path) -> None:
    path = tmp_path / "grpo.yaml"
    path.write_text(
        "model_name_or_path: model\noutput_dir: output\ncontrol_style: aggressive\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="calibrated for control_style=balanced"):
        load_grpo_train_config(path)


def test_sft_config_requires_aligned_save_and_eval_steps(tmp_path: Path) -> None:
    path = tmp_path / "sft.yaml"
    path.write_text(
        "model_name_or_path: model\n"
        "dataset_path: data.jsonl\n"
        "output_dir: output\n"
        "save_steps: 50\n"
        "eval_steps: 30\n"
        "load_best_model_at_end: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="save_steps must be a multiple"):
        load_sft_train_config(path)


def test_artifact_hash_and_atomic_json_write(tmp_path: Path) -> None:
    data_path = tmp_path / "data.txt"
    data_path.write_text("pid-data", encoding="utf-8")
    manifest_path = write_json_atomic(
        tmp_path / "manifest.json",
        {"sha256": sha256_file(data_path)},
    )
    assert manifest_path.read_text(encoding="utf-8").startswith("{")
    assert sha256_file(data_path) == (
        "6f669cd7d3bc23e50010a1dc8fd79eaee679d37fcdace185b3bb7757b3ed457c"
    )
