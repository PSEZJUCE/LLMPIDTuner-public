from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from llmpidtuner.training.config import GRPOTrainConfig
from llmpidtuner.training.grpo import _load_resume_training_state
from llmpidtuner.training.grpo_control import (
    AdaptiveKLController,
    learning_rate_at_step,
    resolve_resume_checkpoint,
    resume_compatibility_payload,
    validation_selection_key,
)


def _config(tmp_path: Path | None = None) -> GRPOTrainConfig:
    output_dir = "outputs/grpo/test" if tmp_path is None else str(tmp_path / "grpo")
    return GRPOTrainConfig(
        model_name_or_path="outputs/sft/test",
        output_dir=output_dir,
    )


def test_learning_rate_uses_warmup_cosine_and_minimum_floor() -> None:
    config = _config()

    assert learning_rate_at_step(config, 0) == 0.0
    assert learning_rate_at_step(config, 25) == pytest.approx(5.0e-6)
    assert learning_rate_at_step(config, 50) == pytest.approx(1.0e-5)
    assert config.min_learning_rate < learning_rate_at_step(config, 400) < config.learning_rate
    assert learning_rate_at_step(config, 800) == pytest.approx(1.0e-6)
    assert learning_rate_at_step(config, 1200) == pytest.approx(1.0e-6)


def test_adaptive_kl_updates_beta_and_triggers_guard() -> None:
    controller = AdaptiveKLController.from_config(_config())

    for step in range(1, 11):
        controller.update(0.20, step)
    assert controller.beta == pytest.approx(0.15)

    for step in range(11, 36):
        controller.update(0.30, step)
    assert controller.guard_triggered
    assert controller.ema is not None and controller.ema > 0.20


def test_validation_selection_is_pass_rate_first() -> None:
    high_pass = {
        "success_rate": 0.90,
        "iae_improvement_mean": 0.10,
        "reward_mean": 0.60,
        "kl_ema": 0.10,
    }
    low_pass_high_reward = {
        "success_rate": 0.80,
        "iae_improvement_mean": 0.90,
        "reward_mean": 0.99,
        "kl_ema": 0.01,
    }

    assert validation_selection_key(high_pass) > validation_selection_key(low_pass_high_reward)


def test_resume_payload_allows_only_cumulative_target_and_log_save_frequencies() -> None:
    config = _config()
    allowed = replace(
        config,
        train_steps=1200,
        resume_from_checkpoint="auto",
        save_every=100,
        log_every=5,
        save_rollouts_every=50,
    )
    changed_training = replace(config, learning_rate=2.0e-5)

    assert resume_compatibility_payload(config) == resume_compatibility_payload(allowed)
    assert resume_compatibility_payload(config) != resume_compatibility_payload(changed_training)


def test_resolve_resume_checkpoint_auto_uses_latest_complete_state(tmp_path: Path) -> None:
    config = replace(_config(tmp_path), resume_from_checkpoint="auto")
    checkpoints = Path(config.output_dir) / "checkpoints"
    for step in (50, 100):
        checkpoint = checkpoints / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"step": step}),
            encoding="utf-8",
        )

    assert resolve_resume_checkpoint(config) == checkpoints / "checkpoint-100"


def test_resume_state_accepts_json_normalized_config_and_allowed_changes(tmp_path: Path) -> None:
    config = replace(
        _config(tmp_path),
        excluded_plants_paths=("first.yaml", "second.yaml"),
    )
    resumed = replace(
        config,
        train_steps=1200,
        resume_from_checkpoint="auto",
        save_every=100,
    )
    checkpoint = Path(config.output_dir) / "checkpoints" / "checkpoint-800"
    (checkpoint / "accelerator_state").mkdir(parents=True)
    (checkpoint / "rank_000_state.json").write_text(
        json.dumps({"prompt_provider_state": {"seed": 91001}}),
        encoding="utf-8",
    )
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(
            {
                "world_size": 5,
                "config_compatibility": resume_compatibility_payload(config),
            }
        ),
        encoding="utf-8",
    )

    state = _load_resume_training_state(checkpoint, resumed, rank=0, world_size=5)

    assert state["prompt_provider_state"] == {"seed": 91001}
    with pytest.raises(ValueError, match="world size"):
        _load_resume_training_state(checkpoint, resumed, rank=0, world_size=1)
    with pytest.raises(ValueError, match="configuration differs"):
        _load_resume_training_state(
            checkpoint,
            replace(resumed, learning_rate=2.0e-5),
            rank=0,
            world_size=5,
        )
