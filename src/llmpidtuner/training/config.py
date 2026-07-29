from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from llmpidtuner.models import PIDParams, SimulationSettings
from llmpidtuner.training.rewards import RewardConfig


@dataclass(frozen=True)
class TrainingDataConfig:
    output_path: str
    control_style: str = "balanced"
    count: int = 1000
    first_order_count: int | None = None
    second_order_count: int | None = None
    seed: int = 42
    second_order_prob: float = 0.5
    initial_pid: PIDParams = field(default_factory=lambda: PIDParams(1.0, 0.1, 0.01))
    simulation: SimulationSettings = field(
        default_factory=lambda: SimulationSettings(time_delay=1.0)
    )
    lambda_value: float = 10.0
    excluded_plants_paths: tuple[str, ...] = ()
    format: str = "openai_messages"
    workers: int = 32
    demonstration: dict[str, Any] | None = None
    feedback_sample_probability: float = 0.5
    include_target_metrics: bool = True


@dataclass(frozen=True)
class SFTTrainConfig:
    model_name_or_path: str
    dataset_path: str
    output_dir: str
    control_style: str = "balanced"
    max_length: int = 6144
    learning_rate: float = 1.0e-5
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    bf16: bool = True
    save_steps: int = 50
    eval_steps: int = 50
    logging_steps: int = 10
    validation_fraction: float = 0.05
    seed: int = 42
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    gradient_checkpointing: bool = False


@dataclass(frozen=True)
class GRPOTrainConfig:
    model_name_or_path: str
    output_dir: str
    control_style: str = "balanced"
    prompt_data_path: str | None = None
    seed: int = 42
    train_steps: int = 800
    prompts_per_step: int = 4
    num_generations: int = 4
    num_policy_epochs: int = 1
    micro_batch_size: int = 4
    learning_rate: float = 1.0e-5
    min_learning_rate: float = 1.0e-6
    warmup_steps: int = 50
    lr_decay_steps: int = 800
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    max_prompt_length: int = 6144
    max_completion_length: int = 64
    temperature: float = 0.9
    top_p: float = 0.95
    top_k: int = 50
    clip_range: float = 0.2
    beta_kl: float = 0.10
    adaptive_kl: bool = True
    target_kl: float = 0.05
    kl_tolerance: float = 2.0
    beta_kl_min: float = 0.02
    beta_kl_max: float = 1.0
    kl_ema_alpha: float = 0.10
    kl_update_interval: int = 10
    kl_beta_multiplier: float = 1.5
    kl_guard_multiplier: float = 4.0
    kl_guard_patience: int = 25
    bf16: bool = True
    use_qlora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    second_order_prob: float = 0.5
    excluded_plants_paths: tuple[str, ...] = ()
    demonstration: dict[str, Any] | None = None
    simulation: SimulationSettings = field(
        default_factory=lambda: SimulationSettings(time_delay=1.0)
    )
    reward: RewardConfig = field(default_factory=RewardConfig)
    save_every: int = 50
    log_every: int = 1
    save_rollouts_every: int = 20
    validation_cases_paths: tuple[str, ...] = ()
    validate_every: int = 50
    validation_batch_size: int = 8
    resume_from_checkpoint: str | None = None


@dataclass(frozen=True)
class ServerJobConfig:
    job_name: str
    command: str
    output_dir: str
    partition: str = "cluster"
    gres: str = "gpu:1"
    cpus_per_task: int = 8
    mem: str = "32G"
    time: str = "04:00:00"
    workdir: str | None = None
    setup_commands: list[str] = field(default_factory=list)


def load_training_data_config(path: str | Path) -> TrainingDataConfig:
    data = _load_yaml(path)
    first_order_count = _optional_int(data.get("first_order_count"))
    second_order_count = _optional_int(data.get("second_order_count"))
    if (first_order_count is None) != (second_order_count is None):
        raise ValueError("first_order_count and second_order_count must be provided together.")
    if first_order_count is not None:
        assert second_order_count is not None
        if first_order_count < 0 or second_order_count < 0:
            raise ValueError("Plant-type sample counts must be non-negative.")
        if "count" in data and int(data["count"]) != first_order_count + second_order_count:
            raise ValueError(
                "count must equal first_order_count + second_order_count when both are set."
            )
    control_style = _control_style(data.get("control_style", "balanced"))
    demonstration = data.get("demonstration")
    if (
        demonstration
        and _control_style(demonstration.get("control_style", "balanced")) != control_style
    ):
        raise ValueError("SFT data and demonstration control styles must match.")

    return TrainingDataConfig(
        output_path=str(data["output_path"]),
        control_style=control_style,
        count=int(data.get("count", 1000)),
        first_order_count=first_order_count,
        second_order_count=second_order_count,
        seed=int(data.get("seed", 42)),
        second_order_prob=float(data.get("second_order_prob", 0.5)),
        initial_pid=_pid_from_dict(data.get("initial_pid", {})),
        workers=int(data.get("workers", 32)),
        simulation=_simulation_from_dict(data.get("simulation", {})),
        lambda_value=float(data.get("lambda_value", 10.0)),
        excluded_plants_paths=tuple(str(path) for path in data.get("excluded_plants_paths", [])),
        format=str(data.get("format", "openai_messages")),
        demonstration=demonstration,
        feedback_sample_probability=float(data.get("feedback_sample_probability", 0.5)),
        include_target_metrics=bool(data.get("include_target_metrics", True)),
    )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def load_sft_train_config(path: str | Path) -> SFTTrainConfig:
    data = _load_yaml(path)
    config = SFTTrainConfig(
        model_name_or_path=str(data["model_name_or_path"]),
        dataset_path=str(data["dataset_path"]),
        output_dir=str(data["output_dir"]),
        control_style=_control_style(data.get("control_style", "balanced")),
        max_length=int(data.get("max_length", 6144)),
        learning_rate=float(data.get("learning_rate", 1.0e-5)),
        num_train_epochs=float(data.get("num_train_epochs", 3.0)),
        per_device_train_batch_size=int(data.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(data.get("gradient_accumulation_steps", 8)),
        bf16=bool(data.get("bf16", True)),
        save_steps=int(data.get("save_steps", 50)),
        eval_steps=int(data.get("eval_steps", 50)),
        logging_steps=int(data.get("logging_steps", 10)),
        validation_fraction=float(data.get("validation_fraction", 0.05)),
        seed=int(data.get("seed", 42)),
        warmup_ratio=float(data.get("warmup_ratio", 0.03)),
        weight_decay=float(data.get("weight_decay", 0.01)),
        max_grad_norm=float(data.get("max_grad_norm", 1.0)),
        save_total_limit=int(data.get("save_total_limit", 3)),
        load_best_model_at_end=bool(data.get("load_best_model_at_end", True)),
        gradient_checkpointing=bool(data.get("gradient_checkpointing", False)),
    )
    _validate_sft_train_config(config)
    return config


def load_grpo_train_config(path: str | Path) -> GRPOTrainConfig:
    data = _load_yaml(path)
    config = GRPOTrainConfig(
        model_name_or_path=str(data["model_name_or_path"]),
        output_dir=str(data["output_dir"]),
        control_style=_control_style(data.get("control_style", "balanced")),
        prompt_data_path=data.get("prompt_data_path"),
        seed=int(data.get("seed", 42)),
        train_steps=int(data.get("train_steps", 800)),
        prompts_per_step=int(data.get("prompts_per_step", 4)),
        num_generations=int(data.get("num_generations", 4)),
        num_policy_epochs=int(data.get("num_policy_epochs", 1)),
        micro_batch_size=int(data.get("micro_batch_size", 4)),
        learning_rate=float(data.get("learning_rate", 1.0e-5)),
        min_learning_rate=float(data.get("min_learning_rate", 1.0e-6)),
        warmup_steps=int(data.get("warmup_steps", 50)),
        lr_decay_steps=int(data.get("lr_decay_steps", 800)),
        weight_decay=float(data.get("weight_decay", 0.0)),
        max_grad_norm=float(data.get("max_grad_norm", 1.0)),
        max_prompt_length=int(data.get("max_prompt_length", 6144)),
        max_completion_length=int(data.get("max_completion_length", 64)),
        temperature=float(data.get("temperature", 0.9)),
        top_p=float(data.get("top_p", 0.95)),
        top_k=int(data.get("top_k", 50)),
        clip_range=float(data.get("clip_range", 0.2)),
        beta_kl=float(data.get("beta_kl", 0.10)),
        adaptive_kl=bool(data.get("adaptive_kl", True)),
        target_kl=float(data.get("target_kl", 0.05)),
        kl_tolerance=float(data.get("kl_tolerance", 2.0)),
        beta_kl_min=float(data.get("beta_kl_min", 0.02)),
        beta_kl_max=float(data.get("beta_kl_max", 1.0)),
        kl_ema_alpha=float(data.get("kl_ema_alpha", 0.10)),
        kl_update_interval=int(data.get("kl_update_interval", 10)),
        kl_beta_multiplier=float(data.get("kl_beta_multiplier", 1.5)),
        kl_guard_multiplier=float(data.get("kl_guard_multiplier", 4.0)),
        kl_guard_patience=int(data.get("kl_guard_patience", 25)),
        bf16=bool(data.get("bf16", True)),
        use_qlora=bool(data.get("use_qlora", True)),
        lora_r=int(data.get("lora_r", 16)),
        lora_alpha=int(data.get("lora_alpha", 32)),
        excluded_plants_paths=tuple(str(path) for path in data.get("excluded_plants_paths", [])),
        lora_dropout=float(data.get("lora_dropout", 0.0)),
        second_order_prob=float(data.get("second_order_prob", 0.5)),
        demonstration=data.get("demonstration"),
        simulation=_simulation_from_dict(data.get("simulation", {})),
        reward=_reward_config_from_dict(data.get("reward", {})),
        save_every=int(data.get("save_every", 50)),
        log_every=int(data.get("log_every", 1)),
        save_rollouts_every=int(data.get("save_rollouts_every", 20)),
        validation_cases_paths=tuple(str(path) for path in data.get("validation_cases_paths", [])),
        validate_every=int(data.get("validate_every", 50)),
        validation_batch_size=int(data.get("validation_batch_size", 8)),
        resume_from_checkpoint=data.get("resume_from_checkpoint"),
    )
    _validate_grpo_train_config(config)
    return config


def load_server_job_config(path: str | Path) -> ServerJobConfig:
    data = _load_yaml(path)
    setup_commands = data.get("setup_commands", [])
    if isinstance(setup_commands, str):
        setup_commands = [setup_commands]
    return ServerJobConfig(
        job_name=str(data["job_name"]),
        command=str(data["command"]),
        output_dir=str(data["output_dir"]),
        partition=str(data.get("partition", "cluster")),
        gres=str(data.get("gres", "gpu:1")),
        cpus_per_task=int(data.get("cpus_per_task", 8)),
        mem=str(data.get("mem", "32G")),
        time=str(data.get("time", "04:00:00")),
        workdir=data.get("workdir"),
        setup_commands=[str(command) for command in setup_commands],
    )


def _load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _pid_from_dict(data: dict[str, Any]) -> PIDParams:
    return PIDParams(
        kp=float(data.get("kp", 1.0)),
        ki=float(data.get("ki", 0.1)),
        kd=float(data.get("kd", 0.01)),
    )


def _control_style(value: Any) -> str:
    style = str(value).strip().lower()
    if style not in {"balanced", "aggressive", "conservative"}:
        raise ValueError(f"Unsupported control_style: {value!r}.")
    return style


def _simulation_from_dict(data: dict[str, Any]) -> SimulationSettings:
    return SimulationSettings(
        setpoint=float(data.get("setpoint", 1.0)),
        sim_time=float(data.get("sim_time", 4000.0)),
        num_points=int(data.get("num_points", 40001)),
        time_delay=float(data.get("time_delay", 1.0)),
        max_abs_output=None
        if data.get("max_abs_output") is None
        else float(data["max_abs_output"]),
    )


def _reward_config_from_dict(data: dict[str, Any]) -> RewardConfig:
    defaults = RewardConfig()
    values: dict[str, Any] = {}
    for key, value in data.items():
        if key not in RewardConfig.__dataclass_fields__:
            continue
        default = getattr(defaults, key)
        if isinstance(default, bool):
            values[key] = bool(value)
        elif isinstance(default, int):
            values[key] = int(value)
        elif isinstance(default, float):
            values[key] = float(value)
        else:
            values[key] = value
    return RewardConfig(**values)


def _validate_sft_train_config(config: SFTTrainConfig) -> None:
    positive_values = {
        "max_length": config.max_length,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "save_steps": config.save_steps,
        "eval_steps": config.eval_steps,
        "logging_steps": config.logging_steps,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"SFT {name} must be positive.")
    if not 0.0 <= config.validation_fraction < 1.0:
        raise ValueError("SFT validation_fraction must be in [0, 1).")
    if config.save_steps % config.eval_steps != 0 and config.load_best_model_at_end:
        raise ValueError(
            "SFT save_steps must be a multiple of eval_steps when loading the best model."
        )


def _validate_grpo_train_config(config: GRPOTrainConfig) -> None:
    positive_values = {
        "train_steps": config.train_steps,
        "prompts_per_step": config.prompts_per_step,
        "num_policy_epochs": config.num_policy_epochs,
        "micro_batch_size": config.micro_batch_size,
        "learning_rate": config.learning_rate,
        "min_learning_rate": config.min_learning_rate,
        "lr_decay_steps": config.lr_decay_steps,
        "target_kl": config.target_kl,
        "kl_update_interval": config.kl_update_interval,
        "kl_beta_multiplier": config.kl_beta_multiplier,
        "kl_guard_multiplier": config.kl_guard_multiplier,
        "kl_guard_patience": config.kl_guard_patience,
        "max_prompt_length": config.max_prompt_length,
        "max_completion_length": config.max_completion_length,
        "temperature": config.temperature,
        "save_every": config.save_every,
        "log_every": config.log_every,
        "save_rollouts_every": config.save_rollouts_every,
        "validate_every": config.validate_every,
        "validation_batch_size": config.validation_batch_size,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"GRPO {name} must be positive.")
    if config.num_generations < 2:
        raise ValueError("GRPO num_generations must be at least 2 for group normalization.")
    if not 0.0 <= config.second_order_prob <= 1.0:
        raise ValueError("GRPO second_order_prob must be in [0, 1].")
    if config.lora_dropout != 0.0:
        raise ValueError(
            "GRPO lora_dropout must be 0 so rollout and update log-probabilities are comparable."
        )
    if config.warmup_steps < 0:
        raise ValueError("GRPO warmup_steps must be non-negative.")
    if config.min_learning_rate > config.learning_rate:
        raise ValueError("GRPO min_learning_rate cannot exceed learning_rate.")
    if not 0.0 < config.kl_ema_alpha <= 1.0:
        raise ValueError("GRPO kl_ema_alpha must be in (0, 1].")
    if config.kl_tolerance <= 1.0:
        raise ValueError("GRPO kl_tolerance must be greater than 1.")
    if config.kl_beta_multiplier <= 1.0:
        raise ValueError("GRPO kl_beta_multiplier must be greater than 1.")
    reward = config.reward
    weight_total = (
        reward.stability_weight
        + reward.performance_weight
        + reward.iae_weight
        + reward.regularization_weight
        + reward.format_weight
    )
    if abs(weight_total - 1.0) > 1.0e-9:
        raise ValueError("GRPO reward weights must sum to 1.")
    if not (
        reward.min_reward
        <= reward.invalid_reward
        <= reward.failed_branch_min
        <= reward.failed_branch_max
        < reward.success_branch_min
        <= reward.success_branch_max
        <= reward.max_reward
    ):
        raise ValueError("GRPO reward branch intervals are invalid or overlap.")
    if config.beta_kl_min > config.beta_kl or config.beta_kl > config.beta_kl_max:
        raise ValueError("GRPO beta_kl must be within [beta_kl_min, beta_kl_max].")
    if config.control_style != "balanced":
        raise ValueError(
            "Current GRPO rewards and gain references are calibrated for control_style=balanced."
        )
    if config.demonstration:
        demonstration_style = _control_style(config.demonstration.get("control_style", "balanced"))
        if demonstration_style != config.control_style:
            raise ValueError("GRPO prompt and demonstration control styles must match.")
