from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from llmpidtuner.training.config import GRPOTrainConfig


_RESUME_MUTABLE_FIELDS = {
    "train_steps",
    "resume_from_checkpoint",
    "save_every",
    "log_every",
    "save_rollouts_every",
}


@dataclass
class AdaptiveKLController:
    beta: float
    target: float
    tolerance: float
    beta_min: float
    beta_max: float
    ema_alpha: float
    update_interval: int
    beta_multiplier: float
    guard_multiplier: float
    guard_patience: int
    ema: float | None = None
    guard_steps: int = 0

    @classmethod
    def from_config(cls, config: GRPOTrainConfig) -> AdaptiveKLController:
        return cls(
            beta=config.beta_kl,
            target=config.target_kl,
            tolerance=config.kl_tolerance,
            beta_min=config.beta_kl_min,
            beta_max=config.beta_kl_max,
            ema_alpha=config.kl_ema_alpha,
            update_interval=config.kl_update_interval,
            guard_multiplier=config.kl_guard_multiplier,
            beta_multiplier=config.kl_beta_multiplier,
            guard_patience=config.kl_guard_patience,
        )

    def update(self, approx_kl: float, step: int, *, adaptive: bool = True) -> float:
        value = max(0.0, float(approx_kl))
        self.ema = (
            value
            if self.ema is None
            else (self.ema_alpha * value + (1.0 - self.ema_alpha) * self.ema)
        )
        if self.ema > self.target * self.guard_multiplier:
            self.guard_steps += 1
        else:
            self.guard_steps = 0

        if adaptive and step % self.update_interval == 0:
            if self.ema > self.target * self.tolerance:
                self.beta = min(self.beta_max, self.beta * self.beta_multiplier)
            elif self.ema < self.target / self.tolerance:
                self.beta = max(self.beta_min, self.beta / self.beta_multiplier)
        return self.beta

    @property
    def guard_triggered(self) -> bool:
        return self.guard_steps >= self.guard_patience

    def state_dict(self) -> dict[str, float | int | None]:
        return {
            "beta": self.beta,
            "ema": self.ema,
            "guard_steps": self.guard_steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.beta = float(state["beta"])
        self.ema = None if state.get("ema") is None else float(state["ema"])
        self.guard_steps = int(state.get("guard_steps", 0))


def learning_rate_at_step(config: GRPOTrainConfig, step: int) -> float:
    if step <= 0:
        return 0.0
    if config.warmup_steps > 0 and step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    if step >= config.lr_decay_steps:
        return config.min_learning_rate

    decay_start = max(1, config.warmup_steps)
    progress = (step - decay_start) / max(1, config.lr_decay_steps - decay_start)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + (config.learning_rate - config.min_learning_rate) * cosine


def validation_selection_key(result: dict[str, Any]) -> tuple[float, float, float, float]:
    kl_ema = float(result.get("kl_ema") or 0.0)
    return (
        float(result["success_rate"]),
        float(result["iae_improvement_mean"]),
        float(result["reward_mean"]),
        -kl_ema,
    )


def resume_compatibility_payload(config: GRPOTrainConfig) -> dict[str, Any]:
    payload = asdict(config)
    for field in _RESUME_MUTABLE_FIELDS:
        payload.pop(field, None)
    return json.loads(json.dumps(payload, sort_keys=True))


def resolve_resume_checkpoint(config: GRPOTrainConfig) -> Path | None:
    requested = config.resume_from_checkpoint
    if not requested:
        return None
    if requested != "auto":
        path = Path(requested)
        resolved = path if path.is_absolute() else Path(config.output_dir) / path
        if not (resolved / "trainer_state.json").is_file():
            raise FileNotFoundError(f"GRPO resume checkpoint is incomplete or missing: {resolved}.")
        return resolved

    checkpoints_root = Path(config.output_dir) / "checkpoints"
    candidates: list[tuple[int, Path]] = []
    if checkpoints_root.is_dir():
        for path in checkpoints_root.glob("checkpoint-*"):
            try:
                step = int(path.name.rsplit("-", 1)[-1])
            except ValueError:
                continue
            if (path / "trainer_state.json").is_file():
                candidates.append((step, path))
    if not candidates:
        raise FileNotFoundError(f"No resumable checkpoint was found under {checkpoints_root}.")
    return max(candidates, key=lambda item: item[0])[1]
