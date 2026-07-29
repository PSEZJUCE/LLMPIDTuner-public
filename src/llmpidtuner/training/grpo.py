from __future__ import annotations

import json
import time
from contextlib import nullcontext
from dataclasses import asdict
from statistics import median
from pathlib import Path
from typing import Any, Sequence

from llmpidtuner.demonstrations import (
    demonstration_protocol_id,
    generate_demonstration_from_spec,
)
from llmpidtuner.models import PIDParams
from llmpidtuner.training.artifacts import runtime_metadata, sha256_file, write_json_atomic
from llmpidtuner.training.config import GRPOTrainConfig
from llmpidtuner.training.grpo_control import (
    AdaptiveKLController,
    learning_rate_at_step,
    resolve_resume_checkpoint,
    resume_compatibility_payload,
    validation_selection_key,
)
from llmpidtuner.training.data import (
    PIDPromptGenerator,
    PromptPool,
    PromptSample,
    load_prompt_samples,
    load_protocol_prompt_samples,
)
from llmpidtuner.training.prompts import fit_messages_to_prompt_budget
from llmpidtuner.training.rewards import (
    RewardConfig,
    evaluate_completion,
    group_reward_stats,
    normalize_group_advantages,
)


def train_grpo(config: GRPOTrainConfig) -> None:
    """Run online GRPO training on the GPU server."""

    _validate_source_model_protocol(config)
    import torch
    import torch.nn.functional as F
    from accelerate import Accelerator
    from accelerate.utils import gather_object
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    accelerator = Accelerator(mixed_precision="bf16" if config.bf16 else "no")
    process_seed = config.seed + accelerator.process_index * 100003
    _set_seed(process_seed)
    resume_checkpoint = resolve_resume_checkpoint(config)

    if accelerator.is_main_process:
        output_dir = _prepare_grpo_output_dir(
            config.output_dir,
            resume=resume_checkpoint is not None,
        )
        _write_json(output_dir / "grpo_config.json", asdict(config))
        source_manifest = Path(config.model_name_or_path) / "training_manifest.json"
        manifest = {
            "schema_version": 2,
            "artifact_type": "grpo_adapter",
            "demonstration_protocol": demonstration_protocol_id(config.demonstration),
            "config": asdict(config),
            "status": "running",
            "distributed": {
                "world_size": accelerator.num_processes,
                "rank_seed_stride": 100003,
            },
            **runtime_metadata(
                (
                    "torch",
                    "transformers",
                    "accelerate",
                    "peft",
                    "bitsandbytes",
                )
            ),
        }
        if source_manifest.is_file():
            manifest["source_model_manifest"] = {
                "path": str(source_manifest),
                "sha256": sha256_file(source_manifest),
            }
        if resume_checkpoint is not None:
            manifest["resumed_from"] = str(resume_checkpoint)
        write_json_atomic(output_dir / "training_manifest.json", manifest)
    accelerator.wait_for_everyone()

    with accelerator.main_process_first():
        prompt_provider = _build_prompt_provider(config, process_seed)
    validation_samples = (
        _build_validation_samples(config)
        if accelerator.is_main_process and config.validation_cases_paths
        else []
    )
    reward_config = config.reward
    gain_reference = None
    if accelerator.is_main_process:
        _write_json(
            Path(config.output_dir) / "reward_metadata.json",
            {
                "reward_config": asdict(reward_config),
                "gain_reference": "per_plant_balanced_imc",
                "control_style": config.control_style,
            },
        )

    policy_model, ref_model, tokenizer = _load_policy_and_ref(
        config,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        torch,
    )
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    policy_model, ref_model, optimizer = accelerator.prepare(policy_model, ref_model, optimizer)
    ref_model.eval()

    kl_controller = AdaptiveKLController.from_config(config)
    start_step = 0
    total_tokens = 0
    elapsed_before_resume = 0.0
    baseline_validation: dict[str, Any] | None = None
    best_validation_result: dict[str, Any] | None = None
    best_validation_key: tuple[float, float, float, float] | None = None
    best_validation_step = 0

    if resume_checkpoint is not None:
        resume_state = _load_resume_training_state(
            resume_checkpoint,
            config,
            accelerator.process_index,
            accelerator.num_processes,
        )
        accelerator.load_state(str(resume_checkpoint / "accelerator_state"))
        prompt_provider.load_state_dict(resume_state["prompt_provider_state"])
        start_step = int(resume_state["step"])
        total_tokens = int(resume_state["total_tokens"])
        elapsed_before_resume = float(resume_state.get("elapsed_sec", 0.0))
        kl_controller.load_state_dict(resume_state["adaptive_kl"])
        baseline_validation = resume_state.get("baseline_validation")
        best_validation_result = resume_state.get("best_validation")
        raw_key = resume_state.get("best_validation_key")
        best_validation_key = tuple(float(value) for value in raw_key) if raw_key else None
        best_validation_step = int(resume_state.get("best_validation_step", 0))
        if config.train_steps <= start_step:
            raise ValueError(
                "GRPO train_steps is a cumulative target and must exceed the resumed "
                f"checkpoint step ({start_step})."
            )
    elif config.validation_cases_paths:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            baseline_validation = _evaluate_validation(
                accelerator,
                policy_model,
                tokenizer,
                validation_samples,
                config,
                reward_config,
                torch,
            )
            baseline_validation.update(
                {
                    "step": 0,
                    "kl_ema": 0.0,
                    "source_model": "sft",
                }
            )
            _append_jsonl(
                Path(config.output_dir) / "validation_log.jsonl",
                baseline_validation,
            )
            best_validation_key = validation_selection_key(baseline_validation)
        accelerator.wait_for_everyone()

    start_time = time.time()
    completed_step = start_step
    stopped_for_kl = False
    for step in range(start_step + 1, config.train_steps + 1):
        current_lr = learning_rate_at_step(config, step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        samples = prompt_provider.sample_batch(config.prompts_per_step)
        fitted_prompts = [
            fit_messages_to_prompt_budget(tokenizer, sample.messages, config.max_prompt_length)
            for sample in samples
        ]
        prompts = [item[0] for item in fitted_prompts]
        prompt_metadata = [item[1] for item in fitted_prompts]
        rollout = _generate_rollout(
            accelerator,
            policy_model,
            ref_model,
            tokenizer,
            prompts,
            samples,
            config,
            reward_config,
            gain_reference,
            torch,
        )
        rollout["prompt_metadata"] = prompt_metadata
        local_tokens = rollout["completion_mask"].sum().to(accelerator.device)
        total_tokens += int(accelerator.gather(local_tokens).sum().detach().cpu().item())
        rank_step_data = gather_object(
            [
                _build_rank_step_data(
                    accelerator.process_index,
                    rollout,
                    prompt_metadata,
                )
            ]
        )
        global_step_data = _combine_rank_step_data(rank_step_data)

        policy_model.train()
        loss_values: list[float] = []
        approx_kl_values: list[float] = []
        clipfrac_values: list[float] = []
        beta_used = kl_controller.beta
        for _ in range(config.num_policy_epochs):
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _grpo_backward(
                accelerator=accelerator,
                policy_model=policy_model,
                sequences=rollout["sequences"],
                attention_mask=rollout["attention_mask"],
                completion_start=rollout["completion_start"],
                completion_mask=rollout["completion_mask"],
                old_logps=rollout["old_logps"],
                ref_logps=rollout["ref_logps"],
                advantages=rollout["advantages"],
                clip_range=config.clip_range,
                beta_kl=beta_used,
                micro_batch_size=config.micro_batch_size,
                torch=torch,
                F=F,
            )
            if config.max_grad_norm and config.max_grad_norm > 0:
                accelerator.clip_grad_norm_(policy_model.parameters(), config.max_grad_norm)
            optimizer.step()
            loss_values.append(float(accelerator.gather(loss.detach()).mean().cpu().item()))
            local_metric_totals = torch.tensor(
                [
                    metrics["approx_kl_sum"],
                    metrics["clip_count"],
                    metrics["token_count"],
                ],
                dtype=torch.float64,
                device=accelerator.device,
            )
            global_metric_totals = accelerator.gather(local_metric_totals).reshape(-1, 3).sum(0)
            metric_token_count = global_metric_totals[2].clamp_min(1.0)
            approx_kl_values.append(
                float((global_metric_totals[0] / metric_token_count).cpu().item())
            )
            clipfrac_values.append(
                float((global_metric_totals[1] / metric_token_count).cpu().item())
            )

        approx_kl = float(sum(approx_kl_values) / max(1, len(approx_kl_values)))
        kl_controller.update(approx_kl, step, adaptive=config.adaptive_kl)
        completed_step = step
        elapsed = elapsed_before_resume + time.time() - start_time

        if accelerator.is_main_process and step % max(1, config.log_every) == 0:
            rewards = global_step_data["rewards"]
            advantages = global_step_data["advantages"]
            prompt_metadata_global = global_step_data["prompt_metadata"]
            termination_metadata = global_step_data["termination_metadata"]
            completion_lengths = [int(item["generated_tokens"]) for item in termination_metadata]
            log_item = {
                "step": step,
                "loss": float(sum(loss_values) / max(1, len(loss_values))),
                "reward_mean": float(sum(rewards) / max(1, len(rewards))),
                "reward_min": float(min(rewards)),
                "reward_max": float(max(rewards)),
                "advantage_mean": float(sum(advantages) / max(1, len(advantages))),
                "approx_kl": approx_kl,
                "kl_ema": kl_controller.ema,
                "beta_kl_used": beta_used,
                "beta_kl_next": kl_controller.beta,
                "target_kl": config.target_kl,
                "kl_guard_steps": kl_controller.guard_steps,
                "clipfrac": float(sum(clipfrac_values) / max(1, len(clipfrac_values))),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "tokens": int(total_tokens),
                "elapsed_sec": float(elapsed),
                "tokens_per_sec": float(total_tokens / max(elapsed, 1e-8)),
                "group_reward_stats": group_reward_stats(rewards, config.num_generations),
                "prompt_tokens_mean": float(
                    sum(item["final_tokens"] for item in prompt_metadata_global)
                    / max(1, len(prompt_metadata_global))
                ),
                "prompt_tokens_max": int(
                    max(item["final_tokens"] for item in prompt_metadata_global)
                ),
                "prompts_trimmed": int(
                    sum(bool(item["trimmed"]) for item in prompt_metadata_global)
                ),
                "demonstration_cases_used_min": int(
                    min(item["demonstration_cases_used"] for item in prompt_metadata_global)
                ),
                "completion_tokens_mean": float(
                    sum(completion_lengths) / max(1, len(completion_lengths))
                ),
                "completion_tokens_max": int(max(completion_lengths)),
                "completion_eos_count": int(
                    sum(bool(item["terminated_by_eos"]) for item in termination_metadata)
                ),
                "completion_length_limit_count": int(
                    sum(bool(item["hit_length_limit"]) for item in termination_metadata)
                ),
                "rank_count": int(len(global_step_data["ranks"])),
                "prompt_count": int(len(prompt_metadata_global)),
                "completion_count": int(len(rewards)),
            }
            _append_jsonl(Path(config.output_dir) / "trainer_log.jsonl", log_item)
            print(
                f"step {step}/{config.train_steps} "
                f"loss={log_item['loss']:.4f} reward={log_item['reward_mean']:.4f} "
                f"kl={approx_kl:.4f} beta={kl_controller.beta:.4f}"
            )

        if step % max(1, config.save_rollouts_every) == 0:
            rank_rollouts = gather_object(
                [
                    {
                        "rank": accelerator.process_index,
                        "rows": _build_rollout_log_rows(
                            config,
                            samples,
                            prompts,
                            rollout,
                            accelerator.process_index,
                        ),
                    }
                ]
            )
            if accelerator.is_main_process:
                _write_rollout_log(
                    config,
                    step,
                    _combine_rank_rollout_rows(rank_rollouts),
                    rank_count=len(rank_rollouts),
                )

        if config.validation_cases_paths and step % max(1, config.validate_every) == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                validation = _evaluate_validation(
                    accelerator,
                    policy_model,
                    tokenizer,
                    validation_samples,
                    config,
                    reward_config,
                    torch,
                )
                validation.update(
                    {
                        "step": step,
                        "kl_ema": kl_controller.ema,
                        "source_model": "grpo",
                    }
                )
                _append_jsonl(Path(config.output_dir) / "validation_log.jsonl", validation)
                validation_key = validation_selection_key(validation)
                if best_validation_key is None or validation_key > best_validation_key:
                    best_validation_key = validation_key
                    best_validation_result = dict(validation)
                    best_validation_step = step
                    _save_adapter(
                        accelerator,
                        policy_model,
                        tokenizer,
                        config.output_dir,
                        "best-validation-checkpoint",
                    )
            accelerator.wait_for_everyone()

        stopped_for_kl = kl_controller.guard_triggered
        should_checkpoint = (
            step % max(1, config.save_every) == 0 or step == config.train_steps or stopped_for_kl
        )
        if should_checkpoint:
            _save_training_checkpoint(
                accelerator=accelerator,
                policy_model=policy_model,
                optimizer=optimizer,
                prompt_provider=prompt_provider,
                config=config,
                step=step,
                total_tokens=total_tokens,
                elapsed_sec=elapsed,
                kl_controller=kl_controller,
                baseline_validation=baseline_validation,
                best_validation=best_validation_result,
                best_validation_key=best_validation_key,
                best_validation_step=best_validation_step,
                status="kl_guard" if stopped_for_kl else "running",
            )
        if stopped_for_kl:
            if accelerator.is_main_process:
                print(
                    "Stopping GRPO because KL EMA exceeded the emergency threshold "
                    f"for {config.kl_guard_patience} consecutive steps."
                )
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _save_adapter(accelerator, policy_model, tokenizer, config.output_dir, "lora_adapter")
        elapsed_total = elapsed_before_resume + time.time() - start_time
        selected_source = "grpo" if best_validation_step > 0 else "sft"
        manifest.update(
            {
                "status": "stopped_kl_guard" if stopped_for_kl else "complete",
                "train_steps_completed": completed_step,
                "tokens": total_tokens,
                "elapsed_sec": elapsed_total,
                "latest_adapter_path": "lora_adapter",
                "baseline_validation": baseline_validation,
                "best_validation": best_validation_result or baseline_validation,
                "best_validation_step": best_validation_step,
                "selected_model_source": selected_source,
                "selected_model_path": (
                    "best-validation-checkpoint"
                    if selected_source == "grpo"
                    else config.model_name_or_path
                ),
                "adaptive_kl": kl_controller.state_dict(),
            }
        )
        write_json_atomic(Path(config.output_dir) / "training_manifest.json", manifest)
        print(
            "Training finished. Selected model source: "
            f"{selected_source} (validation step {best_validation_step})."
        )


def _load_resume_training_state(
    checkpoint: Path,
    config: GRPOTrainConfig,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    trainer_state_path = checkpoint / "trainer_state.json"
    rank_state_path = checkpoint / f"rank_{rank:03d}_state.json"
    accelerator_state = checkpoint / "accelerator_state"
    for required in (trainer_state_path, rank_state_path, accelerator_state):
        if not required.exists():
            raise FileNotFoundError(f"Incomplete GRPO resume checkpoint: {required}")

    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    if int(trainer_state["world_size"]) != world_size:
        raise ValueError(
            "Exact GRPO resume requires the same distributed world size: "
            f"{trainer_state['world_size']} != {world_size}."
        )
    expected = trainer_state["config_compatibility"]
    actual = resume_compatibility_payload(config)
    if actual != expected:
        raise ValueError(
            "GRPO resume configuration differs from the checkpoint. Only train_steps, "
            "resume_from_checkpoint, save_every, log_every, and "
            "save_rollouts_every may change."
        )
    rank_state = json.loads(rank_state_path.read_text(encoding="utf-8"))
    trainer_state["prompt_provider_state"] = rank_state["prompt_provider_state"]
    return trainer_state


def _save_training_checkpoint(
    *,
    accelerator,
    policy_model,
    optimizer,
    prompt_provider: PIDPromptGenerator | PromptPool,
    config: GRPOTrainConfig,
    step: int,
    total_tokens: int,
    elapsed_sec: float,
    kl_controller: AdaptiveKLController,
    baseline_validation: dict[str, Any] | None,
    best_validation: dict[str, Any] | None,
    best_validation_key: tuple[float, float, float, float] | None,
    best_validation_step: int,
    status: str,
) -> None:
    del policy_model, optimizer
    checkpoint = Path(config.output_dir) / "checkpoints" / f"checkpoint-{step}"
    if accelerator.is_main_process:
        checkpoint.mkdir(parents=True, exist_ok=False)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(checkpoint / "accelerator_state"))
    write_json_atomic(
        checkpoint / f"rank_{accelerator.process_index:03d}_state.json",
        {
            "rank": accelerator.process_index,
            "prompt_provider_state": prompt_provider.state_dict(),
        },
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        write_json_atomic(
            checkpoint / "trainer_state.json",
            {
                "schema_version": 2,
                "step": step,
                "total_tokens": total_tokens,
                "elapsed_sec": elapsed_sec,
                "world_size": accelerator.num_processes,
                "status": status,
                "adaptive_kl": kl_controller.state_dict(),
                "baseline_validation": baseline_validation,
                "best_validation": best_validation,
                "best_validation_key": (
                    list(best_validation_key) if best_validation_key is not None else None
                ),
                "best_validation_step": best_validation_step,
                "config_compatibility": resume_compatibility_payload(config),
            },
        )
    accelerator.wait_for_everyone()


def _validate_source_model_protocol(config: GRPOTrainConfig) -> None:
    expected_protocol = demonstration_protocol_id(config.demonstration)
    if expected_protocol is None:
        raise ValueError("GRPO training requires a frozen demonstration protocol.")
    manifest_path = Path(config.model_name_or_path) / "training_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"GRPO requires the SFT source-model manifest for protocol validation: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_protocol = manifest.get("demonstration_protocol")
    if actual_protocol != expected_protocol:
        raise ValueError(
            "GRPO source-model demonstration protocol mismatch: "
            f"{actual_protocol!r} != {expected_protocol!r}"
        )

    actual_style = manifest.get("control_style")
    if actual_style != config.control_style:
        raise ValueError(
            "GRPO source-model control style mismatch: "
            f"{actual_style!r} != {config.control_style!r}"
        )


def _build_prompt_provider(config: GRPOTrainConfig, seed: int) -> PIDPromptGenerator | PromptPool:
    if config.demonstration and config.demonstration.get("method") != "frozen":
        raise ValueError("GRPO training requires a versioned frozen demonstration protocol.")
    if config.prompt_data_path:
        samples = load_prompt_samples(config.prompt_data_path)
        if any(sample.control_style != config.control_style for sample in samples):
            raise ValueError("GRPO prompt pool contains a mismatched control style.")
        return PromptPool(samples, seed=seed)

    demonstrations: dict[str, str] = {}
    if config.demonstration:
        initial_pid = PIDParams(1.0, 0.1, 0.01)
        for system in ("first_order", "second_order"):
            text = generate_demonstration_from_spec(
                {**config.demonstration, "system": system},
                initial_pid=initial_pid,
                simulation=config.simulation,
            )
            if text:
                demonstrations[system] = text
    return PIDPromptGenerator(
        seed=seed,
        simulation=config.simulation,
        second_order_prob=config.second_order_prob,
        demonstrations=demonstrations,
        excluded_plants_paths=config.excluded_plants_paths,
        control_style=config.control_style,
    )


def _load_demonstrations(config: GRPOTrainConfig) -> dict[str, str]:
    demonstrations: dict[str, str] = {}
    if not config.demonstration:
        return demonstrations
    initial_pid = PIDParams(1.0, 0.1, 0.01)
    for system in ("first_order", "second_order"):
        text = generate_demonstration_from_spec(
            {**config.demonstration, "system": system},
            initial_pid=initial_pid,
            simulation=config.simulation,
        )
        if text:
            demonstrations[system] = text
    return demonstrations


def _build_validation_samples(config: GRPOTrainConfig) -> list[PromptSample]:
    return load_protocol_prompt_samples(
        config.validation_cases_paths,
        simulation=config.simulation,
        demonstrations=_load_demonstrations(config),
        control_style=config.control_style,
    )


def _evaluate_validation(
    accelerator,
    policy_model,
    tokenizer,
    samples: Sequence[PromptSample],
    config: GRPOTrainConfig,
    reward_config: RewardConfig,
    torch,
) -> dict[str, Any]:
    model = accelerator.unwrap_model(policy_model)
    model.eval()
    rewards: list[float] = []
    successes = 0
    iae_improvements: list[float] = []
    with torch.no_grad():
        for start in range(0, len(samples), config.validation_batch_size):
            batch = samples[start : start + config.validation_batch_size]
            prompts = [
                fit_messages_to_prompt_budget(
                    tokenizer,
                    sample.messages,
                    config.max_prompt_length,
                )[0]
                for sample in batch
            ]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
            )
            encoded = {key: value.to(accelerator.device) for key, value in encoded.items()}
            completion_start = int(encoded["input_ids"].shape[1])
            sequences = model.generate(
                **encoded,
                max_new_tokens=config.max_completion_length,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            completion_ids = sequences[:, completion_start:]
            termination = _completion_termination_metadata(
                completion_ids,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                max_completion_length=config.max_completion_length,
            )
            completions = tokenizer.batch_decode(
                completion_ids.detach().cpu(),
                skip_special_tokens=True,
            )
            for completion, sample, stop in zip(completions, batch, termination):
                result = evaluate_completion(
                    completion,
                    sample,
                    config.simulation,
                    reward_config,
                    None,
                    completion_hit_length_limit=bool(stop["hit_length_limit"]),
                )
                rewards.append(float(result.reward))
                if result.metrics is not None and result.metrics.converged():
                    successes += 1
                iae_improvements.append(float(result.components["iae"]))
    return {
        "count": len(samples),
        "reward_mean": float(sum(rewards) / max(1, len(rewards))),
        "success_count": successes,
        "success_rate": float(successes / max(1, len(samples))),
        "iae_improvement_mean": float(sum(iae_improvements) / max(1, len(iae_improvements))),
        "iae_improvement_median": float(median(iae_improvements)) if iae_improvements else 0.0,
    }


def _load_policy_and_ref(
    config: GRPOTrainConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    torch,
) -> tuple[Any, Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, trust_remote_code=True, use_fast=True
    )
    _ensure_chat_template(tokenizer, config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"

    torch_dtype = torch.bfloat16 if config.bf16 else torch.float16
    quant_config = None
    if config.use_qlora:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )

    policy_model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        quantization_config=quant_config,
    )
    if config.use_qlora:
        policy_model = prepare_model_for_kbit_training(policy_model)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    policy_model = get_peft_model(policy_model, lora_config)

    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        quantization_config=quant_config,
    )
    for param in ref_model.parameters():
        param.requires_grad_(False)
    return policy_model, ref_model, tokenizer


def _generate_rollout(
    accelerator,
    policy_model,
    ref_model,
    tokenizer,
    prompts: Sequence[str],
    samples: Sequence[PromptSample],
    config: GRPOTrainConfig,
    reward_config: RewardConfig,
    gain_reference,
    torch,
) -> dict[str, Any]:
    policy_model.eval()
    repeated_prompts: list[str] = []
    repeated_samples: list[PromptSample] = []
    for prompt, sample in zip(prompts, samples):
        for _ in range(config.num_generations):
            repeated_prompts.append(prompt)
            repeated_samples.append(sample)

    encoded = tokenizer(
        repeated_prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    )
    if int(encoded["input_ids"].shape[1]) > config.max_prompt_length:
        raise ValueError("A GRPO prompt exceeded max_prompt_length after structural fitting.")
    encoded = {key: value.to(accelerator.device) for key, value in encoded.items()}
    completion_start = int(encoded["input_ids"].shape[1])
    sequences = accelerator.unwrap_model(policy_model).generate(
        **encoded,
        max_new_tokens=config.max_completion_length,
        do_sample=True,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        synced_gpus=accelerator.num_processes > 1,
    )
    attention_mask = (sequences != tokenizer.pad_token_id).long()
    completion_ids = sequences[:, completion_start:]
    termination_metadata = _completion_termination_metadata(
        completion_ids,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_completion_length=config.max_completion_length,
    )
    completion_mask = _build_completion_mask(
        completion_ids,
        termination_metadata,
        torch,
    ).to(accelerator.device)
    completions = tokenizer.batch_decode(completion_ids.detach().cpu(), skip_special_tokens=True)

    reward_results = [
        evaluate_completion(
            completion,
            sample,
            config.simulation,
            reward_config,
            gain_reference,
            completion_hit_length_limit=termination["hit_length_limit"],
        )
        for completion, sample, termination in zip(
            completions, repeated_samples, termination_metadata
        )
    ]
    for termination, result in zip(termination_metadata, reward_results):
        termination["parsed_pid_complete"] = result.parsed_pid is not None
        termination["reward_policy"] = (
            "length_limit_invalid"
            if termination["hit_length_limit"] and result.parsed_pid is None
            else "length_limit_format_zero"
            if termination["hit_length_limit"]
            else "normal"
        )
    rewards = [result.reward for result in reward_results]
    advantages = torch.tensor(
        normalize_group_advantages(rewards, config.num_generations),
        dtype=torch.float32,
        device=accelerator.device,
    )
    old_logps = _collect_completion_logps(
        policy_model, sequences, attention_mask, completion_start, config.micro_batch_size, torch
    )
    ref_logps = _collect_completion_logps(
        ref_model, sequences, attention_mask, completion_start, config.micro_batch_size, torch
    )
    return {
        "sequences": sequences,
        "attention_mask": attention_mask,
        "completion_start": completion_start,
        "completion_mask": completion_mask,
        "completion_ids": completion_ids,
        "completions": completions,
        "termination_metadata": termination_metadata,
        "rewards": rewards,
        "reward_results": reward_results,
        "advantages": advantages,
        "old_logps": old_logps.detach(),
        "ref_logps": ref_logps.detach(),
    }


def _build_rank_step_data(
    rank: int,
    rollout: dict[str, Any],
    prompt_metadata: Sequence[dict[str, int | bool]],
) -> dict[str, Any]:
    return {
        "rank": int(rank),
        "rewards": [float(value) for value in rollout["rewards"]],
        "advantages": [float(value) for value in rollout["advantages"].detach().cpu().tolist()],
        "prompt_metadata": [dict(item) for item in prompt_metadata],
        "termination_metadata": [dict(item) for item in rollout["termination_metadata"]],
    }


def _combine_rank_step_data(rank_data: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rank_data, key=lambda item: int(item["rank"]))
    ranks = [int(item["rank"]) for item in ordered]
    if len(set(ranks)) != len(ranks):
        raise ValueError("Duplicate rank data received while gathering GRPO metrics.")

    combined: dict[str, Any] = {
        "ranks": ranks,
        "rewards": [],
        "advantages": [],
        "prompt_metadata": [],
        "termination_metadata": [],
    }
    for item in ordered:
        combined["rewards"].extend(item["rewards"])
        combined["advantages"].extend(item["advantages"])
        combined["prompt_metadata"].extend(item["prompt_metadata"])
        combined["termination_metadata"].extend(item["termination_metadata"])
    return combined


def _grpo_backward(
    accelerator,
    policy_model,
    sequences,
    attention_mask,
    completion_start: int,
    completion_mask,
    old_logps,
    ref_logps,
    advantages,
    clip_range: float,
    beta_kl: float,
    micro_batch_size: int,
    torch,
    F,
) -> tuple[Any, dict[str, float]]:
    if sequences.shape[0] == 0:
        raise ValueError("Cannot compute GRPO loss for an empty rollout.")

    total_loss_num = torch.zeros((), dtype=torch.float32, device=sequences.device)
    total_tokens = completion_mask.float().sum().clamp_min(1.0)
    approx_kl_num = torch.zeros((), dtype=torch.float32, device=sequences.device)
    clip_num = torch.zeros((), dtype=torch.float32, device=sequences.device)
    total_count = torch.zeros((), dtype=torch.float32, device=sequences.device)

    n = sequences.shape[0]
    starts = list(range(0, n, max(1, micro_batch_size)))
    for chunk_index, start in enumerate(starts):
        end = min(n, start + max(1, micro_batch_size))
        sync_context = (
            accelerator.no_sync(policy_model) if chunk_index < len(starts) - 1 else nullcontext()
        )
        with sync_context:
            logps = _completion_logps(
                policy_model, sequences[start:end], attention_mask[start:end], completion_start, F
            )
            mask = completion_mask[start:end].float()
            adv = advantages[start:end].unsqueeze(1)
            old = old_logps[start:end]
            ref = ref_logps[start:end]
            log_ratio = logps - old
            ratio = torch.exp(torch.clamp(log_ratio, min=-20.0, max=20.0))
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
            policy_loss = -torch.minimum(unclipped, clipped)
            kl = torch.exp(torch.clamp(ref - logps, min=-20.0, max=20.0)) - (ref - logps) - 1.0
            token_loss = (policy_loss + beta_kl * kl) * mask
            loss_sum = token_loss.sum()
            accelerator.backward(loss_sum / total_tokens)

        total_loss_num = total_loss_num + loss_sum.detach()
        approx_kl_num = approx_kl_num + (kl.detach() * mask).sum()
        clip_num = clip_num + (((ratio.detach() - 1.0).abs() > clip_range).float() * mask).sum()
        total_count = total_count + mask.sum()

    loss = total_loss_num / total_tokens
    metrics = {
        "approx_kl": float((approx_kl_num / total_count.clamp_min(1.0)).detach().cpu().item()),
        "clipfrac": float((clip_num / total_count.clamp_min(1.0)).detach().cpu().item()),
        "approx_kl_sum": float(approx_kl_num.detach().cpu().item()),
        "clip_count": float(clip_num.detach().cpu().item()),
        "token_count": float(total_count.detach().cpu().item()),
    }
    return loss, metrics


def _collect_completion_logps(
    model,
    sequences,
    attention_mask,
    completion_start: int,
    micro_batch_size: int,
    torch,
):
    chunks = []
    with torch.no_grad():
        for start in range(0, sequences.shape[0], max(1, micro_batch_size)):
            end = min(sequences.shape[0], start + max(1, micro_batch_size))
            chunks.append(
                _completion_logps(
                    model, sequences[start:end], attention_mask[start:end], completion_start
                )
            )
    return torch.cat(chunks, dim=0)


def _completion_logps(model, sequences, attention_mask, completion_start: int, F_module=None):
    if F_module is None:
        import torch.nn.functional as F_module

    outputs = model(input_ids=sequences, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = sequences[:, 1:]
    token_logps = (
        F_module.log_softmax(logits, dim=-1).gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    )
    return token_logps[:, completion_start - 1 :]


def _completion_termination_metadata(
    completion_ids,
    eos_token_id: int | None,
    pad_token_id: int | None,
    max_completion_length: int,
) -> list[dict[str, Any]]:
    rows = (
        completion_ids.detach().cpu().tolist()
        if hasattr(completion_ids, "detach")
        else completion_ids
    )
    metadata: list[dict[str, Any]] = []
    for row in rows:
        eos_index = (
            row.index(eos_token_id) if eos_token_id is not None and eos_token_id in row else None
        )
        if eos_index is not None:
            generated_tokens = eos_index + 1
            terminated_by_eos = True
        else:
            generated_tokens = (
                sum(token != pad_token_id for token in row)
                if pad_token_id is not None and pad_token_id != eos_token_id
                else len(row)
            )
            terminated_by_eos = False
        hit_length_limit = not terminated_by_eos and generated_tokens >= max_completion_length
        metadata.append(
            {
                "generated_tokens": int(generated_tokens),
                "terminated_by_eos": terminated_by_eos,
                "hit_length_limit": hit_length_limit,
                "termination_reason": (
                    "eos"
                    if terminated_by_eos
                    else "length_limit"
                    if hit_length_limit
                    else "generation_stopped"
                ),
            }
        )
    return metadata


def _build_completion_mask(
    completion_ids,
    termination_metadata: Sequence[dict[str, Any]],
    torch,
):
    if len(termination_metadata) != completion_ids.shape[0]:
        raise ValueError("Completion metadata count does not match the generated batch.")
    mask = torch.zeros_like(completion_ids, dtype=torch.bool)
    for row, metadata in enumerate(termination_metadata):
        valid_tokens = int(metadata["generated_tokens"])
        if valid_tokens > 0:
            mask[row, :valid_tokens] = True
    return mask


def _build_rollout_log_rows(
    config: GRPOTrainConfig,
    samples: Sequence[PromptSample],
    prompts: Sequence[str],
    rollout: dict[str, Any],
    rank: int,
) -> list[dict[str, Any]]:
    rows = []
    reward_results = rollout["reward_results"]
    completions = rollout["completions"]
    rewards = rollout["rewards"]
    advantages = rollout["advantages"].detach().cpu().tolist()
    termination_metadata = rollout["termination_metadata"]
    for i, sample in enumerate(samples):
        group = []
        for j in range(config.num_generations):
            idx = i * config.num_generations + j
            group.append(
                {
                    "completion": completions[idx],
                    "reward": float(rewards[idx]),
                    "advantage": float(advantages[idx]),
                    "termination": termination_metadata[idx],
                    "reward_detail": reward_results[idx].as_dict(),
                }
            )
        rows.append(
            {
                "rank": int(rank),
                "prompt": prompts[i],
                "prompt_metadata": rollout["prompt_metadata"][i],
                "sample": sample.as_dict(),
                "group": group,
            }
        )
    return rows


def _combine_rank_rollout_rows(
    rank_rollouts: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted(rank_rollouts, key=lambda item: int(item["rank"]))
    ranks = [int(item["rank"]) for item in ordered]
    if len(set(ranks)) != len(ranks):
        raise ValueError("Duplicate rank data received while gathering GRPO rollouts.")
    return [row for item in ordered for row in item["rows"]]


def _write_rollout_log(
    config: GRPOTrainConfig,
    step: int,
    rows: Sequence[dict[str, Any]],
    rank_count: int,
) -> None:
    _append_jsonl(
        Path(config.output_dir) / "rollouts.jsonl",
        {
            "step": step,
            "rank_count": int(rank_count),
            "rollout_count": len(rows),
            "rollouts": list(rows),
        },
    )


def _save_adapter(accelerator, policy_model, tokenizer, output_dir: str, suffix: str) -> None:
    save_dir = Path(output_dir) / suffix
    save_dir.mkdir(parents=True, exist_ok=True)
    accelerator.unwrap_model(policy_model).save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def _ensure_chat_template(tokenizer: Any, model_path: str) -> None:
    if getattr(tokenizer, "chat_template", None):
        return
    template_path = Path(model_path) / "chat_template.jinja"
    if template_path.exists():
        tokenizer.chat_template = template_path.read_text(encoding="utf-8")


def _set_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_grpo_output_dir(output_dir: str | Path, *, resume: bool = False) -> Path:
    path = Path(output_dir)
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"GRPO output path is not a directory: {path}.")
    if path.exists() and any(path.iterdir()) and not resume:
        raise FileExistsError(
            f"GRPO output directory is not empty: {path}. "
            "Move or remove the existing experiment before starting a new run."
        )
    if resume and not path.is_dir():
        raise FileNotFoundError(f"GRPO resume output directory does not exist: {path}.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def _write_json(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
