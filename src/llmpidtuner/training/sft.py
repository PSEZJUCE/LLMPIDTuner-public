from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

from llmpidtuner.training.artifacts import runtime_metadata, sha256_file, write_json_atomic
from llmpidtuner.training.config import SFTTrainConfig
from llmpidtuner.training.prompts import messages_to_prompt


def validate_sft_dataset(config: SFTTrainConfig) -> dict[str, int]:
    """Tokenize every row and reject any sequence above the configured limit."""

    _validate_dataset_manifest(config.dataset_path, config.control_style)
    audit_sft_dataset(config.dataset_path, config.control_style)
    tokenizer = _load_sft_tokenizer(config.model_name_or_path)
    dataset = _SFTDataset(config.dataset_path, tokenizer, config.max_length)
    return dataset.validate_lengths()


def train_sft(config: SFTTrainConfig) -> None:
    """Run supervised fine-tuning with answer-only loss on the server."""

    import torch
    from torch.utils.data import Subset
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    _validate_dataset_manifest(config.dataset_path, config.control_style)
    audit_sft_dataset(config.dataset_path, config.control_style)
    _prepare_sft_output_dir(config.output_dir)
    tokenizer = _load_sft_tokenizer(config.model_name_or_path)
    dataset = _SFTDataset(config.dataset_path, tokenizer, config.max_length)
    if os.environ.get("LLMPIDTUNER_SFT_PREFLIGHT_DONE") != "1":
        dataset.validate_lengths()

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
    )
    if config.gradient_checkpointing:
        model.config.use_cache = False

    train_indices, eval_indices = _split_indices(
        len(dataset),
        validation_fraction=config.validation_fraction,
        seed=config.seed,
        strata=dataset.validation_strata,
    )
    train_dataset = Subset(dataset, train_indices)
    eval_dataset = Subset(dataset, eval_indices) if eval_indices else None
    use_eval = eval_dataset is not None
    load_best = bool(config.load_best_model_at_end and use_eval)

    print(
        f"SFT dataset: total={len(dataset)}, train={len(train_indices)}, "
        f"validation={len(eval_indices)}, max_length={config.max_length}"
    )
    args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        eval_strategy="steps" if use_eval else "no",
        eval_steps=config.eval_steps if use_eval else None,
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_loss" if load_best else None,
        greater_is_better=False if load_best else None,
        save_total_limit=config.save_total_limit,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        seed=config.seed,
        data_seed=config.seed,
        gradient_checkpointing=config.gradient_checkpointing,
        remove_unused_columns=False,
        report_to=[],
        ddp_find_unused_parameters=False,
        include_num_input_tokens_seen=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=partial(
            _causal_lm_collator,
            pad_token_id=int(tokenizer.pad_token_id),
        ),
    )
    train_result = trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    trainer.save_state()
    trainer.save_metrics("train", train_result.metrics)
    if use_eval:
        trainer.save_metrics("eval", trainer.evaluate())
    if trainer.is_world_process_zero():
        dataset_path = Path(config.dataset_path)
        dataset_manifest_path = dataset_path.with_suffix(dataset_path.suffix + ".manifest.json")
        dataset_manifest = (
            json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
            if dataset_manifest_path.is_file()
            else {}
        )
        write_json_atomic(
            Path(config.output_dir) / "training_manifest.json",
            {
                "schema_version": 1,
                "artifact_type": "sft_model",
                "demonstration_protocol": dataset_manifest.get("demonstration_protocol"),
                "control_style": config.control_style,
                "config": asdict(config),
                "dataset": {
                    "path": str(dataset_path),
                    "rows": len(dataset),
                    "bytes": dataset_path.stat().st_size,
                    "sha256": sha256_file(dataset_path),
                },
                "dataset_manifest": (
                    {
                        "path": str(dataset_manifest_path),
                        "sha256": sha256_file(dataset_manifest_path),
                    }
                    if dataset_manifest_path.is_file()
                    else None
                ),
                "split": {
                    "train_rows": len(train_indices),
                    "validation_rows": len(eval_indices),
                    "seed": config.seed,
                },
                "best_model_checkpoint": trainer.state.best_model_checkpoint,
                **runtime_metadata(("torch", "transformers", "accelerate", "peft", "bitsandbytes")),
            },
        )


def _validate_dataset_manifest(
    dataset_path: str | Path,
    expected_control_style: str | None = None,
) -> None:
    path = Path(dataset_path)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"SFT dataset manifest is required for reproducible training: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("demonstration_protocol"):
        raise ValueError(f"SFT dataset manifest has no demonstration_protocol: {manifest_path}")
    if expected_control_style is None:
        return

    generator_style = (manifest.get("generator_config") or {}).get("control_style")
    if generator_style != expected_control_style:
        raise ValueError(
            "SFT dataset generator control style mismatch: "
            f"{generator_style!r} != {expected_control_style!r}"
        )
    dataset = manifest.get("dataset") or {}
    expected_rows = int(dataset.get("rows", 0))
    style_counts = dataset.get("control_styles")
    if style_counts != {expected_control_style: expected_rows}:
        raise ValueError(
            "SFT dataset manifest does not describe a single-style data set: "
            f"{style_counts!r}"
        )


def audit_sft_dataset(
    dataset_path: str | Path,
    expected_control_style: str,
) -> dict[str, int]:
    """Reject mixed-style or malformed SFT rows before model loading."""

    expected_style_line = f"Desired control style: {expected_control_style}."
    styles: Counter[str] = Counter()
    plant_types: Counter[str] = Counter()
    sample_kinds: Counter[str] = Counter()
    rows = 0

    with Path(dataset_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages") or []
            metadata = row.get("metadata") or {}
            if not messages or messages[-1].get("role") != "assistant":
                raise ValueError(f"SFT row {line_number} has no final assistant answer.")
            if metadata.get("schema_version") != 3:
                raise ValueError(f"SFT row {line_number} is not schema version 3.")
            style = str(metadata.get("control_style", ""))
            if style != expected_control_style:
                raise ValueError(
                    f"SFT row {line_number} control style {style!r} "
                    f"does not match {expected_control_style!r}."
                )
            if metadata.get("target_method") != "imc_style":
                raise ValueError(f"SFT row {line_number} has an unexpected target method.")
            if "target_metrics" not in metadata:
                raise ValueError(f"SFT row {line_number} has no target-response audit metrics.")

            user_messages = [
                str(message.get("content", ""))
                for message in messages
                if message.get("role") == "user"
            ]
            if not user_messages:
                raise ValueError(f"SFT row {line_number} has no user prompt.")
            for content in user_messages:
                style_lines = [
                    prompt_line.strip()
                    for prompt_line in content.splitlines()
                    if prompt_line.strip().startswith("Desired control style:")
                ]
                if style_lines != [expected_style_line]:
                    raise ValueError(
                        f"SFT row {line_number} has inconsistent prompt style lines: "
                        f"{style_lines!r}."
                    )

            styles[style] += 1
            plant_types[str((metadata.get("plant") or {}).get("plant_type"))] += 1
            sample_kinds[str(metadata.get("sample_kind"))] += 1
            rows += 1

    if rows == 0:
        raise ValueError(f"SFT dataset is empty: {dataset_path}")
    stats = {
        "rows": rows,
        expected_control_style: styles[expected_control_style],
        "first_order": plant_types["first_order"],
        "second_order": plant_types["second_order"],
        "initial": sample_kinds["initial"],
        "feedback": sample_kinds["feedback"],
    }
    print("SFT protocol audit: " + ", ".join(f"{key}={value}" for key, value in stats.items()))
    return stats


def _prepare_sft_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(
            f"SFT output directory is not empty: {path}. "
            "Move or remove the existing experiment before starting a new run."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_sft_tokenizer(model_name_or_path: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


class _SFTDataset:
    def __init__(self, path: str, tokenizer: Any, max_length: int) -> None:
        self.rows = []
        self.validation_strata: list[str] = []
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                messages = row.get("messages")
                if not messages or messages[-1].get("role") != "assistant":
                    raise ValueError(f"SFT row {line_number} must end with an assistant message.")
                self.rows.append(row)
                metadata = row.get("metadata") or {}
                plant_type = str((metadata.get("plant") or {}).get("plant_type", "unknown"))
                sample_kind = str(metadata.get("sample_kind", "unknown"))
                self.validation_strata.append(f"{plant_type}:{sample_kind}")
        if not self.rows:
            raise ValueError(f"SFT dataset is empty: {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return _encode_sft_messages(
            self.tokenizer,
            self.rows[index]["messages"],
            self.max_length,
        )

    def validate_lengths(self, batch_size: int = 128) -> dict[str, int]:
        lengths: list[int] = []
        longest_row = 0
        for start in range(0, len(self.rows), batch_size):
            batch_rows = self.rows[start : start + batch_size]
            texts = [_render_full_sft_text(self.tokenizer, row["messages"]) for row in batch_rows]
            encoded = self.tokenizer(
                texts,
                add_special_tokens=False,
                truncation=False,
                padding=False,
            )
            for offset, input_ids in enumerate(encoded["input_ids"]):
                length = len(input_ids)
                lengths.append(length)
                if length > lengths[longest_row]:
                    longest_row = start + offset

        ordered = sorted(lengths)
        stats = {
            "rows": len(ordered),
            "p50": _nearest_percentile(ordered, 0.50),
            "p95": _nearest_percentile(ordered, 0.95),
            "p99": _nearest_percentile(ordered, 0.99),
            "max": ordered[-1],
        }
        print(
            "SFT token preflight: " + ", ".join(f"{name}={value}" for name, value in stats.items())
        )
        if stats["max"] > self.max_length:
            raise ValueError(
                f"SFT row {longest_row + 1} uses {stats['max']} tokens, "
                f"exceeding max_length={self.max_length}."
            )
        return stats


def _render_full_sft_text(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    if getattr(tokenizer, "chat_template", None) and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    prompt = messages_to_prompt(tokenizer, messages[:-1])
    eos = getattr(tokenizer, "eos_token", None) or ""
    return prompt + str(messages[-1]["content"]) + eos


def _nearest_percentile(ordered_values: list[int], quantile: float) -> int:
    if not ordered_values:
        raise ValueError("Cannot calculate a percentile for an empty sequence.")
    index = int(round((len(ordered_values) - 1) * quantile))
    return ordered_values[index]


def _encode_sft_messages(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    max_length: int,
) -> dict[str, list[int]]:
    if max_length <= 0:
        raise ValueError("max_length must be positive.")

    if getattr(tokenizer, "chat_template", None) and hasattr(tokenizer, "apply_chat_template"):
        prompt_ids = _as_token_list(
            tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        full_ids = _as_token_list(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        )
    else:
        prompt = messages_to_prompt(tokenizer, messages[:-1])
        assistant = messages[-1]
        answer = str(assistant["content"])
        eos = getattr(tokenizer, "eos_token", None) or ""
        prompt_ids = _tokenize_text(tokenizer, prompt)
        full_ids = _tokenize_text(tokenizer, prompt + answer + eos)

    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Tokenizer chat template does not keep the generation prompt as a prefix "
            "of the completed conversation; answer-only masking would be ambiguous."
        )

    answer_ids = full_ids[len(prompt_ids) :]
    if not answer_ids:
        raise ValueError("The assistant answer produced no supervised tokens.")
    if len(full_ids) > max_length:
        raise ValueError(
            f"SFT sequence uses {len(full_ids)} tokens, exceeding max_length={max_length}."
        )

    labels = [-100] * len(prompt_ids) + list(answer_ids)
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def _split_indices(
    dataset_size: int,
    validation_fraction: float,
    seed: int,
    strata: list[str] | None = None,
) -> tuple[list[int], list[int]]:
    if dataset_size < 2 or validation_fraction <= 0.0:
        return list(range(dataset_size)), []
    if validation_fraction >= 1.0:
        raise ValueError("validation_fraction must be less than 1.")

    indices = list(range(dataset_size))
    if strata is not None:
        if len(strata) != dataset_size:
            raise ValueError("strata length must match dataset_size.")
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, stratum in enumerate(strata):
            grouped[stratum].append(index)

        rng = random.Random(seed)
        train_indices: list[int] = []
        validation_indices: list[int] = []
        for stratum in sorted(grouped):
            indices = grouped[stratum]
            rng.shuffle(indices)
            validation_size = int(round(len(indices) * validation_fraction))
            if len(indices) > 1:
                validation_size = max(1, min(validation_size, len(indices) - 1))
            else:
                validation_size = 0
            validation_indices.extend(indices[:validation_size])
            train_indices.extend(indices[validation_size:])
        rng.shuffle(train_indices)
        rng.shuffle(validation_indices)
        return train_indices, validation_indices

    random.Random(seed).shuffle(indices)
    validation_size = max(1, int(round(dataset_size * validation_fraction)))
    validation_size = min(validation_size, dataset_size - 1)
    return indices[validation_size:], indices[:validation_size]


def _tokenize_text(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, truncation=False, padding=False)
    return _as_token_list(encoded["input_ids"])


def _as_token_list(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token) for token in value]


def _causal_lm_collator(
    features: list[dict[str, list[int]]],
    pad_token_id: int,
) -> dict[str, Any]:
    import torch
    from torch.nn.utils.rnn import pad_sequence

    input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in features]
    attention = [torch.tensor(item["attention_mask"], dtype=torch.long) for item in features]
    labels = [torch.tensor(item["labels"], dtype=torch.long) for item in features]
    return {
        "input_ids": pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=pad_token_id,
        ),
        "attention_mask": pad_sequence(
            attention,
            batch_first=True,
            padding_value=0,
        ),
        "labels": pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        ),
    }
