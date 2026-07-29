import json
from collections import Counter
from pathlib import Path

import pytest

from llmpidtuner.training.sft import (
    _encode_sft_messages,
    _prepare_sft_output_dir,
    _split_indices,
    audit_sft_dataset,
)


class FakeTokenizer:
    chat_template = "fake"
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking=True):
        text = ""
        for message in messages:
            text += f"<{message['role']}>"
            if message["role"] == "assistant":
                reasoning = message.get("reasoning_content", "")
                if enable_thinking and reasoning:
                    text += f"<think>{reasoning}</think>"
                elif not enable_thinking:
                    text += "<think></think>"
                text += f"{message['content']}<eos>"
            else:
                text += message["content"] + "<end>"
        if add_generation_prompt:
            text += "<assistant>"
            if not enable_thinking:
                text += "<think></think>"
        return [ord(character) for character in text] if tokenize else text


def test_sft_encoding_masks_prompt_and_keeps_pid_answer_without_thinking() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "P:1; I:0.1; D:0.01"},
    ]
    tokenizer = FakeTokenizer()
    encoded = _encode_sft_messages(tokenizer, messages, max_length=256)

    first_supervised = next(index for index, label in enumerate(encoded["labels"]) if label != -100)
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert first_supervised == len(prompt_ids)
    supervised = "".join(chr(token) for token in encoded["labels"] if token != -100)
    assert "<think>" not in supervised
    assert "P:1; I:0.1; D:0.01" in supervised
    assert supervised.endswith("<eos>")


def test_sft_encoding_rejects_an_overlength_sequence() -> None:
    messages = [
        {"role": "system", "content": "s" * 200},
        {"role": "user", "content": "u" * 200},
        {
            "role": "assistant",
            "content": "P:1; I:0.1; D:0.01",
        },
    ]
    with pytest.raises(ValueError, match="SFT sequence uses .* exceeding max_length=96"):
        _encode_sft_messages(FakeTokenizer(), messages, max_length=96)


def test_validation_split_is_reproducible_and_disjoint() -> None:
    train_a, validation_a = _split_indices(100, validation_fraction=0.1, seed=42)
    train_b, validation_b = _split_indices(100, validation_fraction=0.1, seed=42)

    assert (train_a, validation_a) == (train_b, validation_b)
    assert len(validation_a) == 10
    assert set(train_a).isdisjoint(validation_a)
    assert sorted(train_a + validation_a) == list(range(100))


def test_validation_split_preserves_each_training_stratum() -> None:
    strata = (
        ["first_order:initial"] * 40
        + ["first_order:feedback"] * 30
        + ["second_order:initial"] * 20
        + ["second_order:feedback"] * 10
    )
    train, validation = _split_indices(
        len(strata),
        validation_fraction=0.1,
        seed=42,
        strata=strata,
    )

    assert set(train).isdisjoint(validation)
    assert sorted(train + validation) == list(range(len(strata)))
    assert Counter(strata[index] for index in validation) == Counter(
        {
            "first_order:initial": 4,
            "first_order:feedback": 3,
            "second_order:initial": 2,
            "second_order:feedback": 1,
        }
    )


def test_sft_output_directory_accepts_only_missing_or_empty_paths(tmp_path: Path) -> None:
    output_dir = tmp_path / "sft-output"

    assert _prepare_sft_output_dir(output_dir) == output_dir
    assert output_dir.is_dir()
    assert _prepare_sft_output_dir(output_dir) == output_dir

    (output_dir / "config.json").write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        _prepare_sft_output_dir(output_dir)


def test_sft_protocol_audit_rejects_mixed_prompt_style(tmp_path: Path) -> None:
    dataset = tmp_path / "sft.jsonl"
    row = {
        "messages": [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": "Desired control style: balanced.\nCurrent response.",
            },
            {"role": "assistant", "content": "P:1; I:0.1; D:0.01"},
        ],
        "metadata": {
            "schema_version": 3,
            "sample_kind": "initial",
            "control_style": "balanced",
            "plant": {"plant_type": "first_order"},
            "target_method": "imc_style",
            "target_metrics": {},
        },
    }
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")

    assert audit_sft_dataset(dataset, "balanced")["balanced"] == 1

    row["messages"][1]["content"] = (
        "Desired control style: aggressive.\nCurrent response."
    )
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent prompt style"):
        audit_sft_dataset(dataset, "balanced")
