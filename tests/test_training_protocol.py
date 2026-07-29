import json
from pathlib import Path

import pytest

from llmpidtuner.training.config import GRPOTrainConfig
from llmpidtuner.training.grpo import _validate_source_model_protocol
from llmpidtuner.training.sft import _validate_dataset_manifest


PROTOCOL = "perturbed_imc_delay_stratified_v1"


def test_sft_requires_dataset_demonstration_protocol(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    manifest = dataset.with_suffix(".jsonl.manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "demonstration_protocol": PROTOCOL,
                "generator_config": {"control_style": "balanced"},
                "dataset": {"rows": 0, "control_styles": {"balanced": 0}},
            }
        ),
        encoding="utf-8",
    )

    _validate_dataset_manifest(dataset)
    _validate_dataset_manifest(dataset, "balanced")
    manifest.write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(ValueError, match="no demonstration_protocol"):
        _validate_dataset_manifest(dataset)


def test_grpo_requires_matching_sft_protocol(tmp_path: Path) -> None:
    model_dir = tmp_path / "sft"
    model_dir.mkdir()
    (model_dir / "training_manifest.json").write_text(
        json.dumps({"demonstration_protocol": PROTOCOL, "control_style": "balanced"}),
        encoding="utf-8",
    )
    config = GRPOTrainConfig(
        model_name_or_path=str(model_dir),
        output_dir=str(tmp_path / "grpo"),
        demonstration={"method": "frozen", "protocol_id": PROTOCOL},
    )
    _validate_source_model_protocol(config)

    (model_dir / "training_manifest.json").write_text(
        json.dumps({"demonstration_protocol": "old-protocol"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="protocol mismatch"):
        _validate_source_model_protocol(config)

    (model_dir / "training_manifest.json").write_text(
        json.dumps({"demonstration_protocol": PROTOCOL, "control_style": "aggressive"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="control style mismatch"):
        _validate_source_model_protocol(config)
