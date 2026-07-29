from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml

from llmpidtuner.config import load_case_config
from llmpidtuner.metrics import ControlSystemAnalysis


def _analysis() -> ControlSystemAnalysis:
    time = np.arange(11, dtype=float)
    setpoint = np.ones(11, dtype=float)
    output = np.asarray([0.0, 0.4, 0.9, 1.3, 0.8, 1.15, 0.95, 1.02, 1.0, 1.0, 1.0])
    return ControlSystemAnalysis.from_arrays(time, setpoint, output)


def test_prompt_variants_render_one_structured_diagnosis() -> None:
    analysis = _analysis()
    features = analysis.extract_features()
    full = analysis.generate_description("full")
    kpi3 = analysis.generate_description("kpi3")
    numeric8 = analysis.generate_description("numeric8")

    assert f"{features.oscillation_count} oscillations" in full
    assert "Overshoot:" in kpi3
    assert "Steady-State Error" in kpi3
    assert "Oscillations:" not in kpi3
    assert "Attenuation Ratio:" not in kpi3
    for label in (
        "Overshoot (%)",
        "Oscillation count",
        "Attenuation ratio",
        "Oscillation period (s)",
        "Time to 63.2% of setpoint (s)",
        "Settling time (+/-5%) (s)",
        "Steady-state error (%)",
    ):
        assert label in numeric8
    assert "indicating" not in numeric8.lower()


def _response_only(text: str) -> str:
    return "\n".join(
        line
        for line in text.splitlines()
        if not line.startswith(("Recommended control style:", "Suggested PID parameters:"))
    )


def test_frozen_matrix_shares_one_source_and_varies_only_render_or_style() -> None:
    root = Path("cases/demonstrations/perturbed_imc_delay_stratified")
    for system in ("first_order", "second_order"):
        paths = {
            "full": root / "balanced" / "full" / f"{system}.manifest.yaml",
            "kpi3": root / "balanced" / "kpi3" / f"{system}.manifest.yaml",
            "numeric8": root / "balanced" / "numeric8" / f"{system}.manifest.yaml",
            "aggressive": root / "aggressive" / "full" / f"{system}.manifest.yaml",
            "conservative": root / "conservative" / "full" / f"{system}.manifest.yaml",
        }
        metadata = {
            key: yaml.safe_load(path.read_text(encoding="utf-8"))
            for key, path in paths.items()
        }
        assert len({item["source"]["sha256"] for item in metadata.values()}) == 1
        assert metadata["full"]["prompt_variant"] == "full"
        assert metadata["kpi3"]["prompt_variant"] == "kpi3"
        assert metadata["numeric8"]["prompt_variant"] == "numeric8"

        texts = {
            style: (root / style / "full" / f"{system}.txt").read_text(encoding="utf-8")
            for style in ("balanced", "aggressive", "conservative")
        }
        assert len({_response_only(text) for text in texts.values()}) == 1
        assert len(set(texts.values())) == 3

        for item in metadata.values():
            prompt = Path(item["prompt"]["path"]).read_bytes()
            assert hashlib.sha256(prompt).hexdigest() == item["prompt"]["sha256"]


def test_api_ablation_and_style_matrix_is_complete() -> None:
    conditions = ("kpi3", "numeric8", "aggressive", "conservative")
    expected = {
        f"{system}_100_{provider}_{condition}.yaml"
        for system in ("first_order", "second_order")
        for provider in ("deepseek_v4_flash", "qwen3_7_plus")
        for condition in conditions
    }
    paths = [Path("cases/eval") / name for name in sorted(expected)]
    assert all(path.is_file() for path in paths)

    for path in paths:
        config = load_case_config(path)
        assert config.batch is not None
        assert config.batch["cases_path"].startswith(
            "cases/protocol/perturbed_imc_delay_stratified/"
        )
        assert config.demonstration is not None
        assert config.prompt_variant == config.demonstration["prompt_variant"]
