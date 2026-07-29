import hashlib
from pathlib import Path

import yaml

import pytest

from llmpidtuner.config import CaseConfig
from llmpidtuner.demonstrations import (
    generate_demonstration_from_spec,
    imc_pid_tuning_first_order,
    imc_pid_tuning_second_order,
)
from llmpidtuner.experiment_protocol import PROTOCOL_ID
from llmpidtuner.models import FirstOrderPlant, PIDParams, SimulationSettings
from llmpidtuner.runner import run_case


def _frozen_spec(system: str, style: str = "balanced", variant: str = "full") -> dict[str, object]:
    root = f"cases/demonstrations/perturbed_imc_delay_stratified/{style}/{variant}"
    return {
        "method": "frozen",
        "system": system,
        "protocol_id": PROTOCOL_ID,
        "control_style": style,
        "prompt_variant": variant,
        "path": f"{root}/{system}.txt",
        "manifest_path": f"{root}/{system}.manifest.yaml",
    }


def test_imc_first_order_formula_uses_rivera_derivative_time() -> None:
    pid = imc_pid_tuning_first_order(k=0.648, t=112.51, time_delay=80, lambda_value=10)
    kp = (1 / 0.648) * ((112.51 + 80 / 2) / (10 + 80 / 2))
    ti = 112.51 + 80 / 2
    td = (112.51 * 80) / (2 * 112.51 + 80)
    assert pid.kp == kp
    assert pid.ki == kp / ti
    assert pid.kd == kp * td


def test_imc_second_order_formula() -> None:
    pid = imc_pid_tuning_second_order(
        k=2.687, tau1=8.79, tau2=42.25, time_delay=20, lambda_value=10
    )
    kp = (8.79 + 42.25) / (2.687 * (10 + 20))
    ti = 8.79 + 42.25
    td = (8.79 * 42.25) / (8.79 + 42.25)
    assert pid == PIDParams(kp, kp / ti, kp * td)


@pytest.mark.parametrize("system", ["first_order", "second_order"])
def test_frozen_demonstration_loads_and_verifies_hashes(system: str) -> None:
    text = generate_demonstration_from_spec(
        _frozen_spec(system),
        initial_pid=PIDParams(1.0, 0.1, 0.01),
        simulation=SimulationSettings(),
    )
    assert text is not None
    assert text.count("Experiment ") == 10
    assert "Observed process dead time:" in text
    assert "Suggested PID parameters:" in text


def test_frozen_demonstration_rejects_style_mismatch() -> None:
    spec = _frozen_spec("first_order")
    spec["control_style"] = "aggressive"
    with pytest.raises(ValueError, match="control_style"):
        generate_demonstration_from_spec(
            spec,
            initial_pid=PIDParams(1.0, 0.1, 0.01),
            simulation=SimulationSettings(),
        )


def test_removed_runtime_methods_fail_fast() -> None:
    for method in ("imc", "generated", "default"):
        with pytest.raises(ValueError, match="no longer a runtime method"):
            generate_demonstration_from_spec({"method": method, "system": "first_order"})


def test_legacy_default_is_explicit_and_warns() -> None:
    with pytest.warns(UserWarning, match="manually curated historical prompt"):
        text = generate_demonstration_from_spec(
            {"method": "legacy_default", "system": "first_order"}
        )
    assert text is not None
    assert text.count("Experiment ") == 10


def test_committed_protocol_assets_have_valid_hashes() -> None:
    root = Path("cases/demonstrations/perturbed_imc_delay_stratified")
    manifests = sorted(root.rglob("*.manifest.yaml"))
    assert len(manifests) == 10
    for path in manifests:
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
        prompt = Path(manifest["prompt"]["path"]).read_bytes()
        source = Path(manifest["source"]["path"]).read_bytes()
        assert hashlib.sha256(prompt).hexdigest() == manifest["prompt"]["sha256"]
        assert hashlib.sha256(source).hexdigest() == manifest["source"]["sha256"]


def test_frozen_demonstration_run_writes_protocol_metadata(tmp_path: Path) -> None:
    config = CaseConfig(
        name="frozen_demo_case",
        system="first_order",
        mode="dry_run",
        output_dir=str(tmp_path),
        simulation=SimulationSettings(),
        first_order=FirstOrderPlant(k=0.65, t=112),
        demonstration=_frozen_spec("first_order"),
    )
    output_dir = run_case(config)
    metadata = (output_dir / "demonstration_metadata.yaml").read_text(encoding="utf-8")
    assert f"demonstration_protocol: {PROTOCOL_ID}" in metadata
    assert (output_dir / "demonstration_prompt.txt").exists()
