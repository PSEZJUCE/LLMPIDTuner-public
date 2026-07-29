from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from llmpidtuner.default_demo import get_default_demonstration
from llmpidtuner.metrics import (
    ControlSystemAnalysis,
    ResponseFeatures,
    normalize_prompt_variant,
    render_response_features,
)
from llmpidtuner.models import FirstOrderPlant, PIDParams, SecondOrderPlant, SimulationSettings
from llmpidtuner.simulation import FirstOrderDelaySimulator, SecondOrderDelaySimulator


CURRENT_PID = PIDParams(kp=1.0, ki=0.1, kd=0.01)



@dataclass(frozen=True)
class SimulatedDemonstrationRecord:
    example_id: int
    initial_pid: PIDParams
    iae: float
    time_delay: float
    features: ResponseFeatures
    plant: FirstOrderPlant | SecondOrderPlant


def imc_pid_tuning_first_order(
    k: float,
    t: float,
    time_delay: float = 1.0,
    lambda_value: float = 10.0,
) -> PIDParams:
    kp = (1 / k) * ((t + time_delay / 2) / (lambda_value + time_delay / 2))
    ti = t + time_delay / 2
    td = (t * time_delay) / (2 * t + time_delay)
    return PIDParams(kp=kp, ki=kp / ti, kd=kp * td)


def imc_pid_tuning_second_order(
    k: float,
    tau1: float,
    tau2: float,
    time_delay: float = 1.0,
    lambda_value: float = 10.0,
) -> PIDParams:
    kp = (tau1 + tau2) / (k * (lambda_value + time_delay))
    ti = tau1 + tau2
    td = (tau1 * tau2) / (tau1 + tau2)
    return PIDParams(kp=kp, ki=kp / ti, kd=kp * td)


def generate_demonstration_from_spec(
    data: dict[str, Any],
    initial_pid: PIDParams | None = None,
    simulation: SimulationSettings | None = None,
) -> str | None:
    if not data:
        return None
    method = data.get("method")
    if method == "legacy_default":
        warnings.warn(
            "legacy_default reproduces a manually curated historical prompt and must not be "
            "used for current training or benchmark evaluation.",
            UserWarning,
            stacklevel=2,
        )
        return get_default_demonstration(data["system"])
    if method == "frozen":
        return load_frozen_demonstration(
            data,
            initial_pid=initial_pid or CURRENT_PID,
            simulation=simulation or SimulationSettings(),
        )
    if method in {"generated", "imc", "default"}:
        raise ValueError(
            f"Demonstration method '{method}' is no longer a runtime method. "
            "Use 'frozen' for current experiments or 'legacy_default' for historical replay."
        )
    raise ValueError(f"Unsupported demonstration method: {method}")


def load_frozen_demonstration(
    data: dict[str, Any],
    *,
    initial_pid: PIDParams,
    simulation: SimulationSettings,
) -> str:
    if data.get("paths"):
        system_paths = data["paths"].get(data["system"])
        if not system_paths:
            raise ValueError(f"No frozen demonstration path configured for {data['system']}.")
        data = {**data, **system_paths}
    prompt_path = Path(data["path"])
    manifest_path = Path(data.get("manifest_path") or prompt_path.with_suffix(".manifest.yaml"))
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != "frozen_demonstration":
        raise ValueError(f"Not a frozen demonstration manifest: {manifest_path}")

    _require_equal("protocol_id", manifest["demonstration_protocol"], str(data["protocol_id"]))
    _require_equal(
        "prompt_variant",
        normalize_prompt_variant(manifest.get("prompt_variant")),
        normalize_prompt_variant(data.get("prompt_variant")),
    )
    if data.get("system"):
        _require_equal("system", manifest["system"], data["system"])

    if "control_style" in manifest:
        if data.get("control_style"):
            _require_equal(
                "control_style",
                manifest["control_style"],
                str(data["control_style"]),
            )
        prompt_bytes = prompt_path.read_bytes()
        _require_equal(
            "prompt.sha256",
            _sha256_bytes(prompt_bytes),
            manifest["prompt"]["sha256"],
        )
        source_path = Path(manifest["source"]["path"])
        _require_equal(
            "source.sha256",
            _sha256_bytes(source_path.read_bytes()),
            manifest["source"]["sha256"],
        )
        return prompt_bytes.decode("utf-8")
    _require_close("lambda_value", float(manifest["lambda_value"]), float(data["lambda_value"]))
    for key, value in manifest["initial_pid"].items():
        _require_close(f"initial_pid.{key}", float(value), float(getattr(initial_pid, key)))
    for key in ("setpoint", "sim_time", "time_delay"):
        _require_close(
            f"simulation.{key}",
            float(manifest["simulation"][key]),
            float(getattr(simulation, key)),
        )
    _require_equal(
        "simulation.num_points",
        int(manifest["simulation"]["num_points"]),
        simulation.num_points,
    )

    prompt_bytes = prompt_path.read_bytes()
    _require_equal("prompt.sha256", _sha256_bytes(prompt_bytes), manifest["prompt"]["sha256"])
    examples_path = Path(manifest["examples"]["path"])
    _require_equal(
        "examples.sha256",
        _sha256_bytes(examples_path.read_bytes()),
        manifest["examples"]["sha256"],
    )
    return prompt_bytes.decode("utf-8")


def demonstration_protocol_id(data: dict[str, Any] | None) -> str | None:
    if not data:
        return None
    return str(data["protocol_id"]) if data.get("method") == "frozen" else None


def generate_simulated_demonstration(
    data: dict[str, Any],
    initial_pid: PIDParams,
    simulation: SimulationSettings,
) -> str:
    system = str(data["system"])
    examples_path_value = data.get("examples_path")
    if not examples_path_value:
        raise ValueError(
            "The historical demonstration builder requires an explicit examples_path."
        )
    examples_path = Path(examples_path_value)
    lambda_value = float(data.get("lambda_value", 10.0))
    prompt_variant = normalize_prompt_variant(data.get("prompt_variant"))
    use_cache = bool(data.get("cache", True))
    cache_dir = Path(data.get("cache_dir", ".cache/demonstrations"))
    cache_key = _generated_cache_key(
        system=system,
        examples_path=examples_path,
        lambda_value=lambda_value,
        initial_pid=initial_pid,
        simulation=simulation,
        prompt_variant=prompt_variant,
    )
    cache_path = cache_dir / f"{cache_key}.txt"
    if use_cache and cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    text = build_simulated_demonstration(
        system=system,
        examples=_load_examples(examples_path),
        initial_pid=initial_pid,
        simulation=simulation,
        lambda_value=lambda_value,
        prompt_variant=prompt_variant,
    )
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8", newline="\n")
    return text


def build_simulated_demonstration(
    *,
    system: str,
    examples: list[dict[str, Any]],
    initial_pid: PIDParams,
    simulation: SimulationSettings,
    lambda_value: float,
    prompt_variant: str = "full",
) -> str:
    records = build_simulated_demonstration_records(
        system=system,
        examples=examples,
        initial_pid=initial_pid,
        simulation=simulation,
    )
    return render_simulated_demonstration(
        records,
        prompt_variant=prompt_variant,
        lambda_value=lambda_value,
    )


def build_simulated_demonstration_records(
    *,
    system: str,
    examples: list[dict[str, Any]],
    initial_pid: PIDParams,
    simulation: SimulationSettings,
) -> list[SimulatedDemonstrationRecord]:
    records: list[SimulatedDemonstrationRecord] = []
    for fallback_id, example in enumerate(examples, start=1):
        plant = _plant_from_example(system, example)
        if isinstance(plant, FirstOrderPlant):
            result = FirstOrderDelaySimulator(plant, initial_pid, simulation).run()
        else:
            result = SecondOrderDelaySimulator(plant, initial_pid, simulation).run()
        features = ControlSystemAnalysis.from_arrays(
            result.results_array[:, 0],
            result.results_array[:, 1],
            result.results_array[:, 2],
            filename=f"generated_demo_{fallback_id}",
        ).extract_features()
        records.append(
            SimulatedDemonstrationRecord(
                example_id=int(example.get("id", fallback_id)),
                initial_pid=initial_pid,
                iae=result.iae,
                time_delay=simulation.time_delay,
                features=features,
                plant=plant,
            )
        )
    return records


def render_simulated_demonstration(
    records: list[SimulatedDemonstrationRecord],
    *,
    prompt_variant: str = "full",
    lambda_value: float = 10.0,
) -> str:
    variant = normalize_prompt_variant(prompt_variant)
    sections: list[str] = []
    for record in records:
        suggested = _suggested_pid(
            record.plant,
            time_delay=record.time_delay,
            lambda_value=lambda_value,
        )
        sections.append(
            _format_simulated_experiment(
                example_id=record.example_id,
                initial_pid=record.initial_pid,
                iae=record.iae,
                time_delay=record.time_delay,
                description=render_response_features(record.features, variant),
                suggested=suggested,
            )
        )
    return "\n\n".join(sections) + "\n"


def _suggested_pid(
    plant: FirstOrderPlant | SecondOrderPlant,
    *,
    time_delay: float,
    lambda_value: float,
) -> PIDParams:
    if isinstance(plant, FirstOrderPlant):
        return imc_pid_tuning_first_order(
            plant.k,
            plant.t,
            time_delay=time_delay,
            lambda_value=lambda_value,
        )
    return imc_pid_tuning_second_order(
        plant.k,
        plant.tau1,
        plant.tau2,
        time_delay=time_delay,
        lambda_value=lambda_value,
    )


def _format_simulated_experiment(
    example_id: int,
    initial_pid: PIDParams,
    iae: float,
    time_delay: float,
    description: str,
    suggested: PIDParams,
) -> str:
    return "\n".join(
        [
            f"Experiment {example_id}:",
            (
                "Current PID parameters and IAE: "
                f"Kp={initial_pid.kp:.3f}, Ki={initial_pid.ki:.3f}, "
                f"Kd={initial_pid.kd:.3f}, Integral Absolute Error (IAE): {iae:.2f}"
            ),
            f"Process time delay: {time_delay:.2f} seconds.",
            description.strip(),
            (
                "Suggested new PID parameters: "
                f"P={suggested.kp:.2f}, I={suggested.ki:.3f}, D={suggested.kd:.2f}"
            ),
        ]
    )


def _load_examples(path: Path) -> list[dict[str, Any]]:
    return list(yaml.safe_load(path.read_text(encoding="utf-8"))["examples"])


def _plant_from_example(
    system: str,
    example: dict[str, Any],
) -> FirstOrderPlant | SecondOrderPlant:
    if system == "first_order":
        return FirstOrderPlant(k=float(example["k"]), t=float(example["t"]))
    if system == "second_order":
        return SecondOrderPlant(
            k=float(example["k"]),
            tau1=float(example["tau1"]),
            tau2=float(example["tau2"]),
        )
    raise ValueError(f"Unsupported generated demonstration system: {system}")


def _generated_cache_key(
    system: str,
    examples_path: Path,
    lambda_value: float,
    initial_pid: PIDParams,
    simulation: SimulationSettings,
    prompt_variant: str,
) -> str:
    payload = {
        "version": "generated-v3",
        "system": system,
        "prompt_variant": normalize_prompt_variant(prompt_variant),
        "examples_path": str(examples_path),
        "examples_hash": hashlib.sha256(examples_path.read_bytes()).hexdigest(),
        "lambda_value": lambda_value,
        "initial_pid": {
            "kp": initial_pid.kp,
            "ki": initial_pid.ki,
            "kd": initial_pid.kd,
        },
        "simulation": {
            "setpoint": simulation.setpoint,
            "sim_time": simulation.sim_time,
            "num_points": simulation.num_points,
            "time_delay": simulation.time_delay,
        },
    }
    payload_text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload_text.encode("utf-8")).hexdigest()


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"Frozen demonstration mismatch for {name}: {actual!r} != {expected!r}")


def _require_close(name: str, actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"Frozen demonstration mismatch for {name}: {actual!r} != {expected!r}")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
