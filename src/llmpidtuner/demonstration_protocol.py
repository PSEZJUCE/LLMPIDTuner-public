from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import qmc

from llmpidtuner.demonstrations import (
    build_simulated_demonstration,
    build_simulated_demonstration_records,
    render_simulated_demonstration,
)
from llmpidtuner.metrics import normalize_prompt_variant
from llmpidtuner.models import (
    FirstOrderPlant,
    PIDParams,
    SecondOrderPlant,
    SimulationSettings,
)
from llmpidtuner.simulation import FirstOrderDelaySimulator, SecondOrderDelaySimulator


def build_demonstration_protocol(
    config_path: str | Path,
    *,
    check: bool = False,
    force: bool = False,
) -> list[Path]:
    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if config.get("protocols"):
        return _build_protocol_matrix(config_file, config, check=check, force=force)
    protocol_id = str(config["protocol_id"])
    initial_pid = _pid_from_dict(config["initial_pid"])
    simulation = _simulation_from_dict(config["simulation"])
    lambda_value = float(config["lambda_value"])

    desired: dict[Path, bytes] = {}
    for system in ("first_order", "second_order"):
        system_config = config["systems"][system]
        examples, design = _generate_examples(
            system,
            system_config,
            initial_pid=initial_pid,
            simulation=simulation,
        )
        examples_path = Path(system_config["examples_path"])
        prompt_path = Path(system_config["prompt_path"])
        manifest_path = Path(system_config["manifest_path"])

        examples_payload = {
            "system": system,
            "demonstration_protocol": protocol_id,
            "design": design,
            "examples": examples,
        }
        examples_bytes = _yaml_bytes(examples_payload)
        prompt_text = build_simulated_demonstration(
            system=system,
            examples=examples,
            initial_pid=initial_pid,
            simulation=simulation,
            lambda_value=lambda_value,
        )
        prompt_bytes = _text_bytes(prompt_text)
        response_peaks = _response_peaks(system, examples, initial_pid, simulation)
        manifest = {
            "schema_version": 1,
            "artifact_type": "frozen_demonstration",
            "demonstration_protocol": protocol_id,
            "system": system,
            "lambda_value": lambda_value,
            "initial_pid": asdict(initial_pid),
            "simulation": {
                "setpoint": simulation.setpoint,
                "sim_time": simulation.sim_time,
                "num_points": simulation.num_points,
                "time_delay": simulation.time_delay,
            },
            "design": design,
            "response_peaks": response_peaks,
            "examples": {
                "path": examples_path.as_posix(),
                "count": len(examples),
                "sha256": _sha256(examples_bytes),
            },
            "prompt": {
                "path": prompt_path.as_posix(),
                "sha256": _sha256(prompt_bytes),
            },
            "generator": {
                "config_path": config_file.as_posix(),
                "config_sha256": _sha256(config_file.read_bytes()),
                "python_version": platform.python_version(),
                "package_versions": {
                    name: metadata.version(name)
                    for name in ("numpy", "scipy", "PyYAML", "llmpidtuner")
                },
            },
        }
        desired[examples_path] = examples_bytes
        desired[prompt_path] = prompt_bytes
        desired[manifest_path] = _yaml_bytes(manifest)

    if check:
        _check_artifacts(desired)
        return list(desired)
    _write_artifacts(desired, force=force)
    return list(desired)


def _build_protocol_matrix(
    config_file: Path,
    config: dict[str, Any],
    *,
    check: bool,
    force: bool,
) -> list[Path]:
    initial_pid = _pid_from_dict(config["initial_pid"])
    simulation = _simulation_from_dict(config["simulation"])
    desired: dict[Path, bytes] = {}

    for system in ("first_order", "second_order"):
        system_config = config["systems"][system]
        protocols = _protocol_entries(config, system_config)
        examples, design = _generate_examples(
            system,
            system_config,
            initial_pid=initial_pid,
            simulation=simulation,
        )
        examples_path = Path(system_config["examples_path"])
        examples_payload = {
            "system": system,
            "demonstration_protocol": protocols[0][1]["protocol_id"],
            "demonstration_protocols": {
                key: entry["protocol_id"] for key, entry in protocols
            },
            "design": design,
            "examples": examples,
        }
        examples_bytes = _yaml_bytes(examples_payload)
        records = build_simulated_demonstration_records(
            system=system,
            examples=examples,
            initial_pid=initial_pid,
            simulation=simulation,
        )
        source_bytes = _text_bytes(
            json.dumps(
                [asdict(record) for record in records],
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        response_peaks = _response_peaks(system, examples, initial_pid, simulation)
        desired[examples_path] = examples_bytes

        for artifact_key, entry in protocols:
            prompt_path = Path(entry["prompt_path"])
            manifest_path = Path(entry["manifest_path"])
            prompt_variant = normalize_prompt_variant(entry.get("prompt_variant"))
            lambda_value = float(entry["lambda_value"])
            prompt_bytes = _text_bytes(
                render_simulated_demonstration(
                    records,
                    prompt_variant=prompt_variant,
                    lambda_value=lambda_value,
                )
            )
            manifest = {
                "schema_version": 2,
                "artifact_type": "frozen_demonstration",
                "demonstration_protocol": str(entry["protocol_id"]),
                "parent_protocol": entry.get("parent_protocol"),
                "artifact_key": artifact_key,
                "prompt_variant": prompt_variant,
                "system": system,
                "lambda_value": lambda_value,
                "initial_pid": asdict(initial_pid),
                "simulation": {
                    "setpoint": simulation.setpoint,
                    "sim_time": simulation.sim_time,
                    "num_points": simulation.num_points,
                    "time_delay": simulation.time_delay,
                },
                "design": design,
                "response_peaks": response_peaks,
                "structured_source_sha256": _sha256(source_bytes),
                "examples": {
                    "path": examples_path.as_posix(),
                    "count": len(examples),
                    "sha256": _sha256(examples_bytes),
                },
                "prompt": {
                    "path": prompt_path.as_posix(),
                    "sha256": _sha256(prompt_bytes),
                },
                "generator": {
                    "config_path": config_file.as_posix(),
                    "config_sha256": _sha256(config_file.read_bytes()),
                    "python_version": platform.python_version(),
                    "package_versions": {
                        name: metadata.version(name)
                        for name in ("numpy", "scipy", "PyYAML", "llmpidtuner")
                    },
                },
            }
            desired[prompt_path] = prompt_bytes
            desired[manifest_path] = _yaml_bytes(manifest)

    if check:
        _check_artifacts(desired)
        return list(desired)
    _write_artifacts(desired, force=force)
    return list(desired)


def _protocol_entries(
    config: dict[str, Any],
    system_config: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    if config.get("protocols"):
        artifacts = system_config["artifacts"]
        return [
            (
                key,
                {
                    **protocol,
                    **artifacts[key],
                },
            )
            for key, protocol in config["protocols"].items()
        ]
    return [
        (
            "full",
            {
                "protocol_id": str(config["protocol_id"]),
                "prompt_variant": config.get("prompt_variant", "full"),
                "lambda_value": float(config["lambda_value"]),
                "prompt_path": system_config["prompt_path"],
                "manifest_path": system_config["manifest_path"],
            },
        )
    ]


def _generate_examples(
    system: str,
    config: dict[str, Any],
    *,
    initial_pid: PIDParams,
    simulation: SimulationSettings,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    count = int(config.get("count", 10))
    seed = int(config["seed"])
    ranges = config["ranges"]
    test_path = Path(config["excluded_test_plants_path"])
    threshold = float(config["minimum_test_distance"])
    dimensions = 2 if system == "first_order" else 3
    sample = qmc.LatinHypercube(d=dimensions, seed=seed).random(count)

    if system == "first_order":
        examples = _first_order_examples(sample, ranges)
        permutation_metadata: dict[str, Any] | None = None
    elif system == "second_order":
        permutations = config["accepted_permutations"]
        tau1_order = np.asarray(permutations["tau1"], dtype=int)
        tau2_order = np.asarray(permutations["tau2"], dtype=int)
        expected = list(range(count))
        if sorted(tau1_order.tolist()) != expected or sorted(tau2_order.tolist()) != expected:
            raise ValueError("SOPDT accepted_permutations must each contain 0..count-1 once.")
        examples = _second_order_examples(sample, ranges, tau1_order, tau2_order)
        permutation_metadata = {
            "search_seed": int(config["permutation_search_seed"]),
            "accepted_attempt": int(config["accepted_permutation_attempt"]),
            "tau1": tau1_order.tolist(),
            "tau2": tau2_order.tolist(),
        }
    else:
        raise ValueError(f"Unsupported demonstration system: {system}")

    normalized = _normalized_examples(system, examples, ranges)
    test_points = _load_normalized_test_points(system, test_path, ranges)
    test_distances = _nearest_cross_distances(normalized, test_points)
    if float(test_distances.min()) < threshold:
        raise ValueError(
            f"{system} demonstration minimum test distance {test_distances.min():.12g} "
            f"is below {threshold:.12g}."
        )
    if _has_exact_overlap(normalized, test_points):
        raise ValueError(f"{system} demonstration contains an exact benchmark plant.")
    internal_distances = _nearest_internal_distances(normalized)

    response_peaks = _response_peaks(system, examples, initial_pid, simulation)
    response_constraints = config.get("response_constraints")
    if response_constraints:
        normal_limit = float(response_constraints["normal_max_abs_output"])
        absolute_limit = float(response_constraints["absolute_max_abs_output"])
        max_extreme = int(response_constraints["max_above_normal"])
        above_normal = sum(value > normal_limit for value in response_peaks)
        if max(response_peaks) > absolute_limit or above_normal > max_extreme:
            raise ValueError(
                f"{system} response constraints failed: max={max(response_peaks):.12g}, "
                f"above_normal={above_normal}."
            )

    design: dict[str, Any] = {
        "method": "scipy.stats.qmc.LatinHypercube",
        "seed": seed,
        "start_seed": int(config.get("start_seed", seed)),
        "accepted_seed": seed,
        "seed_attempts": int(config.get("seed_attempts", 1)),
        "count": count,
        "ranges": ranges,
        "excluded_test_plants_path": test_path.as_posix(),
        "minimum_test_distance": threshold,
        "actual_minimum_test_distance": float(test_distances.min()),
        "minimum_internal_distance": float(internal_distances.min()),
        "parameter_precision": config["parameter_precision"],
    }
    if permutation_metadata is not None:
        design["accepted_permutations"] = permutation_metadata
        design["response_constraints"] = response_constraints
    return examples, design


def _first_order_examples(
    sample: np.ndarray, ranges: dict[str, list[float]]
) -> list[dict[str, Any]]:
    k_low, k_high = map(float, ranges["k"])
    t_low, t_high = map(float, ranges["t"])
    k_values = k_low + sample[:, 0] * (k_high - k_low)
    t_values = np.rint(t_low + sample[:, 1] * (t_high - t_low)).astype(int)
    return [
        {"id": index, "k": float(k), "t": int(t)}
        for index, (k, t) in enumerate(zip(k_values, t_values, strict=True), start=1)
    ]


def _second_order_examples(
    sample: np.ndarray,
    ranges: dict[str, list[float]],
    tau1_order: np.ndarray,
    tau2_order: np.ndarray,
) -> list[dict[str, Any]]:
    k_low, k_high = map(float, ranges["k"])
    tau_low, tau_high = map(float, ranges["tau"])
    k_values = k_low + sample[:, 0] * (k_high - k_low)
    tau_a = tau_low + sample[tau1_order, 1] * (tau_high - tau_low)
    tau_b = tau_low + sample[tau2_order, 2] * (tau_high - tau_low)
    return [
        {
            "id": index,
            "k": float(k),
            "tau1": float(min(a, b)),
            "tau2": float(max(a, b)),
        }
        for index, (k, a, b) in enumerate(zip(k_values, tau_a, tau_b, strict=True), start=1)
    ]


def _normalized_examples(
    system: str,
    examples: list[dict[str, Any]],
    ranges: dict[str, list[float]],
) -> np.ndarray:
    if system == "first_order":
        k_low, k_high = map(float, ranges["k"])
        t_low, t_high = map(float, ranges["t"])
        return np.asarray(
            [
                [
                    (item["k"] - k_low) / (k_high - k_low),
                    (item["t"] - t_low) / (t_high - t_low),
                ]
                for item in examples
            ]
        )
    k_low, k_high = map(float, ranges["k"])
    tau_low, tau_high = map(float, ranges["tau"])
    return np.asarray(
        [
            [
                (item["k"] - k_low) / (k_high - k_low),
                (item["tau1"] - tau_low) / (tau_high - tau_low),
                (item["tau2"] - tau_low) / (tau_high - tau_low),
            ]
            for item in examples
        ]
    )


def _load_normalized_test_points(
    system: str,
    path: Path,
    ranges: dict[str, list[float]],
) -> np.ndarray:
    plants = yaml.safe_load(path.read_text(encoding="utf-8"))["plants"]
    examples: list[dict[str, Any]] = []
    for index, plant in enumerate(plants, start=1):
        if system == "first_order":
            examples.append({"id": index, "k": float(plant["k"]), "t": float(plant["t"])})
        else:
            tau1, tau2 = sorted((float(plant["tau1"]), float(plant["tau2"])))
            examples.append({"id": index, "k": float(plant["k"]), "tau1": tau1, "tau2": tau2})
    return _normalized_examples(system, examples, ranges)


def _response_peaks(
    system: str,
    examples: list[dict[str, Any]],
    initial_pid: PIDParams,
    simulation: SimulationSettings,
) -> list[float]:
    peaks: list[float] = []
    for example in examples:
        if system == "first_order":
            result = FirstOrderDelaySimulator(
                FirstOrderPlant(k=float(example["k"]), t=float(example["t"])),
                initial_pid,
                simulation,
            ).run()
        else:
            result = SecondOrderDelaySimulator(
                SecondOrderPlant(
                    k=float(example["k"]),
                    tau1=float(example["tau1"]),
                    tau2=float(example["tau2"]),
                ),
                initial_pid,
                simulation,
            ).run()
        peaks.append(float(np.max(np.abs(result.output))))
    return peaks


def _nearest_cross_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2).min(axis=1)


def _nearest_internal_distances(points: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1)


def _has_exact_overlap(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.any(np.all(left[:, None, :] == right[None, :, :], axis=2)))


def _pid_from_dict(data: dict[str, Any]) -> PIDParams:
    return PIDParams(kp=float(data["kp"]), ki=float(data["ki"]), kd=float(data["kd"]))


def _simulation_from_dict(data: dict[str, Any]) -> SimulationSettings:
    return SimulationSettings(
        setpoint=float(data["setpoint"]),
        sim_time=float(data["sim_time"]),
        num_points=int(data["num_points"]),
        time_delay=float(data["time_delay"]),
    )


def _text_bytes(text: str) -> bytes:
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def _yaml_bytes(payload: dict[str, Any]) -> bytes:
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100)
    return _text_bytes(text)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _check_artifacts(desired: dict[Path, bytes]) -> None:
    failures: list[str] = []
    for path, expected in desired.items():
        if not path.is_file():
            failures.append(f"missing: {path}")
        elif path.read_bytes() != expected:
            failures.append(f"stale: {path}")
    if failures:
        raise ValueError("Frozen demonstration check failed:\n" + "\n".join(failures))


def _write_artifacts(desired: dict[Path, bytes], *, force: bool) -> None:
    existing = [path for path in desired if path.exists()]
    if existing and not force:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite demonstration artifacts: {joined}")
    for path, content in desired.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary_path.write_bytes(content)
        temporary_path.replace(path)
