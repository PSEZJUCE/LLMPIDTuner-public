from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

import numpy as np
import yaml

from llmpidtuner.experiment_protocol import (
    generate_protocol_cases,
    imc_pid_for_style,
    simulate_protocol_case,
)
from llmpidtuner.models import FirstOrderPlant, PIDParams, SimulationSettings
from llmpidtuner.training.prompts import (
    build_feedback_message,
    build_messages,
    build_target_message,
    format_pid,
    response_description,
)
from llmpidtuner.training.simulation import (
    PlantSpec,
    ResponseMetrics,
    imc_pid_for_plant,
    simulate_pid,
)


@dataclass
class PromptSample:
    plant: PlantSpec
    current_pid: PIDParams
    current_metrics: ResponseMetrics
    messages: list[dict[str, Any]]
    control_style: str = "balanced"

    def as_dict(self) -> dict[str, Any]:
        return {
            "plant": self.plant.as_dict(),
            "current_pid": _pid_as_dict(self.current_pid),
            "current_metrics": self.current_metrics.as_dict(),
            "messages": self.messages,
            "control_style": self.control_style,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "PromptSample":
        plant = PlantSpec.from_dict(data["plant"])
        pid_data = data["current_pid"]
        pid = PIDParams(
            kp=float(pid_data.get("kp", pid_data.get("Kp"))),
            ki=float(pid_data.get("ki", pid_data.get("Ki"))),
            kd=float(pid_data.get("kd", pid_data.get("Kd"))),
        )
        metrics = ResponseMetrics.from_dict(data["current_metrics"])
        control_style = str(data.get("control_style", "balanced"))
        messages = data.get("messages") or build_messages(
            plant, pid, metrics, control_style=control_style
        )
        return PromptSample(
            plant=plant,
            current_pid=pid,
            current_metrics=metrics,
            messages=messages,
            control_style=control_style,
        )


def generate_protocol_prompt_samples_by_type(
    first_order_count: int,
    second_order_count: int,
    seed: int,
    simulation: SimulationSettings,
    demonstrations: dict[str, str] | None = None,
    excluded_case_paths: Iterable[str | Path] = (),
    workers: int = 1,
    control_style: str = "balanced",
) -> list[PromptSample]:
    """Generate deterministic perturbed-IMC SFT tasks with exact type counts."""

    excluded_hashes = _load_excluded_case_hashes(excluded_case_paths)
    tasks: list[tuple[str, int, int, SimulationSettings, dict[str, str], set[str], str]] = []
    demos = demonstrations or {}
    for index in range(first_order_count):
        tasks.append(
            ("first_order", index, seed, simulation, demos, excluded_hashes, control_style)
        )
    for index in range(second_order_count):
        tasks.append(
            (
                "second_order",
                index,
                seed + 1_000_000,
                simulation,
                demos,
                excluded_hashes,
                control_style,
            )
        )

    if workers <= 1:
        samples = [_protocol_prompt_task(task) for task in tasks]
    else:
        chunksize = max(1, len(tasks) // (workers * 8))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            samples = list(executor.map(_protocol_prompt_task, tasks, chunksize=chunksize))

    order = np.random.default_rng(seed + 2_000_000).permutation(len(samples))
    return [samples[int(index)] for index in order]


def _protocol_prompt_task(
    task: tuple[str, int, int, SimulationSettings, dict[str, str], set[str], str],
) -> PromptSample:
    system, index, seed, simulation, demonstrations, excluded_hashes, control_style = task

    for attempt in range(50):
        task_seed = seed + index * 1009 + attempt * 104729
        try:
            case = generate_protocol_cases(
                system,
                1,
                task_seed,
                purpose="sft_training",
                simulation=simulation,
                required_target_style=control_style,
                excluded_hashes=excluded_hashes,
                slot_offset=index,
                max_candidates=5000,
            )[0]
        except RuntimeError:
            continue
        target_pid = imc_pid_for_style(case.plant, case.time_delay, control_style)
        _, target_metrics = simulate_protocol_case(
            case.plant,
            target_pid,
            case.time_delay,
            simulation,
        )
        if not target_metrics.converged():
            continue

        plant = _plant_spec_from_protocol_case(case)
        current = simulate_pid(plant, case.initial_pid, simulation)
        description = response_description(
            current,
            time_delay=case.time_delay,
        )
        messages = build_messages(
            plant,
            case.initial_pid,
            current.metrics,
            response_description=description,
            demonstration_cases=demonstrations.get(system),
            control_style=control_style,
        )
        return PromptSample(
            plant=plant,
            current_pid=case.initial_pid,
            current_metrics=current.metrics,
            messages=messages,
            control_style=control_style,
        )
    raise RuntimeError(
        f"Could not generate convergent {control_style} SFT target for {system}:{index}."
    )


def _plant_spec_from_protocol_case(case: Any) -> PlantSpec:
    if isinstance(case.plant, FirstOrderPlant):
        return PlantSpec(
            plant_type="first_order",
            k=case.plant.k,
            t=case.plant.t,
            time_delay=case.time_delay,
        )
    return PlantSpec(
        plant_type="second_order",
        k=case.plant.k,
        tau1=case.plant.tau1,
        tau2=case.plant.tau2,
        time_delay=case.time_delay,
    )


def load_protocol_prompt_samples(
    paths: Iterable[str | Path],
    *,
    simulation: SimulationSettings,
    demonstrations: dict[str, str] | None = None,
    control_style: str = "balanced",
) -> list[PromptSample]:
    """Load frozen first-turn validation prompts from protocol source files."""

    samples: list[PromptSample] = []
    demo_text = demonstrations or {}
    for path_value in paths:
        data = yaml.safe_load(Path(path_value).read_text(encoding="utf-8")) or {}
        for row in data.get("cases", []):
            system = str(row["system"])
            plant_data = row["plant"]
            plant = PlantSpec(
                plant_type=system,
                k=float(plant_data["k"]),
                t=float(plant_data["t"]) if system == "first_order" else None,
                tau1=float(plant_data["tau1"]) if system == "second_order" else None,
                tau2=float(plant_data["tau2"]) if system == "second_order" else None,
                time_delay=float(row["time_delay"]),
                setpoint=simulation.setpoint,
            )
            pid_data = row["initial_pid"]
            pid = PIDParams(
                float(pid_data["kp"]),
                float(pid_data["ki"]),
                float(pid_data["kd"]),
            )
            current = simulate_pid(plant, pid, simulation)
            messages = build_messages(
                plant,
                pid,
                current.metrics,
                response_description=response_description(
                    current,
                    time_delay=plant.time_delay,
                ),
                demonstration_cases=demo_text.get(system),
                control_style=control_style,
            )
            samples.append(
                PromptSample(
                    plant=plant,
                    current_pid=pid,
                    current_metrics=current.metrics,
                    messages=messages,
                    control_style=control_style,
                )
            )
    return samples


def _load_excluded_case_hashes(paths: Iterable[str | Path]) -> set[str]:
    hashes: set[str] = set()
    for path_value in paths:
        data = yaml.safe_load(Path(path_value).read_text(encoding="utf-8")) or {}
        for row in data.get("cases", []):
            provenance = row.get("provenance", {})
            if provenance.get("case_hash"):
                hashes.add(str(provenance["case_hash"]))
    return hashes


class PIDPromptGenerator:
    """Generate balanced, first-turn perturbed-IMC prompts online for GRPO."""

    def __init__(
        self,
        seed: int,
        simulation: SimulationSettings,
        second_order_prob: float = 0.5,
        initial_pid: PIDParams | None = None,
        max_resample_attempts: int = 20,
        demonstrations: dict[str, str] | None = None,
        excluded_plants_paths: Iterable[str | Path] = (),
        control_style: str = "balanced",
    ) -> None:
        del initial_pid
        self.seed = int(seed)
        self.counter = 0
        self.rng = np.random.default_rng(seed)
        self.simulation = simulation
        self.second_order_prob = float(second_order_prob)
        self.max_resample_attempts = max(1, int(max_resample_attempts))
        self.demonstrations = demonstrations or {}
        self.control_style = str(control_style)
        self.excluded_case_hashes = _load_excluded_case_hashes(excluded_plants_paths)

    def sample(self) -> PromptSample:
        index = self.counter
        self.counter += 1
        system = "second_order" if self.rng.random() < self.second_order_prob else "first_order"
        for attempt in range(self.max_resample_attempts):
            try:
                case = generate_protocol_cases(
                    system,
                    1,
                    self.seed + index * 1009 + attempt * 104729,
                    purpose="grpo_online",
                    simulation=self.simulation,
                    excluded_hashes=self.excluded_case_hashes,
                    slot_offset=index,
                    max_candidates=5000,
                )[0]
            except RuntimeError:
                continue
            plant = _plant_spec_from_protocol_case(case)
            result = simulate_pid(plant, case.initial_pid, self.simulation)
            if not result.metrics.finite:
                continue
            description = response_description(
                result,
                time_delay=case.time_delay,
            )
            return PromptSample(
                plant=plant,
                current_pid=case.initial_pid,
                current_metrics=result.metrics,
                messages=build_messages(
                    plant,
                    case.initial_pid,
                    result.metrics,
                    response_description=description,
                    demonstration_cases=self.demonstrations.get(system),
                    control_style=self.control_style,
                ),
                control_style=self.control_style,
            )
        raise RuntimeError(f"Could not generate a finite online GRPO prompt for {system}:{index}.")

    def sample_batch(self, batch_size: int) -> list[PromptSample]:
        return [self.sample() for _ in range(int(batch_size))]

    def state_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "counter": self.counter,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.seed = int(state["seed"])
        self.counter = int(state["counter"])
        self.rng.bit_generator.state = state["rng_state"]


def generate_prompt_samples(
    count: int,
    seed: int,
    simulation: SimulationSettings,
    second_order_prob: float = 0.5,
    initial_pid: PIDParams | None = None,
    demonstrations: dict[str, str] | None = None,
    control_style: str = "balanced",
) -> list[PromptSample]:
    generator = PIDPromptGenerator(
        seed=seed,
        simulation=simulation,
        second_order_prob=second_order_prob,
        initial_pid=initial_pid,
        demonstrations=demonstrations,
        control_style=control_style,
    )
    samples = generator.sample_batch(count)
    if not all(sample.current_metrics.finite for sample in samples):
        raise ValueError("Could not generate a finite baseline for every SFT sample.")
    return samples


def generate_prompt_samples_by_type(
    first_order_count: int,
    second_order_count: int,
    seed: int,
    simulation: SimulationSettings,
    initial_pid: PIDParams | None = None,
    demonstrations: dict[str, str] | None = None,
) -> list[PromptSample]:
    """Generate exact plant-type counts and deterministically mix the rows."""

    if first_order_count < 0 or second_order_count < 0:
        raise ValueError("Plant-type sample counts must be non-negative.")
    first_order = generate_prompt_samples(
        count=first_order_count,
        seed=seed,
        simulation=simulation,
        second_order_prob=0.0,
        initial_pid=initial_pid,
        demonstrations=demonstrations,
    )
    second_order = generate_prompt_samples(
        count=second_order_count,
        seed=seed + 1,
        simulation=simulation,
        second_order_prob=1.0,
        initial_pid=initial_pid,
        demonstrations=demonstrations,
    )
    samples = [*first_order, *second_order]
    order = np.random.default_rng(seed + 2).permutation(len(samples))
    mixed = [samples[int(index)] for index in order]

    actual_first = sum(sample.plant.plant_type == "first_order" for sample in mixed)
    actual_second = sum(sample.plant.plant_type == "second_order" for sample in mixed)
    if (actual_first, actual_second) != (first_order_count, second_order_count):
        raise AssertionError("Generated SFT plant-type counts do not match the request.")
    if len({_plant_fingerprint(sample.plant) for sample in mixed}) != len(mixed):
        raise ValueError("Duplicate process parameters found in the generated SFT data.")
    return mixed


def assert_no_prompt_sample_overlap(
    samples: Iterable[PromptSample],
    excluded_plants_paths: Iterable[str | Path],
    *,
    time_delay: float,
    setpoint: float,
) -> None:
    """Reject exact process overlap with versioned evaluation plant lists."""

    sample_fingerprints = {_plant_fingerprint(sample.plant) for sample in samples}
    for path_value in excluded_plants_paths:
        path = Path(path_value)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        overlap: list[tuple[Any, ...]] = []
        for row in data.get("plants", []):
            if "t" in row:
                plant = PlantSpec(
                    "first_order",
                    float(row["k"]),
                    t=float(row["t"]),
                    time_delay=time_delay,
                    setpoint=setpoint,
                )
            else:
                plant = PlantSpec(
                    "second_order",
                    float(row["k"]),
                    tau1=float(row["tau1"]),
                    tau2=float(row["tau2"]),
                    time_delay=time_delay,
                    setpoint=setpoint,
                )
            fingerprint = _plant_fingerprint(plant)
            if fingerprint in sample_fingerprints:
                overlap.append(fingerprint)
        if overlap:
            raise ValueError(
                f"Generated SFT data overlaps {len(overlap)} process(es) in benchmark {path}."
            )


def write_prompt_samples(samples: Iterable[PromptSample], path: str | Path) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with _atomic_text_writer(output_path) as handle:
        for sample in samples:
            handle.write(json.dumps(sample.as_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def load_prompt_samples(path: str | Path) -> list[PromptSample]:
    samples = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                samples.append(PromptSample.from_dict(json.loads(line)))
    return samples


def write_sft_messages_dataset(
    samples: Iterable[PromptSample],
    path: str | Path,
    lambda_value: float = 10.0,
    simulation: SimulationSettings | None = None,
    feedback_sample_probability: float = 0.0,
    seed: int = 42,
    include_target_metrics: bool = False,
) -> int:
    """Write runtime-shaped initial and feedback conversations for SFT."""

    if not 0.0 <= feedback_sample_probability <= 1.0:
        raise ValueError("feedback_sample_probability must be between 0 and 1.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    count = 0
    with _atomic_text_writer(output_path) as handle:
        for sample in samples:
            target_pid = imc_pid_for_plant(
                sample.plant,
                lambda_value=lambda_value,
                control_style=sample.control_style,
            )
            _validate_target_pid(target_pid, count)
            sample_settings = simulation or _settings_for_sample(sample)
            target_result = simulate_pid(sample.plant, target_pid, sample_settings)
            if not target_result.metrics.converged():
                raise ValueError(
                    f"IMC target does not meet the observable convergence criteria at SFT row {count}."
                )

            messages = [dict(message) for message in sample.messages]
            current_pid = sample.current_pid
            current_metrics = sample.current_metrics
            feedback_depth = 0

            if rng.random() < feedback_sample_probability:
                requested_depth = int(rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2]))
                for _ in range(requested_depth):
                    accepted_result = None
                    accepted_pid = None
                    for _attempt in range(20):
                        candidate_pid = _sample_intermediate_pid(rng, current_pid, target_pid)
                        candidate_result = simulate_pid(
                            sample.plant,
                            candidate_pid,
                            sample_settings,
                        )
                        if (
                            candidate_result.metrics.finite
                            and not candidate_result.metrics.converged()
                            and max(
                                abs(candidate_result.metrics.max_value),
                                abs(candidate_result.metrics.min_value),
                            )
                            <= 3.0
                        ):
                            accepted_pid = candidate_pid
                            accepted_result = candidate_result
                            break
                    if accepted_result is None or accepted_pid is None:
                        break
                    messages.extend(
                        [
                            {"role": "assistant", "content": format_pid(accepted_pid)},
                            build_feedback_message(
                                accepted_pid,
                                accepted_result.metrics,
                                response_description(
                                    accepted_result,
                                    time_delay=sample.plant.time_delay,
                                ),
                                sample.plant.time_delay,
                                control_style=sample.control_style,
                            ),
                        ]
                    )
                    current_pid = accepted_pid
                    current_metrics = accepted_result.metrics
                    feedback_depth += 1

            messages.append(build_target_message(target_pid))
            metadata: dict[str, Any] = {
                "schema_version": 3,
                "sample_kind": "initial" if feedback_depth == 0 else "feedback",
                "feedback_depth": feedback_depth,
                "control_style": sample.control_style,
                "plant": sample.plant.as_dict(),
                "current_pid": _pid_as_dict(current_pid),
                "current_metrics": current_metrics.as_dict(),
                "target_pid": _pid_as_dict(target_pid),
                "target_method": "imc_style",
            }

            if include_target_metrics:
                metadata["target_metrics"] = target_result.metrics.as_dict()

            handle.write(
                json.dumps({"messages": messages, "metadata": metadata}, ensure_ascii=False) + "\n"
            )
            count += 1
    return count


def _sample_intermediate_pid(
    rng: np.random.Generator,
    initial: PIDParams,
    target: PIDParams,
) -> PIDParams:
    alpha = float(rng.uniform(0.2, 0.8))
    jitter = rng.lognormal(mean=0.0, sigma=0.12, size=3)

    def interpolate(start: float, end: float, noise: float) -> float:
        start = max(float(start), 1e-8)
        end = max(float(end), 1e-8)
        return float(math.exp((1.0 - alpha) * math.log(start) + alpha * math.log(end)) * noise)

    return PIDParams(
        kp=interpolate(initial.kp, target.kp, float(jitter[0])),
        ki=interpolate(initial.ki, target.ki, float(jitter[1])),
        kd=interpolate(initial.kd, target.kd, float(jitter[2])),
    )


def _validate_target_pid(pid: PIDParams, row_index: int) -> None:
    values = (pid.kp, pid.ki, pid.kd)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite IMC target PID at SFT row {row_index}.")
    if pid.kp <= 0.0 or pid.ki < 0.0 or pid.kd < 0.0:
        raise ValueError(f"Invalid IMC target PID at SFT row {row_index}: {pid}")


def _settings_for_sample(sample: PromptSample) -> SimulationSettings:
    return SimulationSettings(
        setpoint=sample.plant.setpoint,
        time_delay=sample.plant.time_delay,
    )


def _pid_as_dict(pid: PIDParams) -> dict[str, float]:
    return {"kp": pid.kp, "ki": pid.ki, "kd": pid.kd}


def _load_excluded_plant_fingerprints(
    paths: Iterable[str | Path],
    *,
    time_delay: float,
    setpoint: float,
) -> set[tuple[Any, ...]]:
    fingerprints: set[tuple[Any, ...]] = set()
    for path_value in paths:
        path = Path(path_value)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for row in data.get("plants", []):
            if "t" in row:
                plant = PlantSpec(
                    "first_order",
                    float(row["k"]),
                    t=float(row["t"]),
                    time_delay=time_delay,
                    setpoint=setpoint,
                )
            else:
                plant = PlantSpec(
                    "second_order",
                    float(row["k"]),
                    tau1=float(row["tau1"]),
                    tau2=float(row["tau2"]),
                    time_delay=time_delay,
                    setpoint=setpoint,
                )
            fingerprints.add(_plant_fingerprint(plant))
    return fingerprints


def _plant_fingerprint(plant: PlantSpec) -> tuple[Any, ...]:
    if plant.plant_type == "second_order":
        taus = tuple(sorted((float(plant.tau1 or 0.0), float(plant.tau2 or 0.0))))
        return (
            plant.plant_type,
            plant.k,
            *taus,
            plant.time_delay,
            plant.setpoint,
        )
    return (
        plant.plant_type,
        plant.k,
        plant.t,
        plant.time_delay,
        plant.setpoint,
    )


class PromptPool:
    def __init__(self, samples: list[PromptSample], seed: int = 42) -> None:
        if not samples:
            raise ValueError("PromptPool requires at least one sample.")
        self.samples = samples
        self.rng = np.random.default_rng(seed)

    def sample_batch(self, batch_size: int) -> list[PromptSample]:
        indices = self.rng.integers(low=0, high=len(self.samples), size=int(batch_size))
        return [self.samples[int(index)] for index in indices]

    def state_dict(self) -> dict[str, Any]:
        return {"rng_state": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rng.bit_generator.state = state["rng_state"]


@contextmanager
def _atomic_text_writer(output_path: Path) -> Iterator[TextIO]:
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            yield handle
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
