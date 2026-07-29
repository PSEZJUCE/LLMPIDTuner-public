from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from llmpidtuner.config import CaseConfig
from llmpidtuner.demonstrations import (
    demonstration_protocol_id,
    generate_demonstration_from_spec,
    imc_pid_tuning_first_order,
    imc_pid_tuning_second_order,
)
from llmpidtuner.io import write_results_excel, write_simulation_files
from llmpidtuner.experiment_protocol import imc_pid_for_style, lambda_for_style
from llmpidtuner.llm import LLMSettings, PIDControllerClient, _thinking_request_body
from llmpidtuner.metrics import calculate_response_metrics
from llmpidtuner.models import (
    FirstOrderPlant,
    PIDParams,
    ResponseMetrics,
    SecondOrderPlant,
)
from llmpidtuner.plotting import plot_pid_comparison
from llmpidtuner.prompting import DEFAULT_SYSTEM_ROLE, build_feedback_prompt, build_initial_prompt
from llmpidtuner.simulation import (
    FirstOrderDelaySimulator,
    SecondOrderDelaySimulator,
    SimulationResult,
)


@dataclass(frozen=True)
class RunOutcome:
    status: str
    stop_reason: str
    completed_iterations: int
    converged: bool
    final_pid: PIDParams
    final_iae: float
    final_overshoot: float
    final_steady_state_error: float
    failed_next_llm_call: int | None = None
    error_message: str = ""


def run_case(
    config: CaseConfig,
    batch_groups: set[int] | None = None,
    write_batch_excel: bool = True,
) -> Path:
    output_root = Path(config.output_dir) / config.name
    output_root.mkdir(parents=True, exist_ok=True)
    if config.mode == "llm" and config.llm_profile:
        _write_llm_metadata(output_root, _llm_settings(config), config.prompt_variant)

    if config.system == "first_order":
        if config.first_order is None:
            raise ValueError("first_order plant is required.")
        _run_single(config, config.first_order, output_root)
    elif config.system == "second_order":
        if config.second_order is None:
            raise ValueError("second_order plant is required.")
        _run_single(config, config.second_order, output_root)
    elif config.system == "first_order_batch":
        _run_batch(
            config,
            output_root,
            "first_order",
            batch_groups=batch_groups,
            write_batch_excel=write_batch_excel,
        )
    elif config.system == "second_order_batch":
        _run_batch(
            config,
            output_root,
            "second_order",
            batch_groups=batch_groups,
            write_batch_excel=write_batch_excel,
        )
    else:
        raise ValueError(f"Unsupported system: {config.system}")

    return output_root


def _run_single(
    config: CaseConfig,
    plant: FirstOrderPlant | SecondOrderPlant,
    output_dir: Path,
) -> RunOutcome:
    pid = config.initial_pid
    result = _simulate(plant, pid, config)
    parameter_file, curve_file = write_simulation_files(result, pid, output_dir)
    metrics = _response_metrics(result, config)

    if config.mode == "dry_run":
        demonstration_path = _prepare_demonstration(config, output_dir)
        initial_prompt = build_initial_prompt(
            curve_file,
            parameter_file,
            demonstration_path=demonstration_path,
            time_delay=config.simulation.time_delay,
            prompt_variant=config.prompt_variant,
            control_style=config.control_style,
        )
        (output_dir / "system_role.txt").write_text(DEFAULT_SYSTEM_ROLE, encoding="utf-8")
        (output_dir / "initial_prompt.txt").write_text(initial_prompt, encoding="utf-8")
        plot_pid_comparison(
            result,
            result,
            config.initial_pid,
            config.initial_pid,
            config.simulation.setpoint,
            output_dir / "pid_tuning_comparison.png",
        )
        print(f"Dry run complete: {output_dir}")
        print(f"Overshoot={metrics.overshoot:.2f}%, SSE={metrics.steady_state_error:.2f}%")
        outcome = RunOutcome(
            status="dry_run",
            stop_reason="dry_run",
            completed_iterations=0,
            converged=False,
            final_pid=config.initial_pid,
            final_iae=result.iae,
            final_overshoot=metrics.overshoot,
            final_steady_state_error=metrics.steady_state_error,
        )
        _write_run_status(output_dir, outcome, config)
        return outcome

    if config.mode == "imc":
        imc_pid = _imc_pid(plant, config)
        imc_result = _simulate(plant, imc_pid, config)
        write_simulation_files(imc_result, imc_pid, output_dir, stage_label="iteration_1")
        imc_metrics = _response_metrics(imc_result, config)
        plot_pid_comparison(
            result,
            imc_result,
            config.initial_pid,
            imc_pid,
            config.simulation.setpoint,
            output_dir / "pid_tuning_comparison.png",
        )
        _write_imc_metadata(output_dir, config, plant)
        print(f"IMC run complete: {output_dir}")
        print(f"IMC PID: Kp={imc_pid.kp:.3f}, Ki={imc_pid.ki:.3f}, Kd={imc_pid.kd:.3f}")
        print(
            f"IMC metrics: Overshoot={imc_metrics.overshoot:.2f}%, "
            f"SSE={imc_metrics.steady_state_error:.2f}%"
        )
        converged = imc_metrics.converged(
            overshoot=config.success_overshoot,
            steady_state_error=config.success_steady_state_error,
        )
        outcome = RunOutcome(
            status="converged" if converged else "imc_complete",
            stop_reason="success_threshold" if converged else "imc_complete",
            completed_iterations=1,
            converged=converged,
            final_pid=imc_pid,
            final_iae=imc_result.iae,
            final_overshoot=imc_metrics.overshoot,
            final_steady_state_error=imc_metrics.steady_state_error,
        )
        _write_run_status(output_dir, outcome, config)
        return outcome

    demonstration_path = _prepare_demonstration(config, output_dir)
    initial_prompt = build_initial_prompt(
        curve_file,
        parameter_file,
        demonstration_path=demonstration_path,
        time_delay=config.simulation.time_delay,
        prompt_variant=config.prompt_variant,
        control_style=config.control_style,
    )
    (output_dir / "system_role.txt").write_text(DEFAULT_SYSTEM_ROLE, encoding="utf-8")
    (output_dir / "initial_prompt.txt").write_text(initial_prompt, encoding="utf-8")
    llm_settings = _llm_settings(config)
    _write_llm_metadata(output_dir, llm_settings, config.prompt_variant)
    client = PIDControllerClient(llm_settings)
    system_content = DEFAULT_SYSTEM_ROLE
    current_metrics = metrics
    final_result = result
    final_pid = config.initial_pid
    completed_iterations = 0

    try:
        new_pid = client.call_pid_parameters(system_content, initial_prompt, use_initial_model=True)
    except Exception as error:
        outcome = _llm_failure_outcome(
            error=error,
            failed_next_llm_call=1,
            completed_iterations=completed_iterations,
            final_pid=final_pid,
            final_result=final_result,
            final_metrics=current_metrics,
        )
        _write_run_status(output_dir, outcome, config)
        _plot_final_comparison(output_dir, plant, config, final_result, final_pid)
        print(f"Run stopped: {output_dir}")
        print(f"Stop reason: {outcome.stop_reason} at LLM call {outcome.failed_next_llm_call}")
        return outcome

    for iteration in range(1, config.max_iterations + 1):
        result = _simulate(plant, new_pid, config)
        final_result = result
        final_pid = new_pid
        completed_iterations = iteration
        parameter_file, curve_file = write_simulation_files(
            result, new_pid, output_dir, stage_label=f"iteration_{iteration}"
        )
        current_metrics = _response_metrics(result, config)
        if current_metrics.converged(
            overshoot=config.success_overshoot,
            steady_state_error=config.success_steady_state_error,
        ):
            break
        if iteration == config.max_iterations:
            break
        feedback_prompt = build_feedback_prompt(
            curve_file,
            parameter_file,
            time_delay=config.simulation.time_delay,
            prompt_variant=config.prompt_variant,
            control_style=config.control_style,
        )
        (output_dir / f"feedback_prompt_{iteration}.txt").write_text(
            feedback_prompt, encoding="utf-8"
        )
        try:
            new_pid = client.call_pid_parameters(
                system_content, feedback_prompt, use_initial_model=False
            )
        except Exception as error:
            outcome = _llm_failure_outcome(
                error=error,
                failed_next_llm_call=iteration + 1,
                completed_iterations=completed_iterations,
                final_pid=final_pid,
                final_result=final_result,
                final_metrics=current_metrics,
            )
            _write_run_status(output_dir, outcome, config)
            _plot_final_comparison(output_dir, plant, config, final_result, final_pid)
            print(f"Run stopped: {output_dir}")
            print(f"Stop reason: {outcome.stop_reason} at LLM call {outcome.failed_next_llm_call}")
            print(
                f"Last completed PID: Kp={final_pid.kp:.3f}, "
                f"Ki={final_pid.ki:.3f}, Kd={final_pid.kd:.3f}"
            )
            print(
                f"Last metrics: Overshoot={current_metrics.overshoot:.2f}%, "
                f"SSE={current_metrics.steady_state_error:.2f}%"
            )
            return outcome

    _plot_final_comparison(output_dir, plant, config, final_result, final_pid)
    print(f"Run complete: {output_dir}")
    print(f"Final PID: Kp={final_pid.kp:.3f}, Ki={final_pid.ki:.3f}, Kd={final_pid.kd:.3f}")
    print(
        f"Final metrics: Overshoot={current_metrics.overshoot:.2f}%, "
        f"SSE={current_metrics.steady_state_error:.2f}%"
    )
    converged = current_metrics.converged(
        overshoot=config.success_overshoot,
        steady_state_error=config.success_steady_state_error,
    )
    outcome = RunOutcome(
        status="converged" if converged else "max_iterations",
        stop_reason="success_threshold" if converged else "max_iterations",
        completed_iterations=completed_iterations,
        converged=converged,
        final_pid=final_pid,
        final_iae=final_result.iae,
        final_overshoot=current_metrics.overshoot,
        final_steady_state_error=current_metrics.steady_state_error,
    )
    _write_run_status(output_dir, outcome, config)
    return outcome


def _run_batch(
    config: CaseConfig,
    output_root: Path,
    plant_kind: str,
    batch_groups: set[int] | None = None,
    write_batch_excel: bool = True,
) -> None:
    batch = config.batch or {}
    all_plants = _batch_plants(batch, plant_kind)
    plants = [item for item in all_plants if batch_groups is None or item[0] in batch_groups]
    if batch_groups is not None:
        available_groups = {group for group, _ in all_plants}
        missing_groups = sorted(batch_groups - available_groups)
        if missing_groups:
            raise ValueError(f"Requested batch groups do not exist: {missing_groups}")
    records: list[dict[str, object]] = []

    base_config = config
    for group, plant in plants:
        config = _batch_case_config(base_config, batch, group)
        folder = output_root / _batch_folder_name(group, plant)

        if config.mode == "llm" and config.resume and _batch_group_complete(folder):
            result = _simulate(plant, config.initial_pid, config)
            metrics = _response_metrics(result, config)
            record = _batch_record(config, group, plant, metrics, result.iae)
            record["Skipped Existing"] = True
            records.append(record)
            print(f"Skipping completed group: {folder}")
            continue

        result = _simulate(plant, config.initial_pid, config)
        parameter_file, curve_file = write_simulation_files(result, config.initial_pid, folder)
        metrics = _response_metrics(result, config)

        record = _batch_record(config, group, plant, metrics, result.iae)

        if config.mode == "dry_run":
            plot_pid_comparison(
                result,
                result,
                config.initial_pid,
                config.initial_pid,
                config.simulation.setpoint,
                folder / "pid_tuning_comparison.png",
            )
            demonstration_path = _prepare_demonstration(config, output_root)
            initial_prompt = build_initial_prompt(
                curve_file,
                parameter_file,
                demonstration_path=demonstration_path,
                time_delay=config.simulation.time_delay,
                prompt_variant=config.prompt_variant,
                control_style=config.control_style,
            )
            (folder / "initial_prompt.txt").write_text(initial_prompt, encoding="utf-8")
        elif config.mode == "imc":
            imc_pid = _imc_pid(plant, config)
            imc_result = _simulate(plant, imc_pid, config)
            write_simulation_files(imc_result, imc_pid, folder, stage_label="iteration_1")
            imc_metrics = _response_metrics(imc_result, config)
            plot_pid_comparison(
                result,
                imc_result,
                config.initial_pid,
                imc_pid,
                config.simulation.setpoint,
                folder / "pid_tuning_comparison.png",
            )
            _write_imc_metadata(folder, config, plant)
            record.update(
                {
                    "IMC Lambda": _imc_lambda_value(config, plant),
                    "Final Kp": imc_pid.kp,
                    "Final Ki": imc_pid.ki,
                    "Final Kd": imc_pid.kd,
                    "Final IAE": imc_result.iae,
                    "Final Overshoot": imc_metrics.overshoot,
                    "Final Steady-state Error": imc_metrics.steady_state_error,
                }
            )
        else:
            outcome = _run_single(config, plant, folder)
            record.update(_run_outcome_record(outcome))
        records.append(record)

    if write_batch_excel:
        write_results_excel(records, output_root / "experiment_results.xlsx")
    print(f"Batch run complete: {output_root} ({len(plants)} groups)")


def write_batch_results_from_config(config: CaseConfig) -> Path:
    """Rebuild one batch workbook from completed per-group result files."""
    if config.system == "first_order_batch":
        plant_kind = "first_order"
    elif config.system == "second_order_batch":
        plant_kind = "second_order"
    else:
        raise ValueError("Only batch case configs can be collected.")

    output_root = Path(config.output_dir) / config.name
    records: list[dict[str, object]] = []
    base_config = config
    for group, plant in _batch_plants(config.batch or {}, plant_kind):
        config = _batch_case_config(base_config, base_config.batch or {}, group)
        initial_result = _simulate(plant, config.initial_pid, config)
        initial_metrics = _response_metrics(initial_result, config)
        record = _batch_record(config, group, plant, initial_metrics, initial_result.iae)
        status_path = output_root / _batch_folder_name(group, plant) / "run_status.yaml"
        if status_path.exists():
            status = yaml.safe_load(status_path.read_text(encoding="utf-8")) or {}
            final_pid = status.get("final_pid", {})
            record.update(
                {
                    "Run Status": status.get("status", ""),
                    "Stop Reason": status.get("stop_reason", ""),
                    "Completed Iterations": status.get("completed_iterations", ""),
                    "Failed Next LLM Call": status.get("failed_next_llm_call") or "",
                    "Final Kp": final_pid.get("kp", ""),
                    "Final Ki": final_pid.get("ki", ""),
                    "Final Kd": final_pid.get("kd", ""),
                    "Final IAE": status.get("final_iae", ""),
                    "Final Overshoot": status.get("final_overshoot", ""),
                    "Final Steady-state Error": status.get("final_steady_state_error", ""),
                }
            )
        else:
            record["Run Status"] = "missing"
        records.append(record)

    output_path = output_root / "experiment_results.xlsx"
    write_results_excel(records, output_path)
    return output_path


def _batch_plants(
    batch: dict[str, Any],
    plant_kind: str,
) -> list[tuple[int, FirstOrderPlant | SecondOrderPlant]]:
    plants_path = batch.get("cases_path") or batch.get("plants_path")
    if plants_path:
        return _load_batch_plants(Path(str(plants_path)), plant_kind)

    total = int(batch.get("count", 100))
    seed = int(batch.get("seed", 42))
    rng = random.Random(seed)
    plants: list[tuple[int, FirstOrderPlant | SecondOrderPlant]] = []
    for group in range(1, total + 1):
        if plant_kind == "first_order":
            plant = FirstOrderPlant(
                k=rng.uniform(*batch.get("k_range", [0.2, 0.9])),
                t=float(rng.randint(*batch.get("t_range", [100, 600]))),
            )
        else:
            plant = SecondOrderPlant(
                k=rng.uniform(*batch.get("k_range", [0.1, 3.0])),
                tau1=rng.uniform(*batch.get("tau1_range", [0.1, 100.0])),
                tau2=rng.uniform(*batch.get("tau2_range", [0.1, 100.0])),
            )
        plants.append((group, plant))
    return plants


def _load_batch_plants(
    path: Path,
    plant_kind: str,
) -> list[tuple[int, FirstOrderPlant | SecondOrderPlant]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows = data.get("cases", data.get("plants", data if isinstance(data, list) else []))
    plants: list[tuple[int, FirstOrderPlant | SecondOrderPlant]] = []
    for index, row in enumerate(rows, start=1):
        group = int(row.get("group", index))
        plant_row = row.get("plant", row)
        if plant_kind == "first_order":
            plants.append((group, FirstOrderPlant(k=float(plant_row["k"]), t=float(plant_row["t"]))))
        else:
            plants.append(
                (
                    group,
                    SecondOrderPlant(
                        k=float(plant_row["k"]),
                        tau1=float(plant_row["tau1"]),
                        tau2=float(plant_row["tau2"]),
                    ),
                )
            )
    return plants



def _batch_case_config(
    config: CaseConfig,
    batch: dict[str, Any],
    group: int,
) -> CaseConfig:
    path_value = batch.get("cases_path")
    if not path_value:
        return config
    data = yaml.safe_load(Path(str(path_value)).read_text(encoding="utf-8")) or {}
    rows = data.get("cases", data if isinstance(data, list) else [])
    row = next(
        (
            item
            for index, item in enumerate(rows, start=1)
            if int(item.get("group", index)) == group
        ),
        None,
    )
    if row is None:
        raise ValueError(f"Group {group} is missing from {path_value}.")
    pid_data = row["initial_pid"]
    return replace(
        config,
        initial_pid=PIDParams(
            kp=float(pid_data["kp"]),
            ki=float(pid_data["ki"]),
            kd=float(pid_data["kd"]),
        ),
        simulation=replace(
            config.simulation,
            time_delay=float(row["time_delay"]),
        ),
    )


def write_batch_plants_from_config(config: CaseConfig, output_path: str | Path) -> Path:
    if config.system == "first_order_batch":
        plant_kind = "first_order"
    elif config.system == "second_order_batch":
        plant_kind = "second_order"
    else:
        raise ValueError("Only batch case configs can be exported as plant lists.")

    plants = _batch_plants(config.batch or {}, plant_kind)
    rows: list[dict[str, float | int]] = []
    for group, plant in plants:
        if isinstance(plant, FirstOrderPlant):
            rows.append({"group": group, "k": plant.k, "t": plant.t})
        else:
            rows.append({"group": group, "k": plant.k, "tau1": plant.tau1, "tau2": plant.tau2})

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"plants": rows}, sort_keys=False), encoding="utf-8")
    return path


def _simulate(
    plant: FirstOrderPlant | SecondOrderPlant,
    pid: PIDParams,
    config: CaseConfig,
) -> SimulationResult:
    if isinstance(plant, FirstOrderPlant):
        return FirstOrderDelaySimulator(plant, pid, config.simulation).run()
    return SecondOrderDelaySimulator(plant, pid, config.simulation).run()

def _response_metrics(result: SimulationResult, config: CaseConfig) -> ResponseMetrics:
    return calculate_response_metrics(
        result.time,
        result.output,
        result.control_signal,
        result.errors,
        result.iae,
        setpoint=config.simulation.setpoint,
        time_delay=config.simulation.time_delay,
        finite=result.finite,
    )



def _batch_folder_name(group: int, plant: FirstOrderPlant | SecondOrderPlant) -> str:
    prefix = f"group_{group:03d}"
    if isinstance(plant, FirstOrderPlant):
        return f"{prefix}_example_K_{plant.k:.2f}_T_{int(plant.t)}"
    return f"{prefix}_example_K_second_{plant.k:.2f}_tau1_{plant.tau1:.2f}_tau2_{plant.tau2:.2f}"


def _batch_group_complete(folder: Path) -> bool:
    if (folder / "run_status.yaml").exists() and (folder / "pid_tuning_comparison.png").exists():
        return True
    return (folder / "pid_tuning_comparison.png").exists() and bool(
        list(folder.glob("value_curve_iteration_*.txt"))
    )


def _batch_record(
    config: CaseConfig,
    group: int,
    plant: FirstOrderPlant | SecondOrderPlant,
    metrics: ResponseMetrics,
    initial_iae: float,
) -> dict[str, object]:
    record: dict[str, object] = {
        "Experiment Group": group,
        "LLM Profile": config.llm_profile or "",
        "Mode": config.mode,
        "overshoot": metrics.overshoot,
        "Steady-state Error": metrics.steady_state_error,
        "Initial IAE": initial_iae,
    }
    if isinstance(plant, FirstOrderPlant):
        record.update({"K Value": plant.k, "T Value": plant.t})
    else:
        record.update(
            {"K_second Value": plant.k, "tau1 Value": plant.tau1, "tau2 Value": plant.tau2}
        )
    return record


def _run_outcome_record(outcome: RunOutcome) -> dict[str, object]:
    return {
        "Run Status": outcome.status,
        "Stop Reason": outcome.stop_reason,
        "Completed Iterations": outcome.completed_iterations,
        "Failed Next LLM Call": outcome.failed_next_llm_call or "",
        "Final Kp": outcome.final_pid.kp,
        "Final Ki": outcome.final_pid.ki,
        "Final Kd": outcome.final_pid.kd,
        "Final IAE": outcome.final_iae,
        "Final Overshoot": outcome.final_overshoot,
        "Final Steady-state Error": outcome.final_steady_state_error,
    }


def _llm_failure_outcome(
    error: Exception,
    failed_next_llm_call: int,
    completed_iterations: int,
    final_pid: PIDParams,
    final_result: SimulationResult,
    final_metrics: ResponseMetrics,
) -> RunOutcome:
    stop_reason = _classify_llm_failure(error)
    return RunOutcome(
        status="llm_failed",
        stop_reason=stop_reason,
        completed_iterations=completed_iterations,
        converged=False,
        final_pid=final_pid,
        final_iae=final_result.iae,
        final_overshoot=final_metrics.overshoot,
        final_steady_state_error=final_metrics.steady_state_error,
        failed_next_llm_call=failed_next_llm_call,
        error_message=str(error)[:1000],
    )


def _classify_llm_failure(error: Exception) -> str:
    message = str(error).lower()
    if "maximum context length" in message or "context length" in message:
        return "context_length_exceeded"
    if "pid values not found" in message:
        return "pid_parse_failed"
    return "llm_call_failed"


def _plot_final_comparison(
    output_dir: Path,
    plant: FirstOrderPlant | SecondOrderPlant,
    config: CaseConfig,
    final_result: SimulationResult,
    final_pid: PIDParams,
) -> None:
    plot_pid_comparison(
        initial_result=_simulate(plant, config.initial_pid, config),
        final_result=final_result,
        initial_pid=config.initial_pid,
        final_pid=final_pid,
        setpoint=config.simulation.setpoint,
        output_path=output_dir / "pid_tuning_comparison.png",
    )


def _write_run_status(output_dir: Path, outcome: RunOutcome, config: CaseConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": outcome.status,
        "stop_reason": outcome.stop_reason,
        "completed_iterations": outcome.completed_iterations,
        "max_iterations": config.max_iterations,
        "prompt_variant": config.prompt_variant,
        "failed_next_llm_call": outcome.failed_next_llm_call,
        "converged": outcome.converged,
        "final_pid": {
            "kp": outcome.final_pid.kp,
            "ki": outcome.final_pid.ki,
            "kd": outcome.final_pid.kd,
        },
        "final_iae": outcome.final_iae,
        "final_overshoot": outcome.final_overshoot,
        "final_steady_state_error": outcome.final_steady_state_error,
        "error_message": outcome.error_message,
    }
    (output_dir / "run_status.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )


def _imc_pid(plant: FirstOrderPlant | SecondOrderPlant, config: CaseConfig) -> PIDParams:
    style = (config.imc or {}).get("style")
    if style:
        return imc_pid_for_style(plant, config.simulation.time_delay, str(style))
    lambda_value = _imc_lambda_value(config, plant)
    time_delay = config.simulation.time_delay
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


def _imc_lambda_value(
    config: CaseConfig,
    plant: FirstOrderPlant | SecondOrderPlant,
) -> float:
    style = (config.imc or {}).get("style")
    if style:
        return lambda_for_style(plant, config.simulation.time_delay, str(style))
    return float((config.imc or {}).get("lambda_value", 10.0))


def _prepare_demonstration(config: CaseConfig, output_dir: Path) -> str | None:
    if config.demonstration:
        text = generate_demonstration_from_spec(
            config.demonstration,
            initial_pid=config.initial_pid,
            simulation=config.simulation,
        )
        if text is None:
            return config.demonstration_path
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "demonstration_prompt.txt"
        output_path.write_text(text, encoding="utf-8")
        _write_demonstration_metadata(config.demonstration, output_dir)
        return str(output_path)
    return config.demonstration_path


def _write_demonstration_metadata(spec: dict[str, Any], output_dir: Path) -> None:
    protocol_id = demonstration_protocol_id(spec)
    if protocol_id is None:
        return
    manifest_path = Path(spec["manifest_path"])
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    source = manifest.get("source") or manifest.get("examples") or {}
    metadata = {
        "demonstration_protocol": protocol_id,
        "parent_protocol": manifest.get("parent_protocol"),
        "prompt_variant": manifest.get("prompt_variant", "full"),
        "system": manifest["system"],
        "control_style": manifest.get("control_style"),
        "lambda_value": manifest.get("lambda_value"),
        "prompt_sha256": manifest["prompt"]["sha256"],
        "source_sha256": source.get("sha256"),
        "manifest_path": manifest_path.as_posix(),
    }
    (output_dir / "demonstration_metadata.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8", newline="\n"
    )


def _write_llm_metadata(output_dir: Path, settings: LLMSettings, prompt_variant: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    thinking_body = _thinking_request_body(settings)
    thinking_body_json = json.dumps(thinking_body, sort_keys=True) if thinking_body else ""
    content = "\n".join(
        [
            f"profile={settings.profile or ''}",
            f"provider={settings.provider}",
            f"base_url={settings.base_url or ''}",
            f"model={settings.model}",
            f"temperature={settings.temperature}",
            f"top_p={settings.top_p}",
            f"enable_thinking={settings.enable_thinking}",
            f"thinking_request_body={thinking_body_json}",
            f"seed={settings.seed if settings.seed is not None else ''}",
            f"prompt_variant={prompt_variant}",
            f"max_retries={settings.max_retries}",
        ]
    )
    (output_dir / "llm_metadata.txt").write_text(content + "\n", encoding="utf-8")


def _llm_settings(config: CaseConfig) -> LLMSettings:
    return LLMSettings.from_env(profile=config.llm_profile).with_overrides(config.llm)


def _write_imc_metadata(output_dir: Path, config: CaseConfig, plant: FirstOrderPlant | SecondOrderPlant) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        [
            "method=imc",
            f"style={(config.imc or {}).get('style', 'explicit_lambda')}",
            f"lambda_value={_imc_lambda_value(config, plant)}",
            f"time_delay={config.simulation.time_delay}",
        ]
    )
    (output_dir / "imc_metadata.txt").write_text(content + "\n", encoding="utf-8")
