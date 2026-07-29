import math
from pathlib import Path

import numpy as np
import pytest

from llmpidtuner.models import PIDParams, SimulationSettings
from llmpidtuner.training.data import PromptSample
from llmpidtuner.training.grpo import (
    _combine_rank_rollout_rows,
    _combine_rank_step_data,
    _build_completion_mask,
    _completion_termination_metadata,
    _prepare_grpo_output_dir,
)
from llmpidtuner.training.prompts import build_messages
from llmpidtuner.training.simulation import PlantSpec, simulate_pid
from llmpidtuner.training.rewards import (
    RewardConfig,
    calibrate_gain_reference,
    characteristic_polynomial,
    routh_hurwitz,
    stability_proxy,
    _open_loop_phase_margin,
    _controller_phase_radians,
    evaluate_completion,
    format_compliance_score,
    normalize_group_advantages,
    parse_pid,
)


def _reward_sample(settings: SimulationSettings) -> PromptSample:
    plant = PlantSpec(
        plant_type="first_order",
        k=0.8,
        t=5.0,
        time_delay=settings.time_delay,
        setpoint=settings.setpoint,
    )
    pid = PIDParams(0.5, 0.05, 0.01)
    baseline = simulate_pid(plant, pid, settings)
    return PromptSample(
        plant=plant,
        current_pid=pid,
        current_metrics=baseline.metrics,
        messages=build_messages(plant, pid, baseline.metrics),
    )


def test_grpo_output_directory_accepts_only_missing_or_empty_paths(tmp_path: Path) -> None:
    output_dir = tmp_path / "grpo-output"

    assert _prepare_grpo_output_dir(output_dir) == output_dir
    assert output_dir.is_dir()
    assert _prepare_grpo_output_dir(output_dir) == output_dir

    (output_dir / "trainer_log.jsonl").write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        _prepare_grpo_output_dir(output_dir)


def test_grpo_rank_data_is_combined_in_rank_order() -> None:
    rank_data = [
        {
            "rank": 1,
            "rewards": [3.0, 4.0],
            "advantages": [0.5, -0.5],
            "prompt_metadata": [{"final_tokens": 120}],
            "termination_metadata": [{"generated_tokens": 20}],
        },
        {
            "rank": 0,
            "rewards": [1.0, 2.0],
            "advantages": [-1.0, 1.0],
            "prompt_metadata": [{"final_tokens": 100}],
            "termination_metadata": [{"generated_tokens": 10}],
        },
    ]

    combined = _combine_rank_step_data(rank_data)

    assert combined["ranks"] == [0, 1]
    assert combined["rewards"] == [1.0, 2.0, 3.0, 4.0]
    assert combined["advantages"] == [-1.0, 1.0, 0.5, -0.5]
    assert [item["final_tokens"] for item in combined["prompt_metadata"]] == [100, 120]
    assert [item["generated_tokens"] for item in combined["termination_metadata"]] == [10, 20]


def test_grpo_rollout_rows_are_combined_with_all_ranks() -> None:
    rank_rollouts = [
        {"rank": 1, "rows": [{"rank": 1, "prompt": "rank-1"}]},
        {"rank": 0, "rows": [{"rank": 0, "prompt": "rank-0"}]},
    ]

    rows = _combine_rank_rollout_rows(rank_rollouts)

    assert [row["rank"] for row in rows] == [0, 1]
    assert [row["prompt"] for row in rows] == ["rank-0", "rank-1"]


def test_parse_pid_accepts_named_values() -> None:
    pid = parse_pid("P:1.2; I:0.03; D:4")

    assert pid == PIDParams(1.2, 0.03, 4.0)
    assert format_compliance_score("Kp=1.2, Ki=0.03, Kd=4") == 1.0


def test_normalize_group_advantages() -> None:
    advantages = normalize_group_advantages([1.0, 2.0, 10.0, 10.0], group_size=2)

    assert advantages == [-1.0, 1.0, 0.0, 0.0]


def test_evaluate_completion_returns_reward() -> None:
    settings = SimulationSettings(sim_time=20.0, num_points=101, time_delay=1.0)
    sample = _reward_sample(settings)

    result = evaluate_completion("Kp=1.0, Ki=0.1, Kd=0.01", sample, settings, RewardConfig())

    assert result.parsed_pid == PIDParams(1.0, 0.1, 0.01)
    assert result.metrics is not None
    assert -2.0 <= result.reward <= 1.0


def test_length_limited_complete_pid_keeps_physical_reward_with_zero_format() -> None:
    settings = SimulationSettings(sim_time=20.0, num_points=101, time_delay=1.0)
    sample = _reward_sample(settings)

    result = evaluate_completion(
        "P:1.0; I:0.1; D:0.01",
        sample,
        settings,
        RewardConfig(),
        completion_hit_length_limit=True,
    )

    assert result.parsed_pid == PIDParams(1.0, 0.1, 0.01)
    assert result.metrics is not None
    assert result.components["format"] == 0.0
    assert result.analysis["completion_hit_length_limit"] is True


def test_length_limited_incomplete_pid_uses_invalid_reward_without_simulation() -> None:
    settings = SimulationSettings(sim_time=20.0, num_points=101, time_delay=1.0)
    sample = _reward_sample(settings)

    result = evaluate_completion(
        "P:1.0; I:",
        sample,
        settings,
        RewardConfig(),
        completion_hit_length_limit=True,
    )

    assert result.reward == RewardConfig().invalid_reward
    assert result.parsed_pid is None
    assert result.metrics is None
    assert result.analysis["completion_hit_length_limit"] is True


def test_completion_termination_metadata_distinguishes_eos_and_length_limit() -> None:
    metadata = _completion_termination_metadata(
        [[10, 11, 2, 2], [10, 11, 12, 13]],
        eos_token_id=2,
        pad_token_id=2,
        max_completion_length=4,
    )

    assert metadata[0] == {
        "generated_tokens": 3,
        "terminated_by_eos": True,
        "hit_length_limit": False,
        "termination_reason": "eos",
    }
    assert metadata[1] == {
        "generated_tokens": 4,
        "terminated_by_eos": False,
        "hit_length_limit": True,
        "termination_reason": "length_limit",
    }


class _NumpyTorch:
    bool = np.bool_

    @staticmethod
    def zeros_like(values, dtype):
        return np.zeros_like(values, dtype=dtype)


def test_completion_mask_includes_first_eos_and_excludes_later_padding() -> None:
    completion_ids = np.asarray([[10, 11, 2, 2], [10, 11, 12, 13]])
    metadata = _completion_termination_metadata(
        completion_ids.tolist(),
        eos_token_id=2,
        pad_token_id=2,
        max_completion_length=4,
    )

    mask = _build_completion_mask(completion_ids, metadata, _NumpyTorch)

    assert mask.tolist() == [
        [True, True, True, False],
        [True, True, True, True],
    ]


def test_nonfinite_candidate_uses_unstable_reward_branch() -> None:
    settings = SimulationSettings(sim_time=20.0, num_points=201, time_delay=1.0, max_abs_output=2.0)
    plant = PlantSpec(plant_type="first_order", k=1.0, t=1.0, time_delay=1.0, setpoint=1.0)
    baseline_pid = PIDParams(1.0, 0.1, 0.01)
    baseline = simulate_pid(plant, baseline_pid, settings)
    sample = PromptSample(
        plant=plant,
        current_pid=baseline_pid,
        current_metrics=baseline.metrics,
        messages=build_messages(plant, baseline_pid, baseline.metrics),
    )

    result = evaluate_completion("Kp=100, Ki=100, Kd=100", sample, settings, RewardConfig())

    assert result.metrics is not None
    assert not result.metrics.finite
    assert result.reward == -1.0
    assert "non-finite" in result.analysis["error"]


def test_controller_phase_uses_pid_real_and_imaginary_parts() -> None:
    pid = PIDParams(kp=2.0, ki=4.0, kd=3.0)
    omega = 5.0

    phase = _controller_phase_radians(pid, omega)

    expected = math.atan2(pid.kd * omega - pid.ki / omega, pid.kp)
    assert math.isclose(phase, expected, rel_tol=0.0, abs_tol=1e-12)


def test_full_loop_phase_margin_includes_process_and_delay() -> None:
    pid = PIDParams(kp=1.0, ki=0.2, kd=0.05)
    plant = PlantSpec(plant_type="second_order", k=1.0, tau1=1.0, tau2=2.0, time_delay=0.5)

    phase_margin, crossover, analysis = _open_loop_phase_margin(pid, plant)

    assert phase_margin is not None
    assert crossover is not None and crossover > 0
    assert analysis["crossover_count"] >= 1
    controller_only = _controller_phase_radians(pid, crossover)
    assert math.isclose(
        analysis["controller_phase_deg"], math.degrees(controller_only), abs_tol=1e-12
    )
    assert analysis["loop_phase_deg"] < analysis["controller_phase_deg"]


def test_fopdt_delayed_pid_uses_first_order_pade_cubic() -> None:
    plant = PlantSpec(plant_type="first_order", k=2.0, t=3.0, time_delay=4.0)
    pid = PIDParams(kp=1.5, ki=0.25, kd=0.5)

    coefficients, metadata = characteristic_polynomial(pid, plant, RewardConfig())

    assert metadata["delay_model"] == "pade_1_1"
    assert metadata["degree"] == 3
    assert coefficients == [4.0, 0.0, 3.0, 0.5]


def test_sopdt_delayed_pid_uses_first_order_pade_quartic() -> None:
    plant = PlantSpec(plant_type="second_order", k=1.0, tau1=2.0, tau2=3.0, time_delay=2.0)
    pid = PIDParams(kp=1.0, ki=0.2, kd=0.4)

    coefficients, metadata = characteristic_polynomial(pid, plant, RewardConfig())

    assert metadata["degree"] == 4
    assert coefficients == [6.0, 10.6, 5.4, 1.8, 0.2]


def test_zero_integral_cancels_controller_denominator_before_routh() -> None:
    fopdt = PlantSpec(plant_type="first_order", k=1.0, t=2.0, time_delay=1.0)
    sopdt = PlantSpec(plant_type="second_order", k=1.0, tau1=2.0, tau2=3.0, time_delay=1.0)
    pid = PIDParams(kp=1.0, ki=0.0, kd=0.1)

    fopdt_coefficients, fopdt_metadata = characteristic_polynomial(pid, fopdt, RewardConfig())
    sopdt_coefficients, sopdt_metadata = characteristic_polynomial(pid, sopdt, RewardConfig())

    assert fopdt_metadata["controller_structure"] == "PD"
    assert fopdt_metadata["degree"] == 2
    assert len(fopdt_coefficients) == 3
    assert sopdt_metadata["degree"] == 3
    assert len(sopdt_coefficients) == 4


def test_routh_hurwitz_detects_stable_and_unstable_cubics() -> None:
    assert routh_hurwitz([1.0, 6.0, 11.0, 6.0], 1e-9)["status"] == "stable"
    assert routh_hurwitz([1.0, -1.0, 1.0, 1.0], 1e-9)["status"] == "unstable"


def test_nonpositive_phase_margin_is_a_safety_failure() -> None:
    plant = PlantSpec(plant_type="first_order", k=1.0, t=1.0, time_delay=3.0)
    pid = PIDParams(kp=20.0, ki=1.0, kd=0.0)

    _, safe, analysis = stability_proxy(pid, plant, RewardConfig())

    assert not safe
    assert analysis["safety_reason"] in {
        "phase_margin_nonpositive",
        "routh_unstable",
        "routh_marginal",
    }


def test_gain_reference_is_positive_even_when_imc_derivative_can_be_zero() -> None:
    config = RewardConfig(gain_calibration_samples=16)
    reference = calibrate_gain_reference(
        config, second_order_prob=0.0, time_delay=0.0, setpoint=1.0
    )

    assert reference.p > 0.0
    assert reference.i > 0.0
    assert reference.d >= config.gain_floor_min


def test_routh_nearzero_leading_coefficient_is_marginal() -> None:
    assert routh_hurwitz([1e-12, 1.0, 1.0], 1e-9)["status"] == "marginal"


def test_grpo_output_directory_allows_nonempty_resume_path(tmp_path: Path) -> None:
    output_dir = tmp_path / "grpo-output"
    output_dir.mkdir()
    (output_dir / "trainer_log.jsonl").write_text("existing", encoding="utf-8")

    assert _prepare_grpo_output_dir(output_dir, resume=True) == output_dir


def test_reward_branches_separate_success_from_stable_failure() -> None:
    from llmpidtuner.training.rewards import comprehensive_pid_reward

    plant = PlantSpec(
        plant_type="first_order",
        k=1.0,
        t=10.0,
        time_delay=1.0,
    )
    pid = PIDParams(kp=1.0, ki=0.1, kd=0.1)
    performance = {
        "overshoot": 5.0,
        "settling_time": 20.0,
        "steady_state_error": 0.1,
        "settling_reference": 20.0,
        "task_success": True,
    }

    success, success_detail = comprehensive_pid_reward(
        pid,
        plant,
        100.0,
        80.0,
        performance,
        1.0,
        RewardConfig(),
    )
    failed, failed_detail = comprehensive_pid_reward(
        pid,
        plant,
        100.0,
        80.0,
        {**performance, "task_success": False},
        1.0,
        RewardConfig(),
    )

    assert 0.5 <= success <= 1.0
    assert -0.5 <= failed <= 0.0
    assert success_detail["reward_branch"] == "success"
    assert failed_detail["reward_branch"] == "stable_failed"
