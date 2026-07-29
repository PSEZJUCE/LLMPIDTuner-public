from pathlib import Path

from llmpidtuner.batch_analysis import (
    OVERSHOOT_BINS,
    STEADY_STATE_ERROR_BINS,
    _bucket_counts,
    analyze_batch_path,
    summarize_batch_results,
    write_batch_analysis,
)


def test_analyze_batch_path_marks_converged_and_unconverged(tmp_path: Path):
    batch_dir = tmp_path / "batch_a"
    converged_case = batch_dir / "case_converged"
    unconverged_case = batch_dir / "case_unconverged"

    _write_curve(converged_case / "value_curve.txt", [0.0, 0.6] + [1.7] * 100)
    _write_curve(converged_case / "value_curve_iteration_1.txt", [0.0, 0.7] + [1.0] * 100)
    _write_curve(unconverged_case / "value_curve.txt", [0.0, 0.8] + [1.4] * 100)
    _write_curve(unconverged_case / "value_curve_iteration_1.txt", [0.0, 0.5] + [1.5] * 100)
    _write_curve(unconverged_case / "value_curve_iteration_2.txt", [0.0, 0.4] + [1.3] * 100)

    results = analyze_batch_path(batch_dir, label="model_a")
    summaries = summarize_batch_results(results)

    by_case = {result.case_id: result for result in results}
    assert by_case["case_converged"].converged is True
    assert by_case["case_converged"].convergence_iteration == 1
    assert by_case["case_unconverged"].converged is False
    assert by_case["case_unconverged"].final_iteration == 2

    assert summaries[0].total == 2
    assert summaries[0].converged == 1
    assert summaries[0].unconverged == 1

    figure_path, _ = write_batch_analysis(
        results,
        summaries,
        tmp_path / "publication_summary.png",
    )
    assert figure_path.exists()


def test_analyze_batch_path_records_context_limit_stop(tmp_path: Path):
    batch_dir = tmp_path / "batch_a"
    stopped_case = batch_dir / "case_context_limit"

    _write_curve(stopped_case / "value_curve.txt", [0.0, 0.8, 1.4])
    _write_curve(stopped_case / "value_curve_iteration_5.txt", [0.0, 0.5, 1.3])
    (stopped_case / "run_status.yaml").write_text(
        "\n".join(
            [
                "status: llm_failed",
                "stop_reason: context_length_exceeded",
                "completed_iterations: 5",
                "max_iterations: 8",
                "failed_next_llm_call: 6",
                "converged: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    results = analyze_batch_path(batch_dir, label="base_model")
    summaries = summarize_batch_results(results)

    assert results[0].converged is False
    assert results[0].final_iteration == 5
    assert results[0].stop_reason == "context_length_exceeded"
    assert results[0].failed_next_llm_call == 6
    assert summaries[0].llm_failed == 1
    assert summaries[0].context_length_exceeded == 1


def _write_curve(path: Path, outputs: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Results Array (Time, Setpoint, Output):",
        "Time Setpoint Output",
    ]
    lines.extend(f"{index:.2f} 1.00000 {output:.5f}" for index, output in enumerate(outputs))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_metric_bucket_counts():
    assert _bucket_counts([-1.0, 0.0, 5.0, 5.1, 10.0, 14.9, 15.1], OVERSHOOT_BINS) == [
        3,
        2,
        1,
        1,
    ]
    assert _bucket_counts(
        [0.0, 0.001, 0.002, 0.01, 0.02, 0.1, 0.5, 1.0, 1.1],
        STEADY_STATE_ERROR_BINS,
    ) == [2, 2, 2, 2, 1]


def test_analysis_horizon_reuses_run_for_pass_at_one_and_two(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch"
    case_dir = batch_dir / "case"
    _write_curve(case_dir / "value_curve.txt", [0.0, 0.8] + [1.4] * 100)
    _write_curve(case_dir / "value_curve_iteration_1.txt", [0.0, 0.8] + [1.3] * 100)
    _write_curve(case_dir / "value_curve_iteration_2.txt", [0.0, 0.8] + [1.0] * 100)

    pass_at_one = analyze_batch_path(batch_dir, horizon=1)[0]
    pass_at_two = analyze_batch_path(batch_dir, horizon=2)[0]

    assert not pass_at_one.converged
    assert pass_at_one.final_iteration == 1
    assert pass_at_one.analysis_horizon == 1
    assert pass_at_two.converged
    assert pass_at_two.convergence_iteration == 2
    assert pass_at_two.final_iteration == 2
