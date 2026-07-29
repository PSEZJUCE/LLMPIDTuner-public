from pathlib import Path

from llmpidtuner.frequency_analysis import FrequencyPlant, compare_frequency_response
from llmpidtuner.models import PIDParams


def test_exact_delay_frequency_comparison_writes_figure_and_summary(tmp_path: Path) -> None:
    plant = FrequencyPlant(system="first_order", k=0.65, t=112.0, time_delay=1.0)

    figure_path, summary_path, summaries = compare_frequency_response(
        plant,
        PIDParams(1.0, 0.1, 0.01),
        PIDParams(5.0, 0.04, 0.5),
        tmp_path / "frequency_comparison.png",
        summary_path=tmp_path / "frequency_comparison.json",
        points=512,
    )

    assert figure_path.exists()
    assert summary_path is not None and summary_path.exists()
    assert [summary.label for summary in summaries] == ["Base model", "GRPO model"]
    assert all(summary.phase_margin_deg is not None for summary in summaries)
    assert all(summary.critical_crossover_frequency is not None for summary in summaries)


def test_second_order_frequency_comparison_uses_exact_delay(tmp_path: Path) -> None:
    plant = FrequencyPlant(
        system="second_order",
        k=2.0,
        tau1=3.0,
        tau2=7.0,
        time_delay=1.0,
    )

    figure_path, _, summaries = compare_frequency_response(
        plant,
        PIDParams(1.0, 0.1, 0.01),
        PIDParams(2.0, 0.05, 1.0),
        tmp_path / "second_order_frequency_comparison.png",
        points=512,
    )

    assert figure_path.exists()
    assert len(summaries) == 2
    assert all(summary.all_crossovers for summary in summaries)
