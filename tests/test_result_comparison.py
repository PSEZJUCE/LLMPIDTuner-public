from pathlib import Path

from llmpidtuner.result_comparison import (
    compare_result_paths,
    resolve_batch_group_paths,
)


def test_compare_result_paths_uses_origin_and_latest_iteration(tmp_path: Path):
    llama_dir = tmp_path / "llama"
    deepseek_dir = tmp_path / "deepseek"
    imc_dir = tmp_path / "imc"

    _write_curve(llama_dir / "value_curve.txt", [0.0, 0.4, 1.5])
    _write_curve(llama_dir / "value_curve_iteration_1.txt", [0.0, 0.8, 1.1])
    _write_curve(llama_dir / "value_curve_iteration_2.txt", [0.0, 0.9, 1.02])
    _write_curve(deepseek_dir / "value_curve.txt", [0.0, 0.5, 1.6])
    _write_curve(deepseek_dir / "value_curve_iteration_1.txt", [0.0, 1.2, 1.0])
    _write_curve(imc_dir / "value_curve.txt", [0.0, 0.3, 1.4])
    _write_curve(imc_dir / "value_curve_iteration_1.txt", [0.0, 0.7, 1.0])

    figure_path, csv_path, summaries = compare_result_paths(
        [llama_dir, deepseek_dir, imc_dir],
        labels=["Llama", "DeepSeek", "IMC"],
        output_path=tmp_path / "compare.png",
        metrics_csv=tmp_path / "compare.csv",
    )

    assert figure_path.exists()
    assert csv_path is not None
    assert csv_path.exists()
    assert [summary.label for summary in summaries] == ["Llama", "DeepSeek", "IMC"]
    assert summaries[0].source_file.name == "value_curve_iteration_2.txt"
    assert summaries[1].source_file.name == "value_curve_iteration_1.txt"


def test_compare_result_paths_supports_broken_time_axis(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_curve(first / "value_curve.txt", [0.0, 0.8, 1.0, 1.0, 1.0])
    _write_curve(first / "value_curve_iteration_1.txt", [0.0, 0.9, 1.0, 1.0, 1.0])
    _write_curve(second / "value_curve.txt", [0.0, 0.7, 1.0, 1.0, 1.0])
    _write_curve(second / "value_curve_iteration_1.txt", [0.0, 0.95, 1.0, 1.0, 1.0])

    figure_path, _, _ = compare_result_paths(
        [first, second],
        output_path=tmp_path / "broken.png",
        x_break=(1.5, 3.0),
    )

    assert figure_path.exists()


def test_resolve_batch_group_paths_selects_same_group_from_each_batch(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_group = first / "group_007_example_K_0.65_T_112"
    second_group = second / "group_007_example_K_0.65_T_112"
    _write_curve(first_group / "value_curve.txt", [0.0, 0.8, 1.0])
    _write_curve(second_group / "value_curve.txt", [0.0, 0.9, 1.0])

    assert resolve_batch_group_paths([first, second], 7) == [
        first_group,
        second_group,
    ]


def _write_curve(path: Path, outputs: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Results Array (Time, Setpoint, Output):",
        "Time Setpoint Output",
    ]
    lines.extend(f"{index:.2f} 1.00000 {output:.5f}" for index, output in enumerate(outputs))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
