from llmpidtuner.models import FirstOrderPlant
from llmpidtuner.runner import _batch_folder_name, _batch_group_complete


def test_batch_folder_name_includes_group_to_avoid_rounded_collisions():
    first = FirstOrderPlant(k=0.65362477, t=411)
    second = FirstOrderPlant(k=0.64726465, t=411)

    assert _batch_folder_name(37, first) == "group_037_example_K_0.65_T_411"
    assert _batch_folder_name(79, second) == "group_079_example_K_0.65_T_411"
    assert _batch_folder_name(37, first) != _batch_folder_name(79, second)


def test_batch_group_complete_requires_plot_and_iteration_curve(tmp_path):
    folder = tmp_path / "group_001"
    folder.mkdir()
    assert _batch_group_complete(folder) is False

    (folder / "value_curve_iteration_1.txt").write_text("", encoding="utf-8")
    assert _batch_group_complete(folder) is False

    (folder / "pid_tuning_comparison.png").write_text("", encoding="utf-8")
    assert _batch_group_complete(folder) is True
