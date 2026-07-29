from pathlib import Path

import yaml

from llmpidtuner.config import load_case_config
from llmpidtuner.models import FirstOrderPlant, SecondOrderPlant
from llmpidtuner.runner import _batch_plants, write_batch_plants_from_config


def test_exported_batch_plants_match_seeded_first_order_case(tmp_path: Path) -> None:
    config = load_case_config("cases/eval/first_order_100_deepseek_v4_flash.yaml")
    output_path = write_batch_plants_from_config(config, tmp_path / "first_order.yaml")
    exported = yaml.safe_load(output_path.read_text(encoding="utf-8"))["plants"]
    seeded = _batch_plants(config.batch or {}, "first_order")

    assert len(exported) == 100
    assert exported[0]["group"] == 1
    assert exported[0]["k"] == seeded[0][1].k
    assert exported[0]["t"] == seeded[0][1].t


def test_batch_plants_can_load_fixed_first_and_second_order_lists(tmp_path: Path) -> None:
    first_path = tmp_path / "first.yaml"
    first_path.write_text("plants:\n- group: 7\n  k: 0.5\n  t: 123\n", encoding="utf-8")
    second_path = tmp_path / "second.yaml"
    second_path.write_text(
        "plants:\n- group: 8\n  k: 1.5\n  tau1: 2.0\n  tau2: 3.0\n",
        encoding="utf-8",
    )

    first = _batch_plants({"plants_path": str(first_path)}, "first_order")
    second = _batch_plants({"plants_path": str(second_path)}, "second_order")

    assert first == [(7, FirstOrderPlant(k=0.5, t=123.0))]
    assert second == [(8, SecondOrderPlant(k=1.5, tau1=2.0, tau2=3.0))]
