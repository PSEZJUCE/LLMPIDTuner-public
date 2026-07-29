from pathlib import Path

import pytest

from llmpidtuner.batch_analysis import analyze_batch_paths
from llmpidtuner.experiment_protocol import PROTOCOL_ID


OTHER_PROTOCOL = "independent-protocol"


def _write_batch(root: Path, protocol: str | None = None, *, imc: bool = False) -> None:
    case = root / "case_001"
    case.mkdir(parents=True)
    (case / "value_curve.txt").write_text(
        "Results Array (Time, Setpoint, Output):\n"
        "Time Setpoint Output\n"
        "0.00 1.00000 0.00000\n"
        "1.00 1.00000 1.00000\n",
        encoding="utf-8",
    )
    if protocol:
        (root / "demonstration_metadata.yaml").write_text(
            f"demonstration_protocol: {protocol}\n",
            encoding="utf-8",
        )
    if imc:
        (root / "imc_metadata.txt").write_text("method=imc\n", encoding="utf-8")


def test_batch_analysis_reads_protocol_from_group_metadata(tmp_path: Path) -> None:
    root = tmp_path / "batch"
    _write_batch(root)
    metadata_path = root / "case_001" / "demonstration_metadata.yaml"
    metadata_path.write_text(
        f"demonstration_protocol: {PROTOCOL_ID}\n",
        encoding="utf-8",
    )

    results, summaries = analyze_batch_paths([root])

    assert {result.demonstration_protocol for result in results} == {PROTOCOL_ID}
    assert summaries[0].demonstration_protocol == PROTOCOL_ID


def test_batch_analysis_rejects_multiple_protocols_within_one_batch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "batch"
    _write_batch(root)
    first_metadata = root / "case_001" / "demonstration_metadata.yaml"
    first_metadata.write_text(
        f"demonstration_protocol: {PROTOCOL_ID}\n",
        encoding="utf-8",
    )
    second = root / "case_002"
    second.mkdir()
    (second / "demonstration_metadata.yaml").write_text(
        f"demonstration_protocol: {OTHER_PROTOCOL}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="multiple demonstration protocols"):
        analyze_batch_paths([root])


def test_batch_analysis_rejects_mixed_demonstration_protocols(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_batch(first, PROTOCOL_ID)
    _write_batch(second, OTHER_PROTOCOL)

    with pytest.raises(ValueError, match="different demonstration protocols"):
        analyze_batch_paths([first, second])


def test_batch_analysis_can_explicitly_compare_protocols(tmp_path: Path) -> None:
    current = tmp_path / "current"
    other = tmp_path / "other"
    _write_batch(current, PROTOCOL_ID)
    _write_batch(other, OTHER_PROTOCOL)

    results, summaries = analyze_batch_paths(
        [current, other], allow_mixed_protocols=True
    )
    assert len(results) == 2
    assert len(summaries) == 2


def test_batch_analysis_allows_imc_baseline_with_current_protocol(tmp_path: Path) -> None:
    llm = tmp_path / "llm"
    imc = tmp_path / "imc"
    _write_batch(llm, PROTOCOL_ID)
    _write_batch(imc, imc=True)

    results, _ = analyze_batch_paths([llm, imc])
    assert {result.demonstration_protocol for result in results} == {
        PROTOCOL_ID,
        "not-applicable",
    }
