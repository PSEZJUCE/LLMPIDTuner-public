import json
from pathlib import Path

import pytest
import yaml

from llmpidtuner.experiment_protocol import FAULT_TYPES
from llmpidtuner.models import SimulationSettings
from llmpidtuner.training.data import (
    PIDPromptGenerator,
    load_protocol_prompt_samples,
    write_sft_messages_dataset,
)
from llmpidtuner.training.prompts import fit_messages_to_prompt_budget, messages_to_prompt


SOURCE_ROOT = Path("cases/protocol/perturbed_imc_delay_stratified/sources")


def test_frozen_evaluation_sources_satisfy_protocol_quotas() -> None:
    for system in ("first_order", "second_order"):
        data = yaml.safe_load((SOURCE_ROOT / f"evaluation_{system}.yaml").read_text("utf-8"))
        rows = data["cases"]
        assert len(rows) == 100
        assert {row["fault_type"] for row in rows} == set(FAULT_TYPES)
        assert {
            fault: sum(row["fault_type"] == fault for row in rows) for fault in FAULT_TYPES
        } == {fault: 10 for fault in FAULT_TYPES}
        assert {
            level: sum(row["severity"] == level for row in rows)
            for level in ("mild", "moderate", "severe")
        } == {"mild": 30, "moderate": 50, "severe": 20}
        assert all(
            not row["initial_metrics"]["settled"]
            or (
                row["initial_metrics"]["overshoot_pct"] >= 15
                or row["initial_metrics"]["steady_state_error_pct"] >= 1
            )
            for row in rows
        )
        assert all(
            row["initial_metrics"]["iae"] / row["reference_metrics"]["iae"] >= 1.5 for row in rows
        )


def test_load_frozen_validation_prompts_and_write_schema_v3_sft(tmp_path: Path) -> None:
    settings = SimulationSettings(max_abs_output=3.0)
    samples = load_protocol_prompt_samples(
        [SOURCE_ROOT / "grpo_validation_first_order.yaml"],
        simulation=settings,
    )
    assert len(samples) == 100
    assert all(sample.plant.plant_type == "first_order" for sample in samples)
    assert all(not sample.current_metrics.converged() for sample in samples)

    output = tmp_path / "sft.jsonl"
    count = write_sft_messages_dataset(
        samples[:2],
        output,
        simulation=settings,
        feedback_sample_probability=0.0,
        include_target_metrics=True,
    )
    rows = [json.loads(line) for line in output.read_text("utf-8").splitlines()]
    assert count == 2
    assert all(row["metadata"]["schema_version"] == 3 for row in rows)
    assert all(row["metadata"]["sample_kind"] == "initial" for row in rows)
    assert all(row["metadata"]["control_style"] == "balanced" for row in rows)
    assert all("reasoning_content" not in row["messages"][-1] for row in rows)


def test_online_grpo_generator_creates_balanced_first_turn_tasks() -> None:
    generator = PIDPromptGenerator(
        seed=91001,
        simulation=SimulationSettings(max_abs_output=3.0),
        second_order_prob=0.0,
    )
    first, second = generator.sample_batch(2)

    assert first.plant != second.plant
    assert first.control_style == second.control_style == "balanced"
    assert not first.current_metrics.converged()
    assert not second.current_metrics.converged()


class _CharacterTokenizer:
    chat_template = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text)))


def test_grpo_prompt_budget_removes_only_whole_demonstration_cases() -> None:
    tokenizer = _CharacterTokenizer()
    experiments = [f"Experiment {index}:\ncase {index} details" for index in range(1, 4)]
    demonstration = "\n\n".join(experiments)
    user_content = (
        "[Demonstration Case]\n"
        f"{demonstration}\n\n"
        "[Current State Description]\n"
        "Current PID and response must be retained.\n\n"
        "[Task]\nReturn P, I, D."
    )
    messages = [
        {"role": "system", "content": "PID system role"},
        {"role": "user", "content": user_content},
    ]
    one_case_messages = [
        messages[0],
        {"role": "user", "content": user_content.replace(demonstration, experiments[0])},
    ]
    one_case_budget = len(tokenizer.encode(messages_to_prompt(tokenizer, one_case_messages)))

    prompt, metadata = fit_messages_to_prompt_budget(tokenizer, messages, one_case_budget)

    assert metadata["demonstration_cases_used"] == 1
    assert metadata["trimmed"] is True
    assert "Experiment 1:" in prompt
    assert "Experiment 2:" not in prompt
    assert "[Current State Description]" in prompt
    assert "[Task]" in prompt


def test_grpo_prompt_budget_rejects_oversized_core_prompt() -> None:
    tokenizer = _CharacterTokenizer()
    messages = [
        {"role": "system", "content": "PID system role"},
        {
            "role": "user",
            "content": (
                "[Demonstration Case]\nExperiment 1:\nshort\n\n"
                "[Current State Description]\n" + "required " * 30 + "\n[Task]\nReturn P, I, D."
            ),
        },
    ]
    with pytest.raises(ValueError, match="Core GRPO prompt exceeds"):
        fit_messages_to_prompt_budget(tokenizer, messages, max_length=40)


def test_online_prompt_generator_state_round_trip_is_exact() -> None:
    settings = SimulationSettings(max_abs_output=3.0)
    original = PIDPromptGenerator(
        seed=91001,
        simulation=settings,
        second_order_prob=0.5,
    )
    original.sample()
    state = original.state_dict()
    expected = original.sample().as_dict()

    restored = PIDPromptGenerator(
        seed=1,
        simulation=settings,
        second_order_prob=0.5,
    )
    restored.load_state_dict(state)

    assert restored.sample().as_dict() == expected
