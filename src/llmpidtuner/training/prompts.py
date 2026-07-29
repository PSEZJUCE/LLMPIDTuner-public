from __future__ import annotations

import re
from typing import Any

from llmpidtuner.metrics import ControlSystemAnalysis
from llmpidtuner.models import PIDParams
from llmpidtuner.prompting import (
    DEFAULT_SYSTEM_ROLE,
    build_feedback_prompt_text,
    build_initial_prompt_text,
    format_pid_response,
)
from llmpidtuner.training.simulation import PlantSpec, ResponseMetrics, TrainingSimulationResult


SYSTEM_PROMPT = DEFAULT_SYSTEM_ROLE


def build_messages(
    plant: PlantSpec,
    current_pid: PIDParams,
    current_metrics: ResponseMetrics,
    response_description: str | None = None,
    demonstration_cases: str | None = None,
    control_style: str = "balanced",
) -> list[dict[str, Any]]:
    description = response_description or metrics_description_for_prompt(current_metrics)
    user = build_initial_prompt_text(
        current_pid,
        current_metrics.iae,
        description,
        demonstration_cases=demonstration_cases,
        time_delay=plant.time_delay,
        control_style=control_style,
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def format_pid(pid: PIDParams) -> str:
    return format_pid_response(pid)


def build_feedback_message(
    pid: PIDParams,
    metrics: ResponseMetrics,
    response_description: str,
    time_delay: float,
    control_style: str = "balanced",
) -> dict[str, str]:
    return {
        "role": "user",
        "content": build_feedback_prompt_text(
            pid,
            metrics.iae,
            response_description,
            time_delay=time_delay,
            control_style=control_style,
        ),
    }


def build_target_message(
    target_pid: PIDParams,
) -> dict[str, str]:
    return {
        "role": "assistant",
        "content": format_pid_response(target_pid),
    }


def response_description(
    result: TrainingSimulationResult,
    time_delay: float = 0.0,
    prompt_variant: str = "full",
) -> str:
    return ControlSystemAnalysis.from_arrays(
        result.time,
        result.setpoint,
        result.output,
        filename="<training-simulation>",
        time_delay=time_delay,
    ).generate_description(prompt_variant)


def metrics_description_for_prompt(metrics: ResponseMetrics) -> str:
    settling = (
        "The system output does not settle within +/-5% of the steady-state value."
        if not metrics.settled
        else f"The settling time (tc) is {metrics.settling_time:.2f} seconds."
    )
    return (
        f"Overshoot: {metrics.overshoot_pct:.2f}%.\n"
        f"Oscillations: {metrics.oscillation_count}.\n"
        f"{settling}\n"
        f"Steady-State Error: {metrics.steady_state_error_pct:.2f}%."
    )


def messages_to_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    if getattr(tokenizer, "chat_template", None) and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    system = ""
    user_parts: list[str] = []
    for message in messages:
        if message["role"] == "system":
            system = message["content"]
        elif message["role"] == "user":
            user_parts.append(message["content"])
    prompt = ""
    if system:
        prompt += f"System: {system}<|im_end|>\n"
    prompt += "Human: " + "\n".join(user_parts) + "<|im_end|>\nAssistant:"
    return prompt


def fit_messages_to_prompt_budget(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    max_length: int,
) -> tuple[str, dict[str, int | bool]]:
    """Fit a PID prompt by removing whole demonstration experiments only."""

    if max_length <= 0:
        raise ValueError("max_length must be positive.")

    prompt = messages_to_prompt(tokenizer, messages)
    original_tokens = _prompt_token_count(tokenizer, prompt)
    user_index, prefix, demonstration, suffix = _demonstration_parts(messages)
    experiments = _split_demonstration_experiments(demonstration)
    original_cases = len(experiments)
    if original_tokens <= max_length:
        return prompt, {
            "original_tokens": original_tokens,
            "final_tokens": original_tokens,
            "demonstration_cases_original": original_cases,
            "demonstration_cases_used": original_cases,
            "trimmed": False,
        }

    if user_index is None:
        raise ValueError(
            f"GRPO prompt has {original_tokens} tokens, exceeding max_prompt_length={max_length}, "
            "and its demonstration section cannot be isolated safely."
        )

    for used_cases in range(original_cases - 1, -1, -1):
        retained = experiments[:used_cases]
        replacement = (
            "\n\n".join(retained)
            if retained
            else "Demonstration cases omitted to fit the prompt token budget."
        )
        fitted_messages = [dict(message) for message in messages]
        fitted_messages[user_index]["content"] = f"{prefix}\n{replacement}\n\n{suffix}"
        fitted_prompt = messages_to_prompt(tokenizer, fitted_messages)
        final_tokens = _prompt_token_count(tokenizer, fitted_prompt)
        if final_tokens <= max_length:
            return fitted_prompt, {
                "original_tokens": original_tokens,
                "final_tokens": final_tokens,
                "demonstration_cases_original": original_cases,
                "demonstration_cases_used": used_cases,
                "trimmed": True,
            }

    raise ValueError(
        f"Core GRPO prompt exceeds max_prompt_length={max_length} even after all "
        "demonstration experiments were removed."
    )


def _demonstration_parts(
    messages: list[dict[str, Any]],
) -> tuple[int | None, str, str, str]:
    demo_marker = "[Demonstration Case]"
    current_marker = "[Current State Description]"
    for index, message in enumerate(messages):
        if message.get("role") != "user":
            continue
        content = str(message.get("content", ""))
        demo_start = content.find(demo_marker)
        current_start = content.find(current_marker, demo_start + len(demo_marker))
        if demo_start < 0 or current_start < 0:
            continue
        prefix = content[: demo_start + len(demo_marker)].rstrip()
        demonstration = content[demo_start + len(demo_marker) : current_start].strip()
        suffix = content[current_start:].lstrip()
        return index, prefix, demonstration, suffix
    return None, "", "", ""


def _split_demonstration_experiments(text: str) -> list[str]:
    matches = list(re.finditer(r"(?m)^Experiment\s+\d+\s*:", text))
    if not matches:
        stripped = text.strip()
        return [stripped] if stripped and stripped != "Demo cases not provided." else []
    return [
        text[
            match.start() : matches[index + 1].start() if index + 1 < len(matches) else None
        ].strip()
        for index, match in enumerate(matches)
    ]


def _prompt_token_count(tokenizer: Any, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False))
