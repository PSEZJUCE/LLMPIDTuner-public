from __future__ import annotations

from pathlib import Path

from llmpidtuner.metrics import ControlSystemAnalysis
from llmpidtuner.models import PIDParams


DEFAULT_SYSTEM_ROLE = (
    "You are an expert in control systems, specializing in PID controller tuning. "
    "Your task is to provide tuning recommendations for the PID parameters (P, I, D) "
    "based on the user's prompts, which include a description of the system's current "
    "state, the task objective, and sample cases. Observing the system response, offer "
    "specific adjustments to the PID parameters that will improve the control performance "
    "according to the requirements."
)

BACKGROUND_DESCRIPTION = (
    "You are an assistant for adjusting a PID controller. Your task is to adjust the PID "
    "control parameters based on the given system response curve to optimize system "
    "performance. The controller's performance is evaluated by the difference between the "
    "system response and the target value, with the goal of minimizing the IAE (Integral "
    "of Absolute Error). The smaller the IAE, the better the controller's performance. "
    "Based on the current system state, PID parameters and IAE, you need to provide new "
    "PID parameters to improve system performance."
)


def read_pid_iae(path: str | Path) -> tuple[PIDParams, float]:
    content = Path(path).read_text().strip()
    params = content.split(", ")
    kp = float(params[0].split("=")[1])
    ki = float(params[1].split("=")[1])
    kd = float(params[2].split("=")[1])
    iae = float(params[3].split("=")[1])
    return PIDParams(kp, ki, kd), iae


def build_initial_prompt(
    curve_file: str | Path,
    pid_file: str | Path,
    demonstration_path: str | Path | None = None,
    time_delay: float | None = None,
    prompt_variant: str = "full",
    control_style: str = "balanced",
) -> str:
    pid, iae = read_pid_iae(pid_file)
    description = ControlSystemAnalysis(str(curve_file), time_delay or 0.0).generate_description(prompt_variant)
    demonstration_cases = _read_demonstration(demonstration_path)
    return build_initial_prompt_text(
        pid,
        iae,
        description,
        demonstration_cases=demonstration_cases,
        time_delay=time_delay,
        control_style=control_style,
    )


def build_initial_prompt_text(
    pid: PIDParams,
    iae: float,
    description: str,
    demonstration_cases: str | None = None,
    time_delay: float | None = None,
    control_style: str = "balanced",
) -> str:
    """Build the canonical first-turn PID prompt from in-memory values."""

    demonstration_cases = demonstration_cases or "Demo cases not provided."
    style_line = f"Desired control style: {control_style}."
    delay_line = (
        f"\nObserved process dead time: {time_delay:.4g} seconds."
        if time_delay is not None
        else ""
    )

    return f"""
[Background Description]
{BACKGROUND_DESCRIPTION}

[Demonstration Case]
{demonstration_cases}

[Current State Description]
Current PID parameters and IAE: Kp={pid.kp:.6g}, Ki={pid.ki:.6g}, Kd={pid.kd:.6g}, current IAE={iae:.6g}{delay_line}
{style_line}
{description}

[Task]
Please provide a new set of PID parameters that bring the response curve closer to the target value and further reduce the IAE. Just output the three PID parameters, no additional content is required. Please output in the following format: P:xx; I:xx; D:xx
"""


def build_feedback_prompt(
    curve_file: str | Path,
    pid_file: str | Path,
    time_delay: float | None = None,
    prompt_variant: str = "full",
    control_style: str = "balanced",
) -> str:
    pid, iae = read_pid_iae(pid_file)
    description = ControlSystemAnalysis(
        str(curve_file), time_delay or 0.0
    ).generate_description(prompt_variant)
    return build_feedback_prompt_text(
        pid,
        iae,
        description,
        time_delay=time_delay,
        control_style=control_style,
    )


def build_feedback_prompt_text(
    pid: PIDParams,
    iae: float,
    description: str,
    time_delay: float | None = None,
    control_style: str = "balanced",
) -> str:
    """Build the canonical follow-up prompt from in-memory values."""

    delay_line = (
        f"\nObserved process dead time: {time_delay:.4g} seconds."
        if time_delay is not None
        else ""
    )
    style_line = f"Desired control style: {control_style}."
    return f"""
[Feedback Example]
After applying the new PID parameters you given (Kp={pid.kp:.6g}, Ki={pid.ki:.6g}, Kd={pid.kd:.6g}), the system response curve can be described as follows:{delay_line}
{style_line}
{description}
At this point, the IAE is {iae:.6g}

[Task]
Please provide a new set of PID parameters that bring the response curve closer to the target value and further reduce the IAE. Just output the three PID parameters, no additional content is required. Please output in the following format: P:xx; I:xx; D:xx
"""


def format_pid_response(pid: PIDParams) -> str:
    """Return the one PID wire format accepted by training and inference."""

    return f"P:{pid.kp:.6g}; I:{pid.ki:.6g}; D:{pid.kd:.6g}"


def _read_demonstration(path: str | Path | None) -> str:
    if path is None:
        return "Demo cases not provided."
    demo_path = Path(path)
    if not demo_path.exists():
        return "Demo cases not provided."
    return demo_path.read_text(encoding="utf-8")
