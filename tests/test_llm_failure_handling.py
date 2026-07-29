from pathlib import Path

import yaml

from llmpidtuner.config import CaseConfig
from llmpidtuner.models import FirstOrderPlant, PIDParams, SimulationSettings
from llmpidtuner.runner import run_case


def test_llm_context_limit_stop_is_recorded(monkeypatch, tmp_path: Path):
    calls = {"count": 0}

    def fake_call_pid_parameters(self, system_content, user_content, use_initial_model=True):
        calls["count"] += 1
        if calls["count"] == 1:
            return PIDParams(kp=1.0, ki=0.1, kd=0.01)
        raise RuntimeError("maximum context length exceeded")

    monkeypatch.setenv("LLM_API_KEY", "dummy")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setattr(
        "llmpidtuner.runner.PIDControllerClient.call_pid_parameters",
        fake_call_pid_parameters,
    )

    config = CaseConfig(
        name="tiny_llm_failure",
        system="first_order",
        mode="llm",
        output_dir=str(tmp_path),
        initial_pid=PIDParams(kp=1.0, ki=0.1, kd=0.01),
        simulation=SimulationSettings(setpoint=1.0, sim_time=20.0, num_points=201, time_delay=1.0),
        first_order=FirstOrderPlant(k=0.65, t=112.0),
        max_iterations=3,
        success_overshoot=-1.0,
        success_steady_state_error=-1.0,
    )

    output_dir = run_case(config)
    status = yaml.safe_load((output_dir / "run_status.yaml").read_text(encoding="utf-8"))

    assert calls["count"] == 2
    assert status["status"] == "llm_failed"
    assert status["stop_reason"] == "context_length_exceeded"
    assert status["completed_iterations"] == 1
    assert status["failed_next_llm_call"] == 2
    assert (output_dir / "value_curve_iteration_1.txt").exists()
    assert (output_dir / "pid_tuning_comparison.png").exists()