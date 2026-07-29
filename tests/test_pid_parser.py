import pytest

from llmpidtuner.llm import parse_pid_parameters


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("P:1.2; I:0.03; D:4", (1.2, 0.03, 4.0)),
        ("P:1.2；I:0.03；D:4", (1.2, 0.03, 4.0)),
        ("Here are values:\nP: 1\nI: 0.1\nD: 0.01", (1.0, 0.1, 0.01)),
        ("P:-1.0; I:+0.2; D:3.5", (-1.0, 0.2, 3.5)),
        ("P:0.0015; I=1000; D:0.125", (0.0015, 1000.0, 0.125)),
        ("P=1e-3, I=2.5e-4, D=3E+2", (0.001, 0.00025, 300.0)),
        ("Kp=1.2, Ki=0.03, Kd=4", (1.2, 0.03, 4.0)),
    ],
)
def test_parse_pid_parameters(text, expected):
    pid = parse_pid_parameters(text)
    assert (pid.kp, pid.ki, pid.kd) == expected


def test_parse_pid_parameters_rejects_incomplete_response():
    with pytest.raises(ValueError):
        parse_pid_parameters("P:1; I:2")
