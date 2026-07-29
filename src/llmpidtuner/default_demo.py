from __future__ import annotations

FIRST_ORDER_DEFAULT_DEMONSTRATION = """Experiment 1:
Current PID parameters and IAE: Kp=1.000, Ki=0.100, Kd=0.010, Integral Absolute Error (IAE): 89.54
Overshoot: After reaching the peak value, the curve shows an overshoot of 37.69%, indicating The overshoot is severe.
Oscillations: The curve contains 2 oscillations. The curve shows noticeable oscillations.
Attenuation Ratio: The attenuation ratio between the first and second peaks is 0.76.
Oscillation Period: The period of oscillation is 274.01 seconds.
The time constant (T) is 48.00 seconds, indicating the time taken for the output to reach 63.2% of the setpoint.
The settling time (tc) is 407.02 seconds, indicating the time when the output enters within ±5% of the steady-state value and does not leave again.
The control performance is good; the Steady-State Error is within 1%, its value is 0.00%.
Suggested new PID parameters: P=2.06, I=0.037, D=8.43

Experiment 2:
Current PID parameters and IAE: Kp=1.000, Ki=0.100, Kd=0.010, Integral Absolute Error (IAE): 199.11
Overshoot: After reaching the peak value, the curve shows an overshoot of 46.19%, indicating The overshoot is severe.
Oscillations: The curve contains 2 oscillations. The curve shows noticeable oscillations.
Attenuation Ratio: The attenuation ratio between the first and second peaks is 0.75.
Oscillation Period: The period of oscillation is 475.03 seconds.
The time constant (T) is 89.00 seconds, indicating the time taken for the output to reach 63.2% of the setpoint.
The settling time (tc) is 786.04 seconds, indicating the time when the output enters within ±5% of the steady-state value and does not leave again.
The control performance is good; the Steady-State Error is within 1%, its value is 0.14%.
Suggested new PID parameters: P=3.40, I=0.032, D=26.16

Experiment 3:
Current PID parameters and IAE: Kp=1.000, Ki=0.100, Kd=0.010, Integral Absolute Error (IAE): 320.90
Overshoot: After reaching the peak value, the curve shows an overshoot of 61.97%, indicating The overshoot is severe.
Oscillations: The curve contains 5 oscillations. Oscillation is very violent and the curve fluctuates greatly.
Attenuation Ratio: The attenuation ratio between the first and second peaks is 0.76.
Oscillation Period: The period of oscillation is 498.03 seconds.
The time constant (T) is 91.00 seconds, indicating the time taken for the output to reach 63.2% of the setpoint.
The settling time (tc) is 1515.08 seconds, indicating the time when the output enters within ±5% of the steady-state value and does not leave again.
In the final stage, the curve did not converge to the steady-state value; there is a residual error of 1.75%.
Suggested new PID parameters: P=3, I=0.01, D=0.01

Experiment 4:
Current PID parameters and IAE: Kp=1.000, Ki=0.100, Kd=0.010, Integral Absolute Error (IAE): 101.70
Overshoot: After reaching the peak value, the curve shows an overshoot of 43.45%, indicating The overshoot is severe.
Oscillations: The curve contains 2 oscillations. The curve shows noticeable oscillations.
Attenuation Ratio: The attenuation ratio between the first and second peaks is 0.75.
Oscillation Period: The period of oscillation is 272.01 seconds.
The time constant (T) is 47.00 seconds, indicating the time taken for the output to reach 63.2% of the setpoint.
The settling time (tc) is 437.02 seconds, indicating the time when the output enters within ±5% of the steady-state value and does not leave again.
The control performance is good; the Steady-State Error is within 1%, its value is 0.00%.
Suggested new PID parameters: P=1.87, I=0.0085, D=29.74

Experiment 5:
Current PID parameters and IAE: Ki=0.100, Kd=0.010, Integral Absolute Error (IAE): 102.00
Overshoot: After reaching the peak value, the curve shows an overshoot of 36.29%, indicating The overshoot is severe.
Oscillations: The curve contains 1 oscillations. The oscillation behavior is normal.
Attenuation Ratio: The attenuation ratio between the first and second peaks is 0.77.
Oscillation Period: The period of oscillation is 318.01 seconds.
The time constant (T) is 57.00 seconds, indicating the time taken for the output to reach 63.2% of the setpoint.
The settling time (tc) is 374.02 seconds, indicating the time when the output enters within ±5% of the steady-state value and does not leave again.
The control performance is good; the Steady-State Error is within 1%, its value is 0.00%.
Suggested new PID parameters: P=1.62, I=0.023, D=8.45

Experiment 6:
Current PID parameters and IAE: Kp=1.000, Ki=0.100, Kd=0.010, Integral Absolute Error (IAE): 331.90
Overshoot: After reaching the peak value, the curve shows an overshoot of 54.41%, indicating The overshoot is severe.
Oscillations: The curve contains 3 oscillations. The curve shows noticeable oscillations.
Attenuation Ratio: The attenuation ratio between the first and second peaks is 0.75.
Oscillation Period: The period of oscillation is 640.03 seconds.
The time constant (T) is 121.01 seconds, indicating the time taken for the output to reach 63.2% of the setpoint.
The settling time (tc) is 1372.07 seconds, indicating the time when the output enters within ±5% of the steady-state value and does not leave again.
In the final stage, the curve did not converge to the steady-state value; there is a residual error of 2.26%.
Suggested new PID parameters: P=3.79, I=0.021, D=48.5

Experiment 7:
Current PID parameters and IAE: Kp=1.000, Ki=0.100, Kd=0.010, Integral Absolute Error (IAE): 221.58
Overshoot: After reaching the peak value, the curve shows an overshoot of 38.56%, indicating The overshoot is severe.
Oscillations: The curve contains 2 oscillations. The curve shows noticeable oscillations.
Attenuation Ratio: The attenuation ratio between the first and second peaks is 0.76.
Oscillation Period: The period of oscillation is 627.03 seconds.
The time constant (T) is 122.01 seconds, indicating the time taken for the output to reach 63.2% of the setpoint.
The settling time (tc) is 981.05 seconds, indicating the time when the output enters within ±5% of the steady-state value and does not leave again.
The control performance is good; the Steady-State Error is within 1%, its value is 0.23%.
Suggested new PID parameters: P=6.11, I=0.061, D=44.23

Experiment 8:
Current PID parameters and IAE: Kp=1.000, Ki=0.100, Kd=0.010, Integral Absolute Error (IAE): 285.81
Overshoot: After reaching the peak value, the curve shows an overshoot of 59.16%, indicating The overshoot is severe.
Oscillations: The curve contains 4 oscillations. Oscillation is very violent and the curve fluctuates greatly.
Attenuation Ratio: The attenuation ratio between the first and second peaks is 0.76.
Oscillation Period: The period of oscillation is 480.03 seconds.
The time constant (T) is 88.00 seconds, indicating the time taken for the output to reach 63.2% of the setpoint.
The settling time (tc) is 1253.06 seconds, indicating the time when the output enters within ±5% of the steady-state value and does not leave again.
In the final stage, the curve did not converge to the steady-state value; there is a residual error of 1.17%.
Suggested new PID parameters: P=2.04, I=0.011, D=27.62

Experiment 9:
Current PID parameters and IAE: Kp=1.000, Ki=0.100, Kd=0.010, Integral Absolute Error (IAE): 365.30
Overshoot: After reaching the peak value, the curve shows an overshoot of 56.33%, indicating The overshoot is severe.
Oscillations: The curve contains 4 oscillations. Oscillation is very violent and the curve fluctuates greatly.
Attenuation Ratio: The attenuation ratio between the first and second peaks is 0.75.
Oscillation Period: The period of oscillation is 674.03 seconds.
The time constant (T) is 127.01 seconds, indicating the time taken for the output to reach 63.2% of the setpoint.
The settling time (tc) is 1725.09 seconds, indicating the time when the output enters within ±5% of the steady-state value and does not leave again.
In the final stage, the curve did not converge to the steady-state value; there is a residual error of 2.42%.
Suggested new PID parameters: P=3.78, I=0.019, D=54.15

Experiment 10:
Current PID parameters and IAE: Kp=1.000, Ki=0.100, Kd=0.010, Integral Absolute Error (IAE): 76.29
Overshoot: After reaching the peak value, the curve shows an overshoot of 36.84%, indicating The overshoot is severe.
Oscillations: The curve contains 1 oscillations. The oscillation behavior is normal.
Attenuation Ratio: The attenuation ratio between the first and second peaks is 0.76.
Oscillation Period: The period of oscillation is 242.01 seconds.
The time constant (T) is 41.00 seconds, indicating the time taken for the output to reach 63.2% of the setpoint.
The settling time (tc) is 283.01 seconds, indicating the time when the output enters within ±5% of the steady-state value and does not leave again.
The control performance is good; the Steady-State Error is within 1%, its value is 0.00%.
Suggested new PID parameters: P=1.74, I=0.034, D=6.54
"""

SECOND_ORDER_DEFAULT_DEMONSTRATION = """Experiment 1:
Overshoot: After reaching the peak value, the curve shows an overshoot of 65.93%, indicating The overshoot is severe.
Oscillations: The curve contains 5 oscillations. Oscillation is very violent and the curve fluctuates greatly.
Attenuation Ratio: The attenuation ratio between the first and second peaks is 0.76.
Oscillation Period: The period of oscillation is 77.01 seconds.
The time constant (T) is 17.00 seconds, indicating the time taken for the output to reach 63.2% of the setpoint.
The settling time (tc) is 242.01 seconds, indicating the time when the output enters within ±5% of the steady-state value and does not leave again.
The control performance is good; the Steady-State Error is within 1%, its value is 0.00%.
Suggested new PID parameters: Kp=0.613, Ki=0.012, Kd=4.456

Experiment 2:
Overshoot: After reaching the peak value, the curve shows an overshoot of 27.10%, indicating The overshoot is relatively large.
Oscillations: The curve contains 1 oscillations. The oscillation behavior is normal.
Attenuation Ratio: The attenuation ratio between the first and second peaks is 0.80.
Oscillation Period: The period of oscillation is 423.02 seconds.
The time constant (T) is 100.01 seconds, indicating the time taken for the output to reach 63.2% of the setpoint.
The settling time (tc) is 495.02 seconds, indicating the time when the output enters within ±5% of the steady-state value and does not leave again.
The control performance is good; the Steady-State Error is within 1%, its value is 0.00%.
Suggested new PID parameters: Kp=15.493, Ki=0.182, Kd=236.642

Experiment 3:
Overshoot: After reaching the peak value, the curve shows an overshoot of 506.99%, indicating The overshoot is extremely severe, reaching 100% or more.
Oscillations: The curve contains 27 oscillations. Oscillation is very violent and the curve fluctuates greatly.
Attenuation Ratio: The attenuation ratio between the first and second peaks is 1.07.
Oscillation Period: The period of oscillation is 138.01 seconds.
The time constant (T) is 31.00 seconds, indicating the time taken for the output to reach 63.2% of the setpoint.
The system output does not settle within ±5% of the steady-state value.
The system has a significant deviation from the steady-state value; there is a large residual error of 178.46%.
Suggested new PID parameters: Kp=1.557, Ki=0.019, Kd=25.043

Experiment 4:
Overshoot: After reaching the peak value, the curve shows an overshoot of 77.83%, indicating The overshoot is severe.
Oscillations: The curve contains 16 oscillations. Oscillation is very violent and the curve fluctuates greatly.
Attenuation Ratio: The attenuation ratio between the first and second peaks is 0.87.
Oscillation Period: The period of oscillation is 90.01 seconds.
The time constant (T) is 20.00 seconds, indicating the time taken for the output to reach 63.2% of the setpoint.
The settling time (tc) is 770.04 seconds, indicating the time when the output enters within ±5% of the steady-state value and does not leave again.
The control performance is good; the Steady-State Error is within 1%, its value is 0.00%.
Suggested new PID parameters: Kp=0.755, Ki=0.015, Kd=8.091

Experiment 5:
Overshoot: After reaching the peak value, the curve shows an overshoot of 87.96%, indicating The overshoot is severe.
Oscillations: The curve contains 9 oscillations. Oscillation is very violent and the curve fluctuates greatly.
Attenuation Ratio: The attenuation ratio between the first and second peaks is 1.00.
Oscillation Period: The period of oscillation is 387.02 seconds.
The time constant (T) is 95.00 seconds, indicating the time taken for the output to reach 63.2% of the setpoint.
The system output does not settle within ±5% of the steady-state value.
The system has a significant deviation from the steady-state value; there is a large residual error of 77.59%.
Suggested new PID parameters: Kp=12.309, Ki=0.085, Kd=434.159

Experiment 6:
Overshoot: After reaching the peak value, the curve shows an overshoot of 49.07%, indicating The overshoot is severe.
Oscillations: The curve contains 2 oscillations. The curve shows noticeable oscillations.
Attenuation Ratio: The attenuation ratio between the first and second peaks is 0.73.
Oscillation Period: The period of oscillation is 124.01 seconds.
The time constant (T) is 23.00 seconds, indicating the time taken for the output to reach 63.2% of the setpoint.
The settling time (tc) is 204.01 seconds, indicating the time when the output enters within ±5% of the steady-state value and does not leave again.
The control performance is good; the Steady-State Error is within 1%, its value is 0.00%.
Suggested new PID parameters: Kp=1.367, Ki=0.018, Kd=5.995

Experiment 7:
Overshoot: After reaching the peak value, the curve shows an overshoot of 84.93%, indicating The overshoot is severe.
Oscillations: The curve contains 14 oscillations. Oscillation is very violent and the curve fluctuates greatly.
Attenuation Ratio: The attenuation ratio between the first and second peaks is 0.94.
Oscillation Period: The period of oscillation is 255.01 seconds.
The time constant (T) is 59.00 seconds, indicating the time taken for the output to reach 63.2% of the setpoint.
The system output does not settle within ±5% of the steady-state value.
The system has a significant deviation from the steady-state value; there is a large residual error of 20.99%.
Suggested new PID parameters: Kp=5.491, Ki=0.046, Kd=114.326

Experiment 8:
Overshoot: After reaching the peak value, the curve shows an overshoot of 82.70%, indicating The overshoot is severe.
Oscillations: The curve contains 27 oscillations. Oscillation is very violent and the curve fluctuates greatly.
Attenuation Ratio: The attenuation ratio between the first and second peaks is 0.92.
Oscillation Period: The period of oscillation is 94.01 seconds.
The time constant (T) is 21.00 seconds, indicating the time taken for the output to reach 63.2% of the setpoint.
The settling time (tc) is 1321.07 seconds, indicating the time when the output enters within ±5% of the steady-state value and does not leave again.
The control performance is good; the Steady-State Error is within 1%, its value is 0.02%.
Suggested new PID parameters: Kp=0.793, Ki=0.015, Kd=9.093

Experiment 9:
Overshoot: After reaching the peak value, the curve shows an overshoot of 82.97%, indicating The overshoot is severe.
Oscillations: The curve contains 16 oscillations. Oscillation is very violent and the curve fluctuates greatly.
Attenuation Ratio: The attenuation ratio between the first and second peaks is 0.91.
Oscillation Period: The period of oscillation is 223.01 seconds.
The time constant (T) is 51.00 seconds, indicating the time taken for the output to reach 63.2% of the setpoint.
The system output does not settle within ±5% of the steady-state value.
In the final stage, the curve did not converge to the steady-state value; there is a residual error of 3.85%.
Suggested new PID parameters: Kp=4.250, Ki=0.037, Kd=73.498

Experiment 10:
Overshoot: After reaching the peak value, the curve shows an overshoot of 78.98%, indicating The overshoot is severe.
Oscillations: The curve contains 20 oscillations. Oscillation is very violent and the curve fluctuates greatly.
Attenuation Ratio: The attenuation ratio between the first and second peaks is 0.90.
Oscillation Period: The period of oscillation is 186.00 seconds.
The time constant (T) is 43.00 seconds, indicating the time taken for the output to reach 63.2% of the setpoint.
The settling time (tc) is 1984.10 seconds, indicating the time when the output enters within ±5% of the steady-state value and does not leave again.
In the final stage, the curve did not converge to the steady-state value; there is a residual error of 3.22%.
Suggested new PID parameters: Kp=3.007, Ki=0.037, Kd=51.641
"""

DEFAULT_DEMONSTRATIONS = {
    "first_order": FIRST_ORDER_DEFAULT_DEMONSTRATION,
    "second_order": SECOND_ORDER_DEFAULT_DEMONSTRATION,
}


def get_default_demonstration(system: str) -> str:
    try:
        return DEFAULT_DEMONSTRATIONS[system]
    except KeyError as error:
        supported = ", ".join(sorted(DEFAULT_DEMONSTRATIONS))
        raise ValueError(
            f"Unsupported default demonstration system: {system}. Supported: {supported}"
        ) from error
